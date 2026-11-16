"""Unpack one persona workspace from the pinned Workspace-Bench zip into --dest.

The zips are about 18.7 GB each, so only the persona's folder is read, with
HTTP range requests (remotezip) on several threads. Afterwards a snapshot of
every file's size and mtime is written to /opt/wsb/before.json; the verifier
uses it, like upstream's runner, to find outputs written outside model_output.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from remotezip import RemoteZip

REPO = "Workspace-Bench/Workspace-Bench-Workspaces"
REVISION = "e245d63bfa20cfdb708cd8e78145ffb087155857"
SNAPSHOT = "/opt/wsb/before.json"
SKIPPED_DIRS = {"model_output", ".workspace_data", ".git", ".cache", "__pycache__", "node_modules"}

_local = threading.local()


def _zip(url: str) -> RemoteZip:
    if getattr(_local, "zip", None) is None:
        _local.zip = RemoteZip(url)
    return _local.zip


def _extract(url: str, name: str, out: str) -> int:
    for attempt in range(5):
        try:
            with _zip(url).open(name) as src, open(out, "wb") as f:
                shutil.copyfileobj(src, f, 1 << 20)
            return os.path.getsize(out)
        except Exception:
            _local.zip = None
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    return 0


def snapshot(work_dir: str) -> dict:
    """Same walk as upstream agent_runner._snapshot_work_dir_files."""
    out = {}
    target = os.path.join(work_dir, "model_output")
    for dirpath, dirnames, filenames in os.walk(work_dir):
        if os.path.abspath(dirpath).startswith(target):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in SKIPPED_DIRS and not d.startswith(".")]
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            try:
                st = os.stat(path)
            except OSError:
                continue
            out[os.path.relpath(path, work_dir)] = [int(st.st_size), float(st.st_mtime)]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--language", choices=["en", "cn"], required=True)
    p.add_argument("--persona", required=True, help="Folder inside the zip, e.g. Research_Workdir")
    p.add_argument("--dest", required=True)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()

    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/filesys_{a.language}.zip"
    prefix = a.persona.rstrip("/") + "/"
    dest = os.path.realpath(a.dest)
    with RemoteZip(url) as z:
        infos = [i for i in z.infolist() if i.filename.startswith(prefix)]
    if not infos:
        raise SystemExit(f"{prefix} not found in {url}")

    jobs = []
    for i in infos:
        rel = i.filename[len(prefix):]
        if not rel:
            continue
        out = os.path.realpath(os.path.join(dest, rel))
        if not out.startswith(dest + os.sep):
            raise SystemExit(f"unsafe path in zip: {i.filename}")
        if i.is_dir():
            os.makedirs(out, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            jobs.append((i.filename, out, i.file_size))

    started = time.time()
    total = 0
    with ThreadPoolExecutor(a.workers) as pool:
        for n in pool.map(lambda j: _extract(url, j[0], j[1]), jobs):
            total += n
    expected = sum(j[2] for j in jobs)
    if total != expected:
        raise SystemExit(f"size mismatch: wrote {total} bytes, zip lists {expected}")
    print(f"{a.persona} ({a.language}): {len(jobs)} files, {total / 1e9:.2f} GB in {time.time() - started:.0f}s")

    os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
    with open(SNAPSHOT, "w") as f:
        json.dump({"created": time.time(), "files": snapshot(dest)}, f)


if __name__ == "__main__":
    main()
