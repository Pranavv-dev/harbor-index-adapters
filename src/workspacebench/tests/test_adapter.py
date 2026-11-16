"""Generation tests on a tiny fake task repo (no network)."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from workspacebench import adapter as ad
from workspacebench.prompt import wrap_prompt


def _fake_repo(root: Path, language: str, ids=range(1, 389)) -> Path:
    tasks = root / ad.TASK_DIRS[language]
    personas = list(ad.PERSONA_DIRS)
    for i in ids:
        d = tasks / str(i)
        (d / "data").mkdir(parents=True)
        (d / "data" / f"abc_in{i}.txt").write_text("input")
        meta = {
            "id": str(i), "language": language, "persona": personas[i % len(personas)],
            "task": f"Task {i}: write `out{i}.md`.", "task_diff": "medium",
            "output_files": [f"out{i}.md"], "rubrics": ["r1", "r2"], "rubric_types": ["Basic Evaluation"] * 2,
            "data_manifest": [{"filename": f"in{i}.txt", "stored_relpath": f"data/abc_in{i}.txt"}],
        }
        (d / "metadata.json").write_text(json.dumps(meta))
    return root


@pytest.fixture
def repo(tmp_path):
    _fake_repo(tmp_path / "repo", "en")
    _fake_repo(tmp_path / "repo", "cn")
    return tmp_path / "repo"


def test_full_split_generates_both_languages(repo, tmp_path):
    out = tmp_path / "out"
    ad.WorkspaceBenchAdapter(out, tasks_root=repo, limit=None).run()
    names = sorted(p.name for p in out.iterdir())
    assert len(names) == 776
    assert names[0] == "cn-001" and "en-388" in names


def test_parity_split_is_english_lite(repo, tmp_path):
    out = tmp_path / "out"
    ad.WorkspaceBenchAdapter(out, tasks_root=repo, split="parity").run()
    names = {p.name for p in out.iterdir()}
    assert len(names) == 100 == len(ad.LITE_IDS)
    assert names == {f"en-{i:03d}" for i in ad.LITE_IDS}


def test_task_contents(repo, tmp_path):
    out = tmp_path / "out"
    oracle = tmp_path / "oracle" / "en-033"
    oracle.mkdir(parents=True)
    (oracle / "out33.md").write_text("answer")
    ad.WorkspaceBenchAdapter(out, tasks_root=repo, task_ids=["en-033"], oracle_dir=tmp_path / "oracle").run()
    t = out / "en-033"

    toml = tomllib.loads((t / "task.toml").read_text())
    assert toml["task"]["name"] == "workspacebench/en-033"
    persona = ad.PERSONA_DIRS[list(ad.PERSONA_DIRS)[33 % 5]]
    assert toml["environment"]["storage_mb"] == ad.PERSONA_STORAGE_MB[persona]

    instruction = (t / "instruction.md").read_text()
    assert instruction == wrap_prompt(prompt="Task 33: write `out33.md`.", work_dir="/app",
                                      task_target_output_dir="model_output", language="en")
    assert (t / "tests" / "instruction.md").read_text() == instruction
    assert f"--persona {persona}" in (t / "environment" / "Dockerfile").read_text()
    assert (t / "tests" / "task" / "data" / "abc_in33.txt").exists()
    assert json.loads((t / "tests" / "task" / "metadata.json").read_text())["id"] == "33"
    assert (t / "solution" / "output" / "out33.md").read_text() == "answer"
    assert "{" not in (t / "task.toml").read_text().split("[verifier.env]")[0].replace("{ name", "")


def test_prompt_matches_upstream_shape():
    en = wrap_prompt(prompt=" do it ", work_dir="/app", task_target_output_dir="model_output", language="en")
    assert en.startswith("[Important Requirement 1: Working Directory]\n")
    assert "Place all output files under: /app/model_output\n\ndo it\n" in en
    cn = wrap_prompt(prompt="做", work_dir="/app", task_target_output_dir="model_output", language="cn")
    assert "/app/model_output下" in cn and cn.rstrip().endswith("['output/a.txt','report.md']")


def test_wrong_task_count_fails(tmp_path):
    _fake_repo(tmp_path / "small", "en", ids=range(1, 10))
    with pytest.raises(RuntimeError, match="Expected 388"):
        ad.WorkspaceBenchAdapter(tmp_path / "out", tasks_root=tmp_path / "small", languages=["en"]).run()
