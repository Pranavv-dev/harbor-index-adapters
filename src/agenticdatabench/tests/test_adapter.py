"""Generation tests on a tiny fake source repo (no network)."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from agenticdatabench import adapter as ad

CHART = {
    "id": "agriculture_14",
    "question": "Plot it and save output.png; save the max to output.csv.",
    "domain": "agriculture",
    "output_file_name": ["output.png", "output.csv"],
    "gold_file_name": ["result.png", "result.csv"],
    "eval_func": [
        "compare_image('output.png', 'result.png', calculate_columns=['type'])",
        "compare_csv(output_file_name='output.csv', gold_file_name='result.csv', ignore_order=True)",
    ],
}
PLAIN = {
    "id": "loan_risk_5",
    "question": "Summarise input.csv into output.csv.",
    "domain": "loan_risk/loan_risk_5",
    "output_file_name": ["output.csv"],
    "gold_file_name": ["result.csv"],
    "eval_func": "compare_csv(output_file_name='output.csv', gold_file_name='result.csv')",
}


def _filler(i: int) -> dict:
    return {
        "id": f"tourism_{i:03d}",
        "question": "q",
        "domain": "tourism",
        "output_file_name": ["output.csv"],
        "gold_file_name": ["result.csv"],
        "eval_func": "compare_csv('output.csv', 'result.csv')",
    }


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "AgenticDataBench"
    tasks = [CHART, PLAIN] + [_filler(i) for i in range(244)]
    (root / "testbed" / "tasks").mkdir(parents=True)
    (root / "testbed" / "tasks" / "dev.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    for t in tasks:
        gold = root / "testbed" / "gold" / t["id"]
        gold.mkdir(parents=True)
        for g in ad.as_list(t["gold_file_name"]):
            (gold / g).write_text(f"gold {g}")
    chart_gold = root / "testbed" / "gold" / "agriculture_14"
    (chart_gold / "result.json").write_text("{}")
    (chart_gold / "result.npy").write_bytes(b"npy")
    return root


@pytest.fixture
def data(monkeypatch):
    files = {
        "agriculture": [ad.DataFile("datasets/agriculture/a b.csv", "a b.csv", 300 * 2**20, "sha256", "ab")],
        "loan_risk/loan_risk_5": [
            ad.DataFile("datasets/loan_risk/loan_risk_5/input.csv", "input.csv", 2 * 2**30, "sha256", "cd")
        ],
        "tourism": [ad.DataFile("datasets/tourism/t.csv", "t.csv", 10, "gitsha1", "ef")],
    }
    monkeypatch.setattr(ad.AgenticDataBenchAdapter, "_data_files", staticmethod(lambda: files))
    return files


def test_generates_every_task_not_excluded(source, data, tmp_path):
    out = tmp_path / "out"
    ad.AgenticDataBenchAdapter(out, source_dir=source).run()
    names = {p.name for p in out.iterdir()}
    all_ids = {json.loads(line)["id"] for line in (source / "testbed" / "tasks" / "dev.jsonl").read_text().splitlines()}
    assert names == all_ids - ad.EXCLUDED_TASKS.keys()


def test_instruction_is_upstream_prompt(source, data, tmp_path):
    out = tmp_path / "out"
    ad.AgenticDataBenchAdapter(out, source_dir=source, task_ids=["agriculture_14", "loan_risk_5"]).run()
    plain = (out / "loan_risk_5" / "instruction.md").read_text()
    assert plain == (
        "Summarise input.csv into output.csv.\n\nYou are working in the directory: /workspace."
        " All required data files are available in this directory."
        " Complete the task and ensure all output files are saved in this directory.\n"
    )
    chart = (out / "agriculture_14" / "instruction.md").read_text()
    assert "saved in this directory.\n### Plotting (REQUIRED)" in chart
    assert "Use ONLY these file names:\noutput.png\n" in chart
    assert 'Plotprocess.plot_process(fig, "output.png")' in chart


def test_chart_task_files(source, data, tmp_path):
    out = tmp_path / "out"
    ad.AgenticDataBenchAdapter(out, source_dir=source, task_ids=["agriculture_14"]).run()
    t = out / "agriculture_14"
    assert (t / "environment" / "image.py").exists()
    assert "COPY image.py /workspace/image.py" in (t / "environment" / "Dockerfile").read_text()
    sol = t / "solution" / "output"
    assert sorted(p.name for p in sol.iterdir()) == ["output.csv", "output.json", "output.npy", "output.png"]
    assert (sol / "output.csv").read_text() == "gold result.csv"
    assert sorted(p.name for p in (t / "tests" / "gold").iterdir()) == [
        "result.csv", "result.json", "result.npy", "result.png"
    ]
    cfg = json.loads((t / "tests" / "task_config.json").read_text())
    assert cfg["eval_func"] == CHART["eval_func"] and "question" not in cfg
    toml = tomllib.loads((t / "task.toml").read_text())
    assert toml["task"]["name"] == "agenticdatabench/agenticdatabench__agriculture_14"
    assert toml["task"]["authors"][0] == {"name": "Zhaoyan Sun"}
    assert toml["metadata"]["output_types"] == ["csv", "png"]
    assert toml["environment"]["memory_mb"] == 8192


def test_plain_task_files(source, data, tmp_path):
    out = tmp_path / "out"
    ad.AgenticDataBenchAdapter(out, source_dir=source, task_ids=["loan_risk_5"]).run()
    t = out / "loan_risk_5"
    assert not (t / "environment" / "image.py").exists()
    dockerfile = (t / "environment" / "Dockerfile").read_text()
    assert "image.py" not in dockerfile and "{" not in dockerfile
    manifest = (t / "environment" / "data_manifest.tsv").read_text().splitlines()
    assert manifest == [
        f"{ad.DATA_REVISION}\tdatasets/loan_risk/loan_risk_5/input.csv\tinput.csv\t{2 * 2**30}\tsha256\tcd"
    ]
    toml = tomllib.loads((t / "task.toml").read_text())
    assert toml["metadata"]["domain"] == "loan_risk"
    assert toml["environment"]["memory_mb"] == 16384
    assert toml["environment"]["storage_mb"] >= 2 * 2048 + 8192


def test_rendered_files_have_no_placeholders(source, data, tmp_path):
    out = tmp_path / "out"
    ad.AgenticDataBenchAdapter(out, source_dir=source, task_ids=["agriculture_14", "loan_risk_5"]).run()
    for t in out.iterdir():
        for name in ("task.toml", "environment/Dockerfile"):
            text = (t / name).read_text()
            for key in ("{task_id}", "{domain", "{image_helper}", "{memory_mb}", "{storage_mb}", "{output_types}"):
                assert key not in text, (t.name, name, key)


def test_overwrite_and_skip(source, data, tmp_path):
    out = tmp_path / "out"
    ad.AgenticDataBenchAdapter(out, source_dir=source, task_ids=["loan_risk_5"]).run()
    marker = out / "loan_risk_5" / "marker"
    marker.write_text("x")
    ad.AgenticDataBenchAdapter(out, source_dir=source, task_ids=["loan_risk_5"]).run()
    assert marker.exists()
    ad.AgenticDataBenchAdapter(out, source_dir=source, task_ids=["loan_risk_5"], overwrite=True).run()
    assert not marker.exists()

