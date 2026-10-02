"""CLI 端到端测试（契约 §4.6：子命令 + JSON schema_version + 索引可重建）。"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def mem(home, *args, as_agent="hopper", check=True):
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "MEM_HOME": str(home),
           "MEM_AGENT": as_agent}
    p = subprocess.run([sys.executable, "-m", "alice_memory", *args],
                       env=env, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise AssertionError(f"mem {' '.join(args)} failed: {p.stderr}")
    return json.loads(p.stdout) if p.stdout.strip() else {}


def test_cli_lifecycle(tmp_path):
    assert mem(tmp_path, "init")["schema_version"] == "mem.init.v1"
    add = mem(tmp_path, "add", "--title", "检索设计", "--text", "FTS5 trigram 词法检索",
              "--layer", "semantic", "--tags", "retrieval")
    assert add["schema_version"] == "mem.add.v1" and add["event"] == "added"
    mid = add["note"]["id"]
    assert add["note"]["namespace"] == "agents/hopper"

    res = mem(tmp_path, "search", "trigram", "--k", "5")
    assert res["schema_version"] == "mem.search.v1"
    assert res["results"] and res["results"][0]["id"] == mid
    assert set(res["results"][0]) >= {"id", "score", "path", "title", "layer",
                                      "namespace", "snippet", "why"}

    got = mem(tmp_path, "get", mid)
    assert got["note"]["id"] == mid

    upd = mem(tmp_path, "update", mid, "--append", "补充：<3 字用 LIKE 兜底")
    assert upd["schema_version"] == "mem.update.v1"

    link = mem(tmp_path, "link", mid, mid, check=False)  # 自链接：允许
    assert link is not None


def test_cli_reindex_rebuild(tmp_path):
    a = mem(tmp_path, "add", "--title", "可重建", "--text", "索引派生可重建")
    mid = a["note"]["id"]
    # 删除派生索引
    (tmp_path / ".mem" / "index.sqlite").unlink()
    r = mem(tmp_path, "reindex")
    assert r["indexed"] == 1
    assert mem(tmp_path, "search", "可重建")["results"][0]["id"] == mid


def test_cli_shared_namespace(tmp_path):
    a = mem(tmp_path, "add", "--title", "共享知识", "--text", "team shared fact",
            "--namespace", "shared")
    assert a["note"]["namespace"] == "shared"
    # 另一 agent（读全部命名空间）能看到共享层
    res = mem(tmp_path, "search", "shared fact", as_agent="curie")
    assert res["count"] >= 1


def test_stats_fixed_fields_i5(tmp_path):
    mem(tmp_path, "add", "--title", "a", "--text", "b")
    st = mem(tmp_path, "stats")
    assert st["schema_version"] == "mem.stats.v1"
    for k in ("total", "by_namespace", "by_layer", "by_status", "gc_candidates",
              "index_schema", "vectors", "links"):
        assert k in st, k


def test_idempotency_key_i4(tmp_path):
    a = mem(tmp_path, "add", "--title", "evt", "--text", "v1", "--idempotency-key", "K1")
    b = mem(tmp_path, "add", "--title", "evt", "--text", "v2", "--idempotency-key", "K1")
    assert a["event"] == "added" and b["event"] == "duplicate"
    assert a["note"]["id"] == b["note"]["id"]


def test_context_delimiter_i2(tmp_path):
    mem(tmp_path, "add", "--title", "检索设计", "--text", "FTS5 trigram 词法检索",
        "--source", "docs/x.md")
    res = mem(tmp_path, "search", "trigram", "--context", "--budget-tokens", "500")
    assert "<!-- mem: id=" in res["context"]
    assert "source=docs/x.md" in res["context"]


def test_exit_code_not_found_i3(tmp_path):
    p = subprocess.run([sys.executable, "-m", "alice_memory", "get", "nope"],
                       env={**os.environ, "PYTHONPATH": str(REPO / "src"),
                            "MEM_HOME": str(tmp_path)}, capture_output=True, text=True)
    assert p.returncode == 3
