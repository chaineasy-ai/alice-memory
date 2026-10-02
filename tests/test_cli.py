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


def test_contract_cli_forms_ci034(tmp_path):
    """CI-034 护栏：§4.6 文档化的调用形式必须逐条成功（防止契约-实现漂移）。"""
    # add <title> [body] ... 位置参数 + 子命令后 --json
    add = mem(tmp_path, "add", "记忆分层设计", "core/semantic/episodic", "--layer", "semantic",
              "--type", "note", "--tags", "memory,design", "--importance", "8",
              "--source", "base/cland-crawler#230", "--source-type", "issue", "--json")
    mid = add["note"]["id"]
    assert add["schema_version"] == "mem.add.v1"

    add2 = mem(tmp_path, "add", "第二篇", "--link", mid, "--summary", "摘要", "--json")
    mid2 = add2["note"]["id"]

    # search "<query>" -k N --layer --tag --mode lexical --all --json
    res = mem(tmp_path, "search", "记忆 分层", "-k", "5", "--layer", "semantic",
              "--tag", "memory", "--mode", "lexical", "--all", "--json")
    assert res["schema_version"] == "mem.search.v1"
    assert res["count"] >= 1, res          # B1 修复后多词 CJK 应命中
    r0 = res["results"][0]
    assert set(r0["why"]) == {"rel", "recency", "importance", "graph"}
    assert "source" in r0

    # get <id> --json
    assert mem(tmp_path, "get", mid, "--json")["note"]["id"] == mid

    # update <id> --title --body --add-tag a,b --importance --status --source
    up = mem(tmp_path, "update", mid2, "--title", "第二篇改名", "--body", "新正文",
             "--add-tag", "a,b", "--importance", "6", "--status", "active",
             "--source", "x", "--json")
    assert up["schema_version"] == "mem.update.v1"
    assert "a" in up["note"]["tags"] and "b" in up["note"]["tags"]

    # link <id> <to-id>
    assert mem(tmp_path, "link", mid, mid2, "--json")["id"] == mid

    # gc --apply --stale-days 180 --min-importance 2 --json
    gc = mem(tmp_path, "gc", "--stale-days", "180", "--min-importance", "2", "--json")
    assert gc["schema_version"] == "mem.gc.v1" and gc["dry_run"] is True

    # reindex / stats / context
    assert mem(tmp_path, "reindex", "--json")["schema_version"] == "mem.reindex.v1"
    assert mem(tmp_path, "stats", "--json")["schema_version"] == "mem.stats.v1"
    ctx = mem(tmp_path, "context", "记忆分层", "--budget", "2000",
              "--budget-unit", "tokens", "-k", "5", "--json")
    assert ctx["schema_version"] == "mem.context.v1"


def test_add_file_title_fallback_i6(tmp_path):
    # 1) frontmatter title
    f1 = tmp_path / "a.md"
    f1.write_text("---\ntitle: FM 标题\n---\n正文一\n", encoding="utf-8")
    assert mem(tmp_path, "add", "--file", str(f1))["note"]["title"] == "FM 标题"
    # 2) name-only（旧实现 exit 2）
    f2 = tmp_path / "b.md"
    f2.write_text("---\nname: Name 标题\n---\n正文二\n", encoding="utf-8")
    assert mem(tmp_path, "add", "--file", str(f2))["note"]["title"] == "Name 标题"
    # 3) 无 meta + H1 → H1
    f3 = tmp_path / "c.md"
    f3.write_text("# H1 标题\n正文三\n", encoding="utf-8")
    assert mem(tmp_path, "add", "--file", str(f3))["note"]["title"] == "H1 标题"
    # 4) 无 meta 无 H1 → 文件名，且**不得取正文**
    f4 = tmp_path / "myfile.md"
    f4.write_text("这是正文第一行，不应作为标题\n第二行\n", encoding="utf-8")
    t4 = mem(tmp_path, "add", "--file", str(f4))["note"]["title"]
    assert t4 == "myfile" and t4 != "这是正文第一行，不应作为标题"


def test_access_log_recall_and_adopt(tmp_path):
    a = mem(tmp_path, "add", "--title", "检索设计", "--text", "FTS5 trigram 词法")
    mid = a["note"]["id"]
    mem(tmp_path, "search", "trigram", "-k", "3", "--touch")
    log = (tmp_path / ".mem" / "access.log").read_text(encoding="utf-8").strip().splitlines()
    import json as _j
    rec = [_j.loads(x) for x in log]
    assert rec and rec[0]["event"] == "recall" and rec[0]["adopted"] is False
    assert "rank" in rec[0] and "components" in rec[0]
    # 采纳信号
    ad = mem(tmp_path, "adopt", mid, "--rank", "1", "--query", "trigram")
    assert ad["schema_version"] == "mem.adopt.v1" and ad["adopted"] is True
    log2 = (tmp_path / ".mem" / "access.log").read_text(encoding="utf-8")
    assert '"event": "adopt"' in log2 and '"adopted": true' in log2


def test_adopt_not_found_exit3(tmp_path):
    p = subprocess.run([sys.executable, "-m", "alice_memory", "adopt", "nope"],
                       env={**os.environ, "PYTHONPATH": str(REPO / "src"),
                            "MEM_HOME": str(tmp_path)}, capture_output=True, text=True)
    assert p.returncode == 3
