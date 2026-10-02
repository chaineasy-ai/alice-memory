"""核心单元测试：存储 / 索引 / 生命周期 / 召回（契约 §4.6/§4.7）。"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from alice_memory import (MemoryIndex, MemoryManager, MemoryStore, Note,
                          RecallEngine, content_hash)


def fresh(tmp_path):
    store = MemoryStore(tmp_path)
    store.init()
    index = MemoryIndex(store.indexdir / "index.sqlite")
    mgr = MemoryManager(store, index)
    recall = RecallEngine(store, index)
    return store, index, mgr, recall


def test_init_layout(tmp_path):
    store, *_ = fresh(tmp_path)
    for layer in ("core", "semantic", "episodic", "entity", "archive"):
        assert (store.memdir / layer).is_dir()
    assert (tmp_path / ".mem").is_dir()


def test_add_get_roundtrip(tmp_path):
    store, index, mgr, _ = fresh(tmp_path)
    r = mgr.add("bge-m3 中文选型", "bge-m3 在中文上完胜 nomic", layer="semantic",
                tags=["embedding", "rag"], source="alice-research-hub")
    assert r.status == "added"
    got = store.find_by_id(r.note.id)
    assert got.title == "bge-m3 中文选型"
    assert "embedding" in got.tags
    # 磁盘上是带 frontmatter 的 md
    assert Path(r.note.path).read_text(encoding="utf-8").startswith("---")
    # 索引可查
    assert index.get_meta(r.note.id)["layer"] == "semantic"


def test_dedup_noop(tmp_path):
    store, index, mgr, _ = fresh(tmp_path)
    a = mgr.add("same title", "same body")
    b = mgr.add("same title", "same body")
    assert a.status == "added" and b.status == "duplicate"
    assert a.note.id == b.note.id
    assert len(list(store.iter_notes())) == 1


def test_supersede(tmp_path):
    store, index, mgr, _ = fresh(tmp_path)
    a = mgr.add("决策", "旧结论")
    b = mgr.add("决策", "新结论")
    assert b.status == "superseded"
    old = store.find_by_id(a.note.id)
    assert old.status == "superseded"
    assert old.superseded_by == b.note.id
    assert b.note.id in old.superseded_by


def test_update_link_and_neighbors(tmp_path):
    store, index, mgr, _ = fresh(tmp_path)
    a = mgr.add("A", "aaa")
    b = mgr.add("B", "bbb")
    mgr.link(a.note.id, b.note.id)
    assert b.note.id in store.find_by_id(a.note.id).links
    assert a.note.id in index.neighbors(b.note.id, "in")
    mgr.update(a.note.id, add_tags=["x"], importance=8)
    assert store.find_by_id(a.note.id).importance == 8


def test_archive_and_gc(tmp_path):
    store, index, mgr, _ = fresh(tmp_path)
    a = mgr.add("stale", "旧", importance=1)
    from alice_memory import GcPolicy
    cands = mgr.gc_candidates(GcPolicy(stale_days=0))
    assert any(c["id"] == a.note.id and c["reason"] == "stale" for c in cands)
    # dry-run 不改
    assert store.find_by_id(a.note.id).status == "active"
    mgr.gc(GcPolicy(stale_days=0), apply=True)
    got = store.find_by_id(a.note.id)
    assert got.status == "archived"
    assert got.layer == "archive"


def test_namespace_isolation(tmp_path):
    store, index, mgr, _ = fresh(tmp_path)
    a = mgr.add("shared note", "s", namespace="shared")
    b = mgr.add("curie note", "c", namespace="curie")
    assert a.note.path != b.note.path
    assert "agents/curie" in b.note.path
    assert store.namespace_of(b.note.path) == "agents/curie"
    assert store.namespace_of(a.note.path) == "shared"


def test_fts_lexical_search(tmp_path):
    store, index, mgr, recall = fresh(tmp_path)
    mgr.add("Knowledge graph memory", "temporal knowledge graph for agents")
    mgr.add("Vector retrieval", "embeddings and cosine similarity")
    hits = recall.search("knowledge graph", k=5, mode="lexical")
    assert hits and "Knowledge graph" in hits[0].note.title


def test_cjk_trigram_and_short_fallback(tmp_path):
    store, index, mgr, recall = fresh(tmp_path)
    mgr.add("向量检索设计", "使用 FTS5 trigram 做中文检索")
    # >=3 字：trigram
    assert recall.search("向量检索", k=5, mode="lexical")
    # <3 字：LIKE 兜底
    assert recall.search("向量", k=5, mode="lexical")


def test_recency_importance_ranking(tmp_path):
    store, index, mgr, recall = fresh(tmp_path)
    mgr.add("config note", "memory weighting", importance=1)
    hi = mgr.add("config memory", "memory weighting", importance=10)
    hits = recall.search("memory weighting", k=5, mode="lexical")
    assert hits[0].note.id == hi.note.id


def test_context_budget(tmp_path):
    store, index, mgr, recall = fresh(tmp_path)
    mgr.add("core identity", "I am hopper", layer="core", importance=10)
    mgr.add("topic", "long " * 500)
    ctx = recall.context("topic", token_budget=50, k=5)
    assert "核心记忆" in ctx
    assert len(ctx) < 4000


def test_reindex_rebuild(tmp_path):
    store, index, mgr, recall = fresh(tmp_path)
    mgr.add("rebuildable", "index is derived")
    index.close()
    # 删掉派生索引后重建
    (store.indexdir / "index.sqlite").unlink()
    index2 = MemoryIndex(store.indexdir / "index.sqlite")
    mgr2 = MemoryManager(store, index2)
    n = mgr2.reindex()
    assert n == 1
    assert RecallEngine(store, index2).search("rebuildable", k=5, mode="lexical")
    index2.close()


def test_concurrent_adds_same_namespace(tmp_path):
    """§4.7-D 并发验收：N 进程并发 mem add 同一 shared/ → 事后计数 = N。"""
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo / "src"), "MEM_HOME": str(tmp_path)}
    procs = []
    for i in range(6):
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "alice_memory", "add", "--text", f"并发事实 {i}",
             "--title", f"concurrent {i}", "--importance", "5"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    for p in procs:
        _, err = p.communicate(timeout=60)
        assert p.returncode == 0, err.decode()
    store = MemoryStore(tmp_path)
    store.init()
    assert len(list(store.iter_notes())) == 6


def test_bigram_tokenizer_spike(tmp_path):
    """#232 item4：bigram 可切，2 字/多词 CJK 走 BM25（免 LIKE 兜底）。"""
    store = MemoryStore(tmp_path)
    store.init()
    idx = MemoryIndex(store.indexdir / "index.sqlite", tokenizer="bigram")
    mgr = MemoryManager(store, idx)
    mgr.add("记忆分层设计", "core/semantic/episodic/entity/archive", tags=["memory"])
    mgr.add("向量检索", "embedding retrieval fusion")
    assert idx.search_fts("向量", 5)          # 2 字
    assert idx.search_fts("记忆 分层", 5)      # 多词 CJK
    assert idx.search_fts("memory 分层", 5)    # 混排
    idx.close()


def test_tokenizer_mismatch_guard_g1(tmp_path):
    """G1：索引 tokenizer 标识守卫——不 reindex 切 tokenizer 必须报错而非静默返回。"""
    store = MemoryStore(tmp_path); store.init()
    idx = MemoryIndex(store.indexdir / "index.sqlite", tokenizer="bigram")
    MemoryManager(store, idx).add("记忆分层", "core/semantic")
    assert idx.search_fts("记忆", 5)
    idx.close()
    idx2 = MemoryIndex(store.indexdir / "index.sqlite", tokenizer="trigram")
    try:
        idx2.search_fts("记忆", 5)
        raise AssertionError("should raise on tokenizer mismatch")
    except ValueError as e:
        assert "tokenizer" in str(e)
    # reindex 可切换
    MemoryManager(store, idx2).reindex()
    assert idx2.search_fts("记忆", 5)
    idx2.close()


class _FakeEmb:
    name = "fake"
    dim = 2

    def __init__(self, vec):
        self.vec = list(vec)

    def embed(self, text):
        return list(self.vec)

    def embed_batch(self, texts):
        return [list(self.vec) for _ in texts]


def test_query_type_routing_g2(tmp_path):
    from alice_memory.recall import RecallEngine, classify_query, HybridRetriever, LexicalRetriever
    assert classify_query("向量") == "short"
    assert classify_query("trigram") == "keyword"
    assert classify_query("解释一下 markdown 记忆系统的检索与召回设计") == "semantic"
    store = MemoryStore(tmp_path); store.init()
    idx = MemoryIndex(store.indexdir / "index.sqlite")
    idx.upsert(Note(id="a", title="记忆检索", body="markdown 记忆检索召回设计"), vector=[1.0, 0.0])
    eng = RecallEngine(store, idx, _FakeEmb([1.0, 0.0]))
    assert eng._retriever("auto", "向量").name == "lexical"           # short → 纯词法
    assert eng._retriever("auto", "解释一下 markdown 记忆系统的检索与召回设计").name == "hybrid"  # semantic → 向量
    assert eng._retriever("lexical", "解释一下 markdown 记忆系统的检索与召回设计").name == "lexical"
    idx.close()


def test_link_second_signal_g4(tmp_path):
    import math
    store = MemoryStore(tmp_path); store.init()
    idx = MemoryIndex(store.indexdir / "index.sqlite")
    # A 向量 [1,0]；候选 note 向量与之 cosine≈0.87 ∈[0.84,0.90)
    other = [0.87, math.sqrt(1 - 0.87 ** 2)]
    idx.upsert(Note(id="A", title="A", body="x", entities=["X"]), vector=[1.0, 0.0])
    idx.upsert(Note(id="B", title="B", body="x"), vector=[1.0, 0.0])   # 实体不重叠
    mgr = MemoryManager(store, idx, _FakeEmb(other))
    cand_overlap = Note(id="N1", title="n1", body="x", entities=["X"])   # 实体重叠 → link
    cand_plain = Note(id="N2", title="n2", body="x")                    # 无第二信号 → 不 link
    assert mgr._link_candidates(cand_overlap) == ["A"]
    assert mgr._link_candidates(cand_plain) == []
    idx.close()
