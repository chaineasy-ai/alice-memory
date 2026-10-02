"""召回层：检索抽象（lexical/hybrid）+ RRF + 时间/重要度重排 + 上下文装配。

契约 §4.4/§4.6：检索层可插拔（v0 lexical，v0.1 hybrid 不改存储/CLI）；
打分 ``w_rel·rel + w_rec·recency + w_imp·importance + w_graph·graph``；
`context` 先钉 core 层，再按分数贪心填到 token 预算。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol

from .embed import Embedder, NullEmbedder, cosine
from .index import MemoryIndex
from .model import Note, parse_iso
from .store import MemoryStore, SHARED

RRF_K = 60.0


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk + math.ceil(other / 4.0))


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int, *, filters: dict) -> list[tuple[str, float]]: ...


class LexicalRetriever:
    """v0：FTS5 trigram + BM25。"""

    name = "lexical"

    def __init__(self, index: MemoryIndex):
        self.index = index

    def retrieve(self, query: str, k: int, *, filters: dict) -> list[tuple[str, float]]:
        hits = self.index.search_fts(query, limit=max(k * 4, 40))
        if not filters:
            return hits[:k]
        return [(nid, s) for nid, s in hits if _match_filters(self.index.get_meta(nid), filters)][:k]


class HybridRetriever:
    """v0.1：FTS5 + 向量 → RRF（零标定）。"""

    name = "hybrid"

    def __init__(self, index: MemoryIndex, embedder: Embedder, k_rrf: float = RRF_K):
        self.index = index
        self.embedder = embedder
        self.k_rrf = k_rrf

    def retrieve(self, query: str, k: int, *, filters: dict) -> list[tuple[str, float]]:
        pool = max(k * 4, 40)
        fts = self.index.search_fts(query, limit=pool)
        rank_lists = [[nid for nid, _ in fts]]
        qvec = self.embedder.embed(query) if self.embedder.name != "none" else None
        if qvec:
            vec = sorted(((nid, cosine(qvec, v)) for nid, v in self.index.iter_vectors()),
                         key=lambda x: x[1], reverse=True)[:pool]
            rank_lists.append([nid for nid, _ in vec])
        fused: dict[str, float] = {}
        for ranks in rank_lists:
            for rank, nid in enumerate(ranks):
                fused[nid] = fused.get(nid, 0.0) + 1.0 / (self.k_rrf + rank + 1)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        if filters:
            ranked = [(nid, s) for nid, s in ranked if _match_filters(self.index.get_meta(nid), filters)]
        return ranked[:k]


@dataclass
class RecallConfig:
    w_rel: float = 1.0
    w_rec: float = 0.6
    w_imp: float = 0.5
    w_graph: float = 0.3
    half_life_hours: float = 720.0


@dataclass
class ScoredNote:
    note: Note
    score: float
    components: dict = field(default_factory=dict)
    reason: str = ""


def _canon_ns(ns: str) -> str:
    ns = (ns or "shared").strip().strip("/")
    if ns in ("", "shared"):
        return "shared"
    return ns if ns.startswith("agents/") else f"agents/{ns}"


def _match_filters(meta: Optional[dict], filters: dict) -> bool:
    if not meta:
        return False
    if filters.get("layer") and meta.get("layer") != filters["layer"]:
        return False
    if filters.get("type") and meta.get("type") != filters["type"]:
        return False
    if filters.get("namespace") and str(filters["namespace"]) not in ("all", "*"):
        if _canon_ns(meta.get("namespace") or "shared") != _canon_ns(filters["namespace"]):
            return False
    if filters.get("tags"):
        tags = set((meta.get("tags") or "").split())
        if not (set(filters["tags"]) & tags):
            return False
    if not filters.get("include_archived"):
        if meta.get("status") in ("archived", "superseded") or meta.get("layer") == "archive":
            return False
    exp = parse_iso(meta.get("expires"))
    if exp and exp < datetime.now(timezone.utc).astimezone():
        return False
    return True


class RecallEngine:
    def __init__(self, store: MemoryStore, index: MemoryIndex,
                 embedder: Optional[Embedder] = None, config: Optional[RecallConfig] = None):
        self.store = store
        self.index = index
        self.embedder = embedder or NullEmbedder()
        self.config = config or RecallConfig()

    def _retriever(self, mode: str) -> Retriever:
        mode = (mode or "auto").lower()
        if mode == "auto":
            has_vec = any(True for _ in self.index.iter_vectors())
            mode = "hybrid" if (has_vec and self.embedder.name != "none") else "lexical"
        if mode == "hybrid":
            if self.embedder.name != "none" and any(True for _ in self.index.iter_vectors()):
                return HybridRetriever(self.index, self.embedder)
            return LexicalRetriever(self.index)  # 降级
        return LexicalRetriever(self.index)

    def search(self, query: str, k: int = 10, *, mode: str = "auto", layer: Optional[str] = None,
               type: Optional[str] = None, tags: Optional[list[str]] = None,
               namespace: Optional[str] = None, include_archived: bool = False,
               touch: bool = False, now: Optional[datetime] = None) -> list[ScoredNote]:
        now = now or datetime.now(timezone.utc).astimezone()
        filters = {"layer": layer, "type": type, "tags": tags or [],
                   "namespace": namespace, "include_archived": include_archived}
        raw = self._retriever(mode).retrieve(query, max(k * 4, k), filters=filters)
        if not raw:
            return []
        max_rel = max(s for _, s in raw) or 1.0

        # 图加成：以最高分为种子
        seed = raw[0][0]
        neighbors = set(self.index.neighbors(seed, "out")) | set(self.index.neighbors(seed, "in"))
        seed_meta = self.index.get_meta(seed) or {}
        seed_entities = set((seed_meta.get("entities") or "").split())

        scored: list[ScoredNote] = []
        for nid, rel_raw in raw:
            meta = self.index.get_meta(nid)
            if not meta:
                continue
            rel = rel_raw / max_rel
            rec = self._recency(meta, now)
            imp = (meta.get("importance") or 5) / 10.0
            graph = 1.0 if nid in neighbors else (
                0.5 if seed_entities & set((meta.get("entities") or "").split()) else 0.0)
            c = self.config
            score = c.w_rel * rel + c.w_rec * rec + c.w_imp * imp + c.w_graph * graph
            scored.append(ScoredNote(
                note=self._meta_note(meta), score=round(score, 6),
                components={"rel": round(rel, 4), "recency": round(rec, 4),
                            "importance": round(imp, 4), "graph": graph},
                reason="linked" if graph else "",
            ))
        scored.sort(key=lambda s: s.score, reverse=True)
        top = scored[:k]
        # 只为最终 top-k 读盘（评分只用元数据）
        for s in top:
            s.note = self._load_note(s.note.id, self.index.get_meta(s.note.id) or {})
        if touch:
            self._touch(top, now, query=query)
        return top

    @staticmethod
    def _meta_note(meta: dict) -> Note:
        return Note(id=meta["id"], title=meta.get("title") or "",
                    layer=meta.get("layer") or "semantic", path=meta.get("path"))

    def _load_note(self, nid: str, meta: dict) -> Note:
        # 优先按索引中记录的 path 直接读（O(1)），避免每个命中全量扫盘
        path = meta.get("path")
        if path:
            note = self.store.read(path)
            if note:
                return note
        note = self.store.find_by_id(nid)
        if note:
            return note
        return Note(id=nid, title=meta.get("title") or "", layer=meta.get("layer") or "semantic")

    def _recency(self, meta: dict, now: datetime) -> float:
        ref = parse_iso(meta.get("last_accessed")) or parse_iso(meta.get("updated")) or parse_iso(meta.get("created"))
        if not ref:
            return 0.5
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=now.tzinfo)
        age_hours = max(0.0, (now - ref).total_seconds() / 3600.0)
        return 0.5 ** (age_hours / self.config.half_life_hours)

    def _touch(self, scored: list[ScoredNote], now: datetime, query: str = "") -> None:
        for s in scored:
            if s.note.path:
                s.note.touch(now.isoformat())
                try:
                    self.store.write(s.note)
                    self.index.upsert(s.note)
                except Exception:
                    pass
                # 访问日志：为半衰期 H / 去重阈值校准提供时间序列真值
                try:
                    self.store.append_access({
                        "ts": now.isoformat(), "id": s.note.id, "event": "recall",
                        "query": query, "score": s.score, "layer": s.note.layer,
                        "components": s.components,
                    })
                except Exception:
                    pass

    # -- 上下文装配 -------------------------------------------------------
    def context(self, query: str, token_budget: int = 2000, k: int = 10,
                mode: str = "lexical", include_core: bool = True,
                budget_unit: str = "tokens", touch: bool = False) -> str:
        def cost_of(text: str) -> int:
            return len(text) if budget_unit == "chars" else estimate_tokens(text)
        sections: list[str] = []
        used = 0
        if include_core:
            core = [n for n in self.store.iter_notes() if n.layer == "core" and n.status == "active"]
            core.sort(key=lambda n: n.importance, reverse=True)
            if core:
                block = self._render_block("核心记忆（常驻）", core)
                used += cost_of(block)
                sections.append(block)
        # I1：context 默认只读，不 touch（--touch 显式才写）
        hits = self.search(query, k=k, mode=mode, touch=touch)
        # I2：确定性排序 score desc, id asc
        hits = sorted(hits, key=lambda h: (-h.score, h.note.id))
        picked = []
        for hit in hits:
            cost = cost_of(self._render_note(hit.note))
            if used + cost > token_budget and picked:
                break
            picked.append(hit.note)
            used += cost
        if picked:
            sections.append(self._render_block("相关记忆（检索）", picked))
        return "\n\n".join(sections) if sections else ""

    @staticmethod
    def _render_note(note: Note) -> str:
        # I2：确定性分隔头（集成层可解析）；排序由 context 保证
        head = (f"<!-- mem: id={note.id} layer={note.layer} "
                f"source={note.source or ''} -->\n### {note.title}\n")
        return head + (note.body or "").strip() + "\n"

    @classmethod
    def _render_block(cls, title: str, notes: list[Note]) -> str:
        return "\n".join([f"## {title}"] + [cls._render_note(n) for n in notes])
