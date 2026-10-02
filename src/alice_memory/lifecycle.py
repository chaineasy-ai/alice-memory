"""生命周期：写入 / 去重 / 合并（supersede） / 遗忘（gc）。

写入判定（mem0 ADD/UPDATE/NOOP 思路，文件可审计为前提）：
- ADD：内容新颖 → 新建；
- NOOP：内容指纹或归一化标题+正文相同 → 跳过，返回已有 id；
- SUPERSEDE：标题相同/近重复但正文不同 → 新建 + 旧笔记标 ``superseded``（保留历史）。

写临界区固定加锁顺序：namespace 锁 → 索引写（同锁内）。遗忘 = 迁 archive + status
= archived，非硬删；gc 默认 dry-run。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .embed import Embedder, NullEmbedder, cosine
from .index import MemoryIndex, content_hash
from .model import Note, now_iso, normalize_title, parse_iso
from .store import MemoryStore, SHARED


@dataclass
class WriteResult:
    status: str  # added | duplicate | superseded
    note: Note
    related: list[str] = field(default_factory=list)
    message: str = ""


@dataclass
class GcPolicy:
    stale_days: int = 180
    min_importance: int = 2
    archive_superseded: bool = True
    archive_expired: bool = True


def new_id(title: str, body: str = "", when: Optional[datetime] = None) -> str:
    """生成稳定 id：``mem-YYYYMMDD-<6hex>``（标题+正文哈希）。

    含正文是为了让「同标题不同正文」得到不同 id（否则会静默覆盖），
    同内容重复添加则得到相同 id，交由内容指纹判 NOOP。
    """
    when = when or datetime.now(timezone.utc).astimezone()
    digest = hashlib.sha1(f"{title}\n{body}".encode("utf-8")).hexdigest()[:6]
    return f"mem-{when.strftime('%Y%m%d')}-{digest}"


class MemoryManager:
    def __init__(self, store: MemoryStore, index: MemoryIndex,
                 embedder: Optional[Embedder] = None):
        self.store = store
        self.index = index
        self.embedder = embedder or NullEmbedder()

    # -- 向量 -------------------------------------------------------------
    def _vector_for(self, note: Note) -> Optional[list[float]]:
        if self.embedder.name == "none":
            return None
        return self.embedder.embed(f"{note.title}\n{note.summary}\n{note.body[:2000]}")

    def _near_duplicates(self, note: Note, threshold: float = 0.90) -> list[str]:
        if self.embedder.name == "none":
            return []
        vec = self._vector_for(note)
        if not vec:
            return []
        hits = [(nid, cosine(vec, v)) for nid, v in self.index.iter_vectors()]
        hits.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, sim in hits if sim >= threshold and nid != note.id]

    def _link_candidates(self, note: Note, lo: float = 0.84, hi: float = 0.90) -> list[str]:
        """G4：link 第二信号 = 余弦 ∈[0.84,0.90) **且** 链接/实体重叠（单余弦不够）。"""
        if self.embedder.name == "none":
            return []
        vec = self._vector_for(note)
        if not vec:
            return []
        ents, links = set(note.entities), set(note.links)
        out: list[str] = []
        for nid, v in self.index.iter_vectors():
            if nid == note.id:
                continue
            c = cosine(vec, v)
            if not (lo <= c < hi):
                continue
            meta = self.index.get_meta(nid) or {}
            o_ents = set((meta.get("entities") or "").split())
            o_links = set((meta.get("links") or "").split())
            if (ents & o_ents) or (links & o_links) or (nid in links):
                out.append(nid)
        return out

    def _link_bidirectional(self, a: Note, b: Note) -> None:
        """调用方需已持命名空间锁。"""
        if b.id not in a.links:
            a.links = sorted(set(a.links) | {b.id})
            a.updated = now_iso()
        if a.id not in b.links:
            b.links = sorted(set(b.links) | {a.id})
            b.updated = now_iso()
        self.store.write(a, namespace=self.store.namespace_of(a.path), locked=True)
        self.index.upsert(a)
        self.store.write(b, namespace=self.store.namespace_of(b.path), locked=True)
        self.index.upsert(b)

    # -- 写入 -------------------------------------------------------------
    def add(self, title: str, body: str = "", *, layer: str = "semantic",
            type: str = "note", status: str = "active", tags=None, entities=None,
            importance: int = 5, confidence=None, source: str = "",
            source_type: str = "unknown", links=None, summary: str = "",
            owner: str = "", scope: str = "agent", domain: str = "",
            expires: Optional[str] = None, note_id: Optional[str] = None,
            idempotency_key: Optional[str] = None,
            created: Optional[str] = None, updated: Optional[str] = None,
            last_accessed: Optional[str] = None, access_count: Optional[int] = None,
            namespace: str = SHARED, dedup: bool = True, force: bool = False) -> WriteResult:
        ns = self.store.normalize_namespace(namespace)
        note = Note(
            id=note_id or new_id(title, body), title=title, body=body, layer=layer, type=type,
            status=status, tags=list(tags or []), entities=list(entities or []),
            importance=importance, confidence=confidence, source=source,
            source_type=source_type, links=list(links or []), summary=summary,
            owner=owner, scope=scope, domain=domain, expires=expires,
            idempotency_key=idempotency_key,
        )
        if created:
            note.created = created.isoformat() if hasattr(created, "isoformat") else str(created)
        if updated:
            note.updated = updated.isoformat() if hasattr(updated, "isoformat") else str(updated)
        if last_accessed:
            note.last_accessed = last_accessed.isoformat() if hasattr(last_accessed, "isoformat") else str(last_accessed)
        if access_count is not None:
            note.access_count = int(access_count)
        if note.expires:
            note.expires = note.expires.isoformat() if hasattr(note.expires, "isoformat") else str(note.expires)
        note.validate()

        # 写临界区：namespace 锁内「重读 → 判定 → 原子写 → 同锁更新索引」
        with self.store.lock(ns):
            if dedup and not force:
                dup = self._find_duplicate(note)
                if dup:
                    return dup
            self.store.write(note, namespace=ns, locked=True)
            self.index.upsert(note, vector=self._vector_for(note))
            # G4：L2 link 第二信号（余弦 [0.84,0.90) 且链接/实体重叠）
            for cid in self._link_candidates(note):
                old = self.store.find_by_id(cid)
                if old:
                    self._link_bidirectional(note, old)
        return WriteResult("added", note, message="新建")

    def _find_duplicate(self, note: Note) -> Optional[WriteResult]:
        # I4：幂等键优先
        if note.idempotency_key:
            for existing_id in self.index.find_by_idempotency_key(note.idempotency_key):
                old = self.store.find_by_id(existing_id)
                if old:
                    return WriteResult("duplicate", old, related=[existing_id],
                                       message="幂等键命中，跳过")
        h = content_hash(note)
        for existing_id in self.index.find_by_content_hash(h):
            old = self.store.find_by_id(existing_id)
            if old:
                return WriteResult("duplicate", old, related=[existing_id],
                                   message="内容指纹相同，跳过")
        same_title = [i for i in self.index.find_by_title(normalize_title(note.title))
                      if i != note.id]
        if same_title:
            old = self.store.find_by_id(same_title[0])
            if old and old.body.strip() == note.body.strip():
                return WriteResult("duplicate", old, related=same_title,
                                   message="标题与正文相同，跳过")
            if old:
                return self._supersede(old, note)
        near = self._near_duplicates(note)
        if near:
            old = self.store.find_by_id(near[0])
            if old:
                return self._supersede(old, note)
        return None

    def _supersede(self, old: Note, new: Note) -> WriteResult:
        new.supersedes = sorted(set(new.supersedes) | {old.id})
        old.superseded_by = new.id
        old.status = "superseded"
        old.updated = now_iso()
        ns = self.store.namespace_of(new.path or self.store.path_for(new))
        self.store.write(new, namespace=ns, locked=True)
        self.index.upsert(new, vector=self._vector_for(new))
        self.store.write(old, namespace=self.store.namespace_of(old.path), locked=True)
        self.index.upsert(old)
        return WriteResult("superseded", new, related=[old.id],
                           message=f"取代 {old.id}（旧笔记保留为 superseded）")

    # -- 更新 / 链接 ------------------------------------------------------
    def update(self, note_id: str, *, title=None, body=None, add_tags=None,
               add_entities=None, importance=None, status=None, source=None,
               summary=None, layer=None) -> Note:
        note = self.store.find_by_id(note_id)
        if not note:
            raise KeyError(f"未找到记忆: {note_id}")
        ns = self.store.namespace_of(note.path)
        with self.store.lock(ns):
            if title is not None:
                note.title = title
            if body is not None:
                note.body = body
            if add_tags:
                note.tags = sorted(set(note.tags) | set(add_tags))
            if add_entities:
                note.entities = sorted(set(note.entities) | set(add_entities))
            if importance is not None:
                note.importance = importance
            if status is not None:
                note.status = status
            if source is not None:
                note.source = source
            if summary is not None:
                note.summary = summary
            if layer is not None:
                note.layer = layer
            note.updated = now_iso()
            note.validate()
            self.store.write(note, namespace=ns, locked=True)
            self.index.upsert(note, vector=self._vector_for(note))
        return note

    def link(self, src_id: str, dst_id: str, *, unlink: bool = False) -> Note:
        src = self.store.find_by_id(src_id)
        if not src:
            raise KeyError(f"源记忆不存在: {src_id}")
        if not self.store.find_by_id(dst_id):
            raise KeyError(f"目标记忆不存在: {dst_id}")
        ns = self.store.namespace_of(src.path)
        with self.store.lock(ns):
            if unlink:
                src.links = [x for x in src.links if x != dst_id]
            else:
                src.links = sorted(set(src.links) | {dst_id})
            src.updated = now_iso()
            self.store.write(src, namespace=ns, locked=True)
            self.index.upsert(src)
        return src

    # -- 遗忘 -------------------------------------------------------------
    def archive(self, note_id: str) -> Note:
        note = self.store.find_by_id(note_id)
        if not note:
            raise KeyError(f"未找到记忆: {note_id}")
        ns = self.store.namespace_of(note.path)
        with self.store.lock(ns):
            note.status = "archived"
            note.updated = now_iso()
            self.store.move(note, "archive", namespace=ns)
            self.index.upsert(note)
        return note

    def gc_candidates(self, policy: Optional[GcPolicy] = None, now=None) -> list[dict]:
        policy = policy or GcPolicy()
        now = now or datetime.now(timezone.utc).astimezone()
        cands: list[dict] = []
        seen: dict[str, str] = {}
        for note in self.store.iter_notes():
            if note.layer == "archive":
                continue
            reason = None
            if policy.archive_expired and note.expires and (parse_iso(note.expires) or now) < now:
                reason = "expired"
            elif policy.archive_superseded and note.status == "superseded":
                reason = "superseded"
            else:
                ref = parse_iso(note.last_accessed) or parse_iso(note.created)
                age = (now - ref).days if ref else 0
                if (note.access_count == 0 and note.importance <= policy.min_importance
                        and age >= policy.stale_days):
                    reason = "stale"
            digest = content_hash(note)
            if digest in seen:
                reason = reason or "duplicate"
            else:
                seen[digest] = note.id
            if reason:
                cands.append({"id": note.id, "title": note.title, "layer": note.layer,
                              "reason": reason})
        return cands

    def gc(self, policy: Optional[GcPolicy] = None, apply: bool = False, now=None) -> dict:
        cands = self.gc_candidates(policy, now)
        archived: list[str] = []
        if apply:
            for c in cands:
                try:
                    self.archive(c["id"])
                    archived.append(c["id"])
                except KeyError:
                    pass
        return {"candidates": cands, "archived": archived, "applied": apply}

    # -- 维护 -------------------------------------------------------------
    def reindex(self) -> int:
        notes = list(self.store.iter_notes())
        vectors = {n.id: v for n in notes if (v := self._vector_for(n))}
        return self.index.reindex(notes, vectors)
