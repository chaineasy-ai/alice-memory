"""派生索引 MemoryIndex：SQLite FTS5(trigram+BM25) + 元数据 + 链接邻接表。

契约 §4.6-D1：`.md` 是唯一真源，本索引全部派生、可删除后 `reindex` 重建。
v0 检索 = FTS5 trigram + BM25（不手搓 grep）；<3 字 CJK 查询用 LIKE 兜底
（trigram 对 2 字/单字失效，见 §3.2）。
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import struct
from pathlib import Path
from typing import Iterable, Optional

from .model import Note, normalize_title

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    path TEXT,
    namespace TEXT,
    layer TEXT, type TEXT, title TEXT, status TEXT, body TEXT,
    title_norm TEXT, content_hash TEXT,
    tags TEXT, entities TEXT, links TEXT,
    importance INTEGER, confidence REAL,
    source TEXT, source_type TEXT, domain TEXT, owner TEXT, scope TEXT,
    idempotency_key TEXT,
    created TEXT, updated TEXT, last_accessed TEXT, access_count INTEGER,
    expires TEXT,
    vector BLOB
);
CREATE INDEX IF NOT EXISTS idx_hash ON notes(content_hash);
CREATE INDEX IF NOT EXISTS idx_idem ON notes(idempotency_key);
CREATE INDEX IF NOT EXISTS idx_title_norm ON notes(title_norm);
CREATE TABLE IF NOT EXISTS links (src TEXT, dst TEXT, PRIMARY KEY (src, dst));
CREATE INDEX IF NOT EXISTS idx_links_dst ON links(dst);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def content_hash(note_or_text) -> str:
    """内容指纹：标题+正文的 md5（mem0 式去重）。接受 Note 或 str。"""
    if isinstance(note_or_text, Note):
        text = f"{note_or_text.title}\n{note_or_text.body}"
    else:
        text = str(note_or_text)
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def _pack_vector(vec: Optional[list[float]]) -> Optional[bytes]:
    if vec is None:
        return None
    try:
        if len(vec) == 0:
            return None
    except TypeError:
        return None
    return struct.pack(f"<{len(vec)}f", *[float(x) for x in vec])


def _unpack_vector(blob: Optional[bytes]) -> list[float]:
    if not blob:
        return []
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


class MemoryIndex:
    def __init__(self, db_path: str | Path, tokenizer: str = "trigram"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self._fts_tokenizer = "trigram"
        self.tokenizer = tokenizer if tokenizer in ("trigram", "bigram") else "trigram"
        self._index_tokenizer = self.tokenizer
        self._setup()

    # -- schema -----------------------------------------------------------
    def _setup(self) -> None:
        self.conn.executescript(_SCHEMA)
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5("
                "id UNINDEXED, title, body, tags, entities, tokenize='trigram')"
            )
        except sqlite3.OperationalError:
            self._fts_tokenizer = "unicode61"
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5("
                "id UNINDEXED, title, body, tags, entities, tokenize='unicode61')"
            )
        # CJK <3 字兜底：unicode61 + 单字切分（报告中 A 级实测过的方案）
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_cjk USING fts5("
            "id UNINDEXED, seg, tokenize='unicode61')"
        )
        # CJK 重叠 2-gram（#232 item4；unicode61 + 2-gram 切分），与 trigram 双建可切
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_bi USING fts5("
            "id UNINDEXED, seg, tokenize='unicode61')"
        )
        self.conn.commit()
        # G1：索引存 tokenizer 标识，供 mismatch 守卫
        row = self.conn.execute("SELECT value FROM meta WHERE key='tokenizer'").fetchone()
        if row and row["value"] in ("trigram", "bigram"):
            self._index_tokenizer = row["value"]
        else:
            self._index_tokenizer = self.tokenizer
            self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('tokenizer',?)",
                              (self.tokenizer,))
            self.conn.commit()

    def _assert_tokenizer(self) -> None:
        if self._index_tokenizer != self.tokenizer:
            raise ValueError(
                f"索引 tokenizer={self._index_tokenizer}，请求 {self.tokenizer}；"
                f"请先 `mem reindex --lexical-tokenizer {self.tokenizer}`（索引可重建）")

    def close(self) -> None:
        self.conn.close()

    # -- 写 ---------------------------------------------------------------
    def upsert(self, note: Note, vector: Optional[list[float]] = None) -> None:
        self._assert_tokenizer()
        meta = note.to_frontmatter()
        self.conn.execute(
            "INSERT OR REPLACE INTO notes (id,path,namespace,layer,type,title,status,body,"
            "title_norm,content_hash,tags,entities,links,importance,confidence,source,"
            "source_type,domain,owner,scope,idempotency_key,created,updated,last_accessed,access_count,"
            "expires,vector) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                note.id, note.path,
                _namespace_from_path(note.path), note.layer, note.type, note.title,
                note.status, note.body,
                normalize_title(note.title), content_hash(note),
                " ".join(note.tags), " ".join(note.entities), " ".join(note.links),
                note.importance, note.confidence, note.source, note.source_type,
                note.domain, note.owner, note.scope, note.idempotency_key,
                note.created, note.updated,
                note.last_accessed or "", note.access_count or 0, note.expires or "",
                _pack_vector(vector),
            ),
        )
        self.conn.execute("DELETE FROM fts WHERE id=?", (note.id,))
        self.conn.execute("DELETE FROM fts_cjk WHERE id=?", (note.id,))
        self.conn.execute("DELETE FROM fts_bi WHERE id=?", (note.id,))
        flat = " ".join([note.title, note.body, " ".join(note.tags), " ".join(note.entities)])
        if self.tokenizer == "bigram":
            # 只填 active 分词表（索引可 rebuild），使体积可比
            self.conn.execute("INSERT INTO fts_bi (id,seg) VALUES (?,?)", (note.id, _seg_bigram(flat)))
        else:
            self.conn.execute(
                "INSERT INTO fts (id,title,body,tags,entities) VALUES (?,?,?,?,?)",
                (note.id, note.title, note.body, " ".join(note.tags), " ".join(note.entities)))
            self.conn.execute("INSERT INTO fts_cjk (id,seg) VALUES (?,?)", (note.id, _cjk_seg(flat)))
        self.conn.execute("DELETE FROM links WHERE src=?", (note.id,))
        for dst in note.links:
            self.conn.execute("INSERT OR IGNORE INTO links (src,dst) VALUES (?,?)", (note.id, dst))
        self.conn.commit()

    def remove(self, note_id: str) -> None:
        self.conn.execute("DELETE FROM notes WHERE id=?", (note_id,))
        self.conn.execute("DELETE FROM fts WHERE id=?", (note_id,))
        self.conn.execute("DELETE FROM fts_cjk WHERE id=?", (note_id,))
        self.conn.execute("DELETE FROM fts_bi WHERE id=?", (note_id,))
        self.conn.execute("DELETE FROM links WHERE src=? OR dst=?", (note_id, note_id))
        self.conn.commit()

    def clear(self) -> None:
        self.conn.executescript(
            "DELETE FROM notes; DELETE FROM fts; DELETE FROM fts_cjk; DELETE FROM fts_bi; DELETE FROM links;")
        self.conn.commit()

    def reindex(self, notes: Iterable[Note], vectors: Optional[dict] = None) -> int:
        # 允许通过 reindex 切换 tokenizer：先更新标识再重建
        self._index_tokenizer = self.tokenizer
        self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('tokenizer',?)", (self.tokenizer,))
        self.conn.commit()
        self.clear()
        n = 0
        for note in notes:
            vec = (vectors or {}).get(note.id)
            self.upsert(note, vector=vec)
            n += 1
        return n

    # -- 读 ---------------------------------------------------------------
    def get_meta(self, note_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
        return dict(row) if row else None

    def find_by_content_hash(self, h: str) -> list[str]:
        return [r["id"] for r in self.conn.execute("SELECT id FROM notes WHERE content_hash=?", (h,))]

    def find_by_title(self, normalized_title: str) -> list[str]:
        return [r["id"] for r in self.conn.execute(
            "SELECT id FROM notes WHERE title_norm=?", (normalized_title,))]

    def find_by_idempotency_key(self, key: str) -> list[str]:
        return [r["id"] for r in self.conn.execute(
            "SELECT id FROM notes WHERE idempotency_key=?", (key,))]

    def iter_vectors(self):
        for r in self.conn.execute("SELECT id, vector FROM notes WHERE vector IS NOT NULL"):
            v = _unpack_vector(r["vector"])
            if v:
                yield r["id"], v

    def neighbors(self, note_id: str, direction: str = "out") -> list[str]:
        if direction == "out":
            rows = self.conn.execute("SELECT dst FROM links WHERE src=?", (note_id,))
        elif direction == "in":
            rows = self.conn.execute("SELECT src FROM links WHERE dst=?", (note_id,))
        else:
            rows = self.conn.execute(
                "SELECT dst AS x FROM links WHERE src=? UNION SELECT src FROM links WHERE dst=?",
                (note_id, note_id))
        return [r[0] for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]

    # -- 检索 -------------------------------------------------------------
    def search_fts(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        self._assert_tokenizer()
        query = (query or "").strip()
        if not query:
            return []
        if self.tokenizer == "bigram":
            rows = self._match_bi(query, limit)
            if not rows:
                rows = self._like(query, limit)
            return rows
        rows = self._match(query, limit)
        if not rows:
            # curie iter3（A 级实测）：short CJK 用 LIKE 逐词 OR（R@1 0.671）
            # 明显优于 CJK 单字切分（R@1 0.214）；故 LIKE 在前，fts_cjk 仅最后救命。
            rows = self._like(query, limit)
        if not rows:
            rows = self._match_cjk(query, limit)
        return rows

    def _match_bi(self, query: str, limit: int) -> list[tuple[str, float]]:
        # 与 curie iter6 一致：2-gram 之间做 OR（AND 对长查询过约束）
        terms = list(dict.fromkeys(t for t in _seg_bigram(query).split() if t))
        if not terms:
            return []
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        try:
            cur = self.conn.execute(
                "SELECT id, bm25(fts_bi) AS rank FROM fts_bi "
                "WHERE fts_bi MATCH ? ORDER BY rank LIMIT ?", (match, limit))
            return [(r["id"], -float(r["rank"])) for r in cur]  # bm25 越小/越负越相关
        except sqlite3.OperationalError:
            return []

    def _match_cjk(self, query: str, limit: int) -> list[tuple[str, float]]:
        seg = _cjk_seg(query)
        terms = [t for t in seg.split() if t]
        if not terms:
            return []
        match = " AND ".join('"' + t.replace('"', '""') + '"' for t in terms)
        try:
            cur = self.conn.execute(
                "SELECT id, bm25(fts_cjk) AS rank FROM fts_cjk "
                "WHERE fts_cjk MATCH ? ORDER BY rank LIMIT ?",
                (match, limit),
            )
            return [(r["id"], -float(r["rank"])) for r in cur]  # bm25 越小/越负越相关
        except sqlite3.OperationalError:
            return []

    def _match(self, query: str, limit: int) -> list[tuple[str, float]]:
        terms = [t for t in re.split(r"\s+", query) if t]
        if not terms:
            return []
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        try:
            cur = self.conn.execute(
                "SELECT f.id, bm25(fts) AS rank FROM fts f "
                "JOIN notes n ON n.id=f.id "
                "WHERE fts MATCH ? ORDER BY rank LIMIT ?",
                (match, limit),
            )
            # bm25() 越小越相关 -> 转成越大越好
            return [(r["id"], -float(r["rank"])) for r in cur]  # bm25 越小/越负越相关
        except sqlite3.OperationalError:
            return []

    def _like(self, query: str, limit: int) -> list[tuple[str, float]]:
        # 按词拆分做 OR（整串 LIKE 对多词查询必然 0 命中）
        terms = [t for t in re.split(r"\s+", query) if t] or [query]
        clauses, params = [], []
        for t in terms:
            pat = f"%{t}%"
            clauses.append("(title LIKE ? OR body LIKE ? OR tags LIKE ? OR entities LIKE ?)")
            params += [pat, pat, pat, pat]
        sql = "SELECT id FROM notes WHERE " + " OR ".join(clauses) + " LIMIT ?"
        cur = self.conn.execute(sql, (*params, limit))
        return [(r["id"], 0.5) for r in cur]


def _seg_bigram(text: str) -> str:
    """CJK 重叠 2-gram（+ ascii 词）切分，供 fts_bi（unicode61）。"""
    out: list[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]+", text or ""):
        for i in range(len(run) - 1):
            out.append(run[i:i + 2])
        if len(run) == 1:
            out.append(run)
    out += re.findall(r"[a-z0-9_]+", (text or "").lower())
    return " ".join(out)


def _cjk_seg(text: str) -> str:
    """把 CJK 字符拆成单字（空格分隔），ASCII 保留——供 unicode61 FTS 索引/查询。"""
    out = []
    for ch in text or "":
        if "\u4e00" <= ch <= "\u9fff":
            out.append(f" {ch} ")
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _namespace_from_path(path: Optional[str]) -> str:
    if not path:
        return "shared"
    m = re.search(r"/memory/agents/([^/]+)/", str(path))
    return f"agents/{m.group(1)}" if m else "shared"
