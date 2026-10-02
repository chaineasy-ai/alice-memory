"""笔记数据模型（领域实体）。

一条记忆 = 一个带 YAML frontmatter 的 Markdown 文件。本模块只定义内存表示和
schema 校验/序列化，不依赖存储或索引实现。

契约来源：`cland-research/docs/markdown-memory-原理调研.md` §4.6（frontmatter）、
§4.2（目录/分层）、§4.7（命名空间）。
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

from . import frontmatter as fm

# 分层 == frontmatter `layer` == 共享层目录名（§4.2）
LAYERS = ("core", "semantic", "episodic", "entity", "procedural", "inbox", "archive")
TYPES = ("note", "core", "episode", "procedure", "entity", "index")
STATUSES = ("active", "superseded", "archived")
SOURCE_TYPES = ("paper", "commit", "issue", "experiment", "chat", "doc", "web", "note", "unknown")

# §4.6：必填 7 项 + 生命周期/溯源字段
REQUIRED_FIELDS = ("id", "title", "type", "layer", "created", "updated", "status")
KNOWN_FIELDS = set(REQUIRED_FIELDS) | {
    "importance", "confidence", "last_accessed", "access_count", "tags", "entities",
    "links", "supersedes", "superseded_by", "expires", "source", "source_type",
    "summary", "owner", "scope", "domain", "idempotency_key",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_confidence(value):
    """R448：confidence 校验前归一化（high/medium/low → 0.9/0.6/0.3；非法报错）。"""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"confidence 非法: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    table = {"high": 0.9, "h": 0.9, "medium": 0.6, "med": 0.6, "m": 0.6, "low": 0.3, "l": 0.3}
    if s in table:
        return table[s]
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"confidence 非法: {value!r}（需数值 0..1 或 high/medium/low）")


def slugify(text: str, maxlen: int = 48) -> str:
    text = unicodedata.normalize("NFKC", text or "").strip()
    import re
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text, flags=re.UNICODE)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[:maxlen] or "untitled"


def normalize_title(title: str) -> str:
    import re
    text = unicodedata.normalize("NFKC", title or "").lower()
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


@dataclass
class Note:
    id: str
    title: str
    body: str = ""
    type: str = "note"
    layer: str = "semantic"
    status: str = "active"
    created: str = field(default_factory=now_iso)
    updated: str = field(default_factory=now_iso)
    importance: int = 5
    confidence: Optional[float] = None
    last_accessed: Optional[str] = None
    access_count: int = 0
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    expires: Optional[str] = None
    source: str = ""
    source_type: str = "unknown"
    summary: str = ""
    owner: str = ""
    scope: str = "agent"
    domain: str = ""
    idempotency_key: Optional[str] = None
    path: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    # -- 校验 -------------------------------------------------------------
    def validate(self) -> "Note":
        if not self.id:
            raise ValueError("note.id 不能为空")
        if self.layer not in LAYERS:
            raise ValueError(f"layer 必须是 {LAYERS} 之一，收到 {self.layer!r}")
        if self.type not in TYPES:
            raise ValueError(f"type 必须是 {TYPES} 之一，收到 {self.type!r}")
        if self.status not in STATUSES:
            raise ValueError(f"status 必须是 {STATUSES} 之一，收到 {self.status!r}")
        if not (1 <= int(self.importance) <= 10):
            raise ValueError("importance 必须在 1..10")
        if self.confidence is not None:
            self.confidence = normalize_confidence(self.confidence)
        if self.confidence is not None and not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError("confidence 必须在 0..1")
        return self

    # -- frontmatter ------------------------------------------------------
    def to_frontmatter(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id, "title": self.title, "type": self.type,
            "layer": self.layer, "status": self.status,
            "created": self.created, "updated": self.updated,
        }
        optional = [
            ("importance", self.importance),
            ("confidence", self.confidence),
            ("last_accessed", self.last_accessed),
            ("access_count", self.access_count),
            ("tags", self.tags), ("entities", self.entities), ("links", self.links),
            ("supersedes", self.supersedes), ("superseded_by", self.superseded_by),
            ("expires", self.expires), ("source", self.source),
            ("source_type", self.source_type if self.source_type != "unknown" else ""),
            ("summary", self.summary), ("owner", self.owner), ("scope", self.scope),
            ("domain", self.domain),
        ]
        for k, v in optional:
            if v not in (None, "", [], 0):
                data[k] = v
        for k, v in self.extra.items():
            data.setdefault(k, v)
        return data

    @classmethod
    def from_frontmatter(cls, data: dict[str, Any], body: str = "", path: Optional[str] = None) -> "Note":
        data = dict(data or {})
        known = {k: data.pop(k) for k in list(data) if k in KNOWN_FIELDS}
        return cls(
            id=str(known.get("id") or ""), title=str(known.get("title") or ""),
            body=body or "", type=str(known.get("type") or "note"),
            layer=str(known.get("layer") or "semantic"), status=str(known.get("status") or "active"),
            created=str(known.get("created") or now_iso()), updated=str(known.get("updated") or now_iso()),
            importance=int(known.get("importance") or 5), confidence=normalize_confidence(known.get("confidence")),
            last_accessed=known.get("last_accessed"), access_count=int(known.get("access_count") or 0),
            tags=list(known.get("tags") or []), entities=list(known.get("entities") or []),
            links=list(known.get("links") or []), supersedes=list(known.get("supersedes") or []),
            superseded_by=known.get("superseded_by"), expires=known.get("expires"),
            source=str(known.get("source") or ""), source_type=str(known.get("source_type") or "unknown"),
            summary=str(known.get("summary") or ""), owner=str(known.get("owner") or ""),
            scope=str(known.get("scope") or "agent"), domain=str(known.get("domain") or ""),
            idempotency_key=known.get("idempotency_key"),
            path=path, extra={k: v for k, v in data.items()},
        )

    # -- IO ---------------------------------------------------------------
    def to_text(self) -> str:
        return fm.render(self.to_frontmatter(), self.body)

    @classmethod
    def from_text(cls, text: str, path: Optional[str] = None) -> "Note":
        meta, body = fm.split_frontmatter(text)
        if not meta:
            # 无 frontmatter：允许，最小 schema，用首行做标题
            first = next((ln.strip(" #") for ln in (body or "").splitlines() if ln.strip()), "")
            return cls(id=slugify(first), title=first, body=(body or "").strip(), path=path)
        return cls.from_frontmatter(meta, body.strip(), path=path)

    def touch(self, when: Optional[str] = None) -> None:
        self.last_accessed = when or now_iso()
        self.access_count = int(self.access_count or 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
