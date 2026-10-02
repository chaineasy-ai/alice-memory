"""``mem`` CLI —— v0 接口契约 §4.6（**逐条对齐**）。

子命令：init / add / search / get / update / link / gc / reindex / stats / context。
全部输出 JSON、顶层 ``schema_version``；``--json`` 可在子命令后（契约写例）。

```bash
mem add <title> [body] [--file f] [--layer L] [--type T] [--tags a,b] ...
mem search "<query>" [-k N] [--layer L] [--tag T] [--mode lexical|hybrid|auto] [--all]
mem get <id>
mem update <id> [--title T] [--body B] [--add-tag a,b] [--importance N] [--status S]
mem link <id> <to-id> [--unlink]
mem gc [--apply] [--stale-days 180] [--min-importance 2]
mem reindex
mem stats
mem context "<task>" [--budget 2000] [--budget-unit tokens|chars] [-k N]
```
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

from .embed import get_embedder
from .index import MemoryIndex
from .lifecycle import GcPolicy, MemoryManager
from .model import LAYERS, TYPES
from .recall import RecallEngine
from .store import MemoryStore, SHARED
from . import frontmatter as fm

SCHEMA = {"add": "mem.add.v1", "search": "mem.search.v1", "get": "mem.get.v1",
          "update": "mem.update.v1", "link": "mem.link.v1", "gc": "mem.gc.v1",
          "reindex": "mem.reindex.v1", "stats": "mem.stats.v1",
          "init": "mem.init.v1", "context": "mem.context.v1",
          "adopt": "mem.adopt.v1"}


def _emit(cmd: str, payload: dict, exit_code: int = 0) -> int:
    print(json.dumps({"schema_version": SCHEMA[cmd], **payload}, ensure_ascii=False, indent=2))
    return exit_code


def _make(root: str, lock_timeout: float, embed_backend: str, tokenizer: str = "trigram",
          embed_model: str = ""):
    store = MemoryStore(root, lock_timeout=lock_timeout)
    store.init()
    index = MemoryIndex(store.indexdir / "index.sqlite", tokenizer=tokenizer)
    embedder = get_embedder(embed_backend, model=embed_model or "none",
                            url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    return store, index, MemoryManager(store, index, embedder), RecallEngine(store, index, embedder)


def _ns(path) -> str:
    if not path:
        return SHARED
    import re
    m = re.search(r"/memory/agents/([^/]+)/", str(path))
    return f"agents/{m.group(1)}" if m else SHARED


def _split_csv(values) -> list[str]:
    out: list[str] = []
    for v in values or []:
        out += [x.strip() for x in str(v).split(",") if x.strip()]
    return out


def _note_json(note) -> dict:
    return {
        "id": note.id, "title": note.title, "path": note.path,
        "namespace": _ns(note.path), "layer": note.layer, "type": note.type,
        "status": note.status, "tags": note.tags, "entities": note.entities,
        "links": note.links, "importance": note.importance, "confidence": note.confidence,
        "source": note.source, "source_type": note.source_type, "summary": note.summary,
        "owner": note.owner, "scope": note.scope, "domain": note.domain,
        "created": note.created, "updated": note.updated,
        "last_accessed": note.last_accessed, "access_count": note.access_count,
        "supersedes": note.supersedes, "superseded_by": note.superseded_by,
        "expires": note.expires, "body": note.body,
    }


# --------------------------------------------------------------------------- #
def cmd_init(store, index, mgr, recall, args):
    return _emit("init", {"root": str(store.root), "ok": True})


def _first_h1(body: str) -> str:
    import re
    m = re.search(r"^#\s+(.+?)\s*$", body or "", re.M)
    return m.group(1).strip() if m else ""


def _resolve_file_title(meta: dict, body: str, path: Path) -> str:
    """add --file 标题回退链：title → name → 首个 H1 → 文件名（**不得取正文**）。"""
    for key in ("title", "name"):
        v = str((meta or {}).get(key) or "").strip()
        if v:
            return v
    h1 = _first_h1(body)
    if h1:
        return h1
    return path.stem


def cmd_add(store, index, mgr, recall, args):
    ns = args.namespace or os.environ.get("MEM_AGENT") or SHARED
    meta: dict = {}
    if args.file:
        import re as _re
        p = Path(args.file)
        meta, body = fm.split_frontmatter(p.read_text(encoding="utf-8"))
        meta = meta or {}
        body = body.strip()
        title = args.title_opt or _resolve_file_title(meta, body, p)
        if not args.title_opt and _first_h1(body) and not meta.get("title"):
            body = _re.sub(r"^#\s+.+?\s*$", "", body, count=1, flags=_re.M).strip()
    else:
        title = args.title_opt or args.pos_title or ""
        body = args.text or args.pos_body or ""
        if not title:
            title = (body or "").strip().split("\n", 1)[0]
    if not title:
        return _emit("add", {"error": "缺少标题/内容"}, 2)

    def pick(cli, key, default):
        if cli is not None and cli != "":
            return cli
        v = meta.get(key)
        return v if v not in (None, "") else default

    def merge(cli_list, key):
        return list(dict.fromkeys(_split_csv(cli_list) + [str(x) for x in (meta.get(key) or [])]))

    # add --file：以 frontmatter 为基，CLI 显式值覆盖（不丢迁移元数据）
    mid = args.id or str(meta.get("id") or "") or None
    if mid and store.find_by_id(mid):        # 防静默覆盖：已存在同 id 则不沿用 meta id
        mid = args.id
    res = mgr.add(
        title.strip(), body.strip(),
        layer=pick(args.layer, "layer", "semantic"),
        type=pick(args.type, "type", "note"),
        status=str(meta.get("status") or "active"),
        tags=merge(list(args.tags) + list(args.tag), "tags"),
        entities=merge(args.entities, "entities"),
        links=merge(args.link, "links"),
        importance=args.importance if args.importance is not None else (meta.get("importance") or 5),
        confidence=args.confidence if args.confidence is not None else meta.get("confidence"),
        source=pick(args.source, "source", ""),
        source_type=pick(args.source_type, "source_type", "unknown"),
        summary=pick(args.summary, "summary", ""),
        owner=pick(args.owner, "owner", ""),
        scope=pick(args.scope, "scope", "agent"),
        domain=pick(args.domain, "domain", ""),
        expires=args.expires or meta.get("expires"),
        note_id=mid,
        idempotency_key=args.idempotency_key or meta.get("idempotency_key"),
        created=args.created or meta.get("created"),
        updated=meta.get("updated"),
        last_accessed=args.last_accessed or meta.get("last_accessed"),
        access_count=meta.get("access_count"),
        namespace=ns, force=args.force,
    )
    return _emit("add", {"event": res.status, "message": res.message,
                         "related": res.related, "note": _note_json(res.note)})


def cmd_search(store, index, mgr, recall, args):
    requested = args.mode
    has_vec = any(True for _ in index.iter_vectors())
    degraded = requested == "hybrid" and (recall.embedder.name == "none" or not has_vec)
    tags = _split_csv(list(args.tags) + list(args.tag))
    hits = recall.search(args.query, k=args.k, mode=args.mode, layer=args.layer,
                         type=args.type, tags=tags, namespace=args.namespace,
                         include_archived=args.all, touch=False)
    actual = "lexical" if (degraded or requested == "lexical" or not has_vec) else requested
    results = [{
        "id": h.note.id, "score": h.score, "path": h.note.path, "title": h.note.title,
        "layer": h.note.layer, "namespace": _ns(h.note.path),
        "snippet": (h.note.body or "")[:120].replace("\n", " "),
        "why": h.components, "why_reason": h.reason or "lexical",
        "source": h.note.source or "",
    } for h in hits]
    out: dict[str, Any] = {"mode": actual, "count": len(results), "results": results,
                           "degraded": degraded}
    if args.context:
        out["context"] = recall.context(args.query, token_budget=args.budget_tokens, k=args.k,
                                        budget_unit=args.budget_unit, touch=args.touch)
    if args.touch and hits:
        from datetime import datetime, timezone
        recall._touch(hits, datetime.now(timezone.utc).astimezone(), query=args.query)
    return _emit("search", out, 6 if degraded else 0)


def cmd_get(store, index, mgr, recall, args):
    note = store.find_by_id(args.id)
    if not note:
        return _emit("get", {"error": f"未找到记忆: {args.id}"}, 3)
    try:
        from .model import now_iso
        store.append_access({"ts": now_iso(), "id": note.id, "event": "get",
                             "layer": note.layer})
    except Exception:
        pass
    return _emit("get", {"note": _note_json(note)})


def cmd_update(store, index, mgr, recall, args):
    body = None
    if args.file:
        from .model import Note
        body = Note.from_text(Path(args.file).read_text(encoding="utf-8")).body
    elif args.body is not None:
        body = args.body
    elif args.append:
        cur = store.find_by_id(args.id)
        body = ((cur.body if cur else "") + "\n" + args.append).strip() if cur else args.append
    try:
        note = mgr.update(args.id, title=args.title, body=body,
                          add_tags=_split_csv(list(args.add_tags) + list(args.add_tag)),
                          add_entities=_split_csv(args.add_entities),
                          importance=args.importance, status=args.status,
                          source=args.source, summary=args.summary, layer=args.layer)
    except KeyError as e:
        return _emit("update", {"error": str(e)}, 3)
    return _emit("update", {"note": _note_json(note)})


def cmd_link(store, index, mgr, recall, args):
    try:
        note = mgr.link(args.id, args.to_id, unlink=args.unlink)
    except KeyError as e:
        return _emit("link", {"error": str(e)}, 3)
    return _emit("link", {"id": note.id, "links": note.links, "unlinked": args.unlink})


def cmd_gc(store, index, mgr, recall, args):
    policy = GcPolicy(stale_days=args.stale_days if args.stale_days is not None else args.archive_stale,
                      min_importance=args.min_importance,
                      archive_superseded=not args.keep_superseded,
                      archive_expired=not args.keep_expired)
    res = mgr.gc(policy, apply=args.apply)
    return _emit("gc", {"dry_run": not args.apply, "count": len(res["candidates"]), **res})


def cmd_reindex(store, index, mgr, recall, args):
    if getattr(args, "embed", None):
        from .embed import get_embedder
        mgr.embedder = get_embedder(args.embed, model=args.embed_model or "none")
    return _emit("reindex", {"indexed": mgr.reindex(), "db": str(index.db_path),
                             "vectors": sum(1 for _ in index.iter_vectors())})


def cmd_stats(store, index, mgr, recall, args):
    notes = list(store.iter_notes())
    by_layer: dict[str, int] = {}
    by_ns: dict[str, int] = {}
    by_status: dict[str, int] = {}
    links = 0
    for n in notes:
        by_layer[n.layer] = by_layer.get(n.layer, 0) + 1
        by_ns[_ns(n.path)] = by_ns.get(_ns(n.path), 0) + 1
        by_status[n.status] = by_status.get(n.status, 0) + 1
        links += len(n.links)
    return _emit("stats", {"total": len(notes), "by_namespace": by_ns, "by_layer": by_layer,
                           "by_status": by_status, "gc_candidates": len(mgr.gc_candidates()),
                           "index_schema": "mem.index.v1",
                           "vectors": sum(1 for _ in index.iter_vectors()), "links": links,
                           "access_log": _access_log_count(store)})


def _access_log_count(store) -> int:
    p = store.indexdir / "access.log"
    if not p.exists():
        return 0
    try:
        return sum(1 for _ in p.open(encoding="utf-8"))
    except OSError:
        return 0


def cmd_adopt(store, index, mgr, recall, args):
    ok = recall.adopt(args.id, query=args.query or "", rank=args.rank,
                      namespace=args.namespace)
    if not ok:
        return _emit("adopt", {"error": f"未找到记忆: {args.id}"}, 3)
    return _emit("adopt", {"id": args.id, "adopted": True, "rank": args.rank,
                           "query": args.query or "", "log": str(store.indexdir / "access.log")})


def cmd_context(store, index, mgr, recall, args):
    text = recall.context(args.query, token_budget=args.budget, k=args.k,
                          budget_unit=args.budget_unit, touch=args.touch)
    return _emit("context", {"context": text, "budget_unit": args.budget_unit})


# --------------------------------------------------------------------------- #
def _add_common_json(p):
    p.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                   help="JSON 输出（默认即 JSON；仅为契约兼容接受）")


def build_parser() -> argparse.ArgumentParser:
    default_root = os.environ.get("MEM_HOME") or os.environ.get("MEM_ROOT") or "./memory-data"
    p = argparse.ArgumentParser(prog="mem", description="C-Land markdown memory (v0, contract §4.6)")
    p.add_argument("--root", default=default_root)
    p.add_argument("--namespace", default=None,
                   help="读写命名空间；写默认 $MEM_AGENT→agents/<id>，读默认全部")
    p.add_argument("--lock-timeout", type=float, default=5.0)
    p.add_argument("--embed", default="none", choices=["none", "ollama", "local"])
    p.add_argument("--embed-model", default=os.environ.get("MEM_EMBED_MODEL", ""),
                   help="embedding 模型路径/名（local 后端默认 BAAI/bge-m3 本地快照）")
    p.add_argument("--lexical-tokenizer", default=os.environ.get("MEM_TOKENIZER", "trigram"),
                   choices=["trigram", "bigram"], help="词法分词（v0 默认 trigram；bigram 为 #232 spike）")
    p.add_argument("--json", action="store_true", help="JSON 输出（默认即 JSON）")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("init"); _add_common_json(c); c.set_defaults(func=cmd_init)

    a = sub.add_parser("add")
    a.add_argument("pos_title", nargs="?", metavar="title")
    a.add_argument("pos_body", nargs="?", metavar="body")
    a.add_argument("--title", dest="title_opt", default="")
    a.add_argument("--text", default="")
    a.add_argument("--file")
    a.add_argument("--type", default=None, choices=TYPES)
    a.add_argument("--layer", default=None, choices=LAYERS)
    a.add_argument("--tags", nargs="*", default=[])
    a.add_argument("--tag", action="append", default=[])
    a.add_argument("--entities", nargs="*", default=[])
    a.add_argument("--link", action="append", default=[])
    a.add_argument("--importance", type=int, default=None)
    a.add_argument("--confidence", type=float)
    a.add_argument("--source", default=None)
    a.add_argument("--source-type", default=None)
    a.add_argument("--summary", default=None)
    a.add_argument("--owner", default=None)
    a.add_argument("--scope", default=None)
    a.add_argument("--domain", default=None)
    a.add_argument("--expires")
    a.add_argument("--created", help="创建时间（迁移用，ISO-8601）")
    a.add_argument("--last-accessed", help="上次访问时间（迁移/校准用）")
    a.add_argument("--id")
    a.add_argument("--idempotency-key")
    a.add_argument("--namespace")
    a.add_argument("--force", action="store_true")
    _add_common_json(a); a.set_defaults(func=cmd_add)

    g = sub.add_parser("get"); g.add_argument("id"); _add_common_json(g); g.set_defaults(func=cmd_get)

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", "--k", type=int, default=10)
    s.add_argument("--mode", default="lexical", choices=["lexical", "hybrid", "auto"])
    s.add_argument("--layer"); s.add_argument("--type")
    s.add_argument("--tags", nargs="*", default=[])
    s.add_argument("--tag", action="append", default=[])
    s.add_argument("--budget-tokens", type=int, default=2000)
    s.add_argument("--budget-unit", default="tokens", choices=["tokens", "chars"])
    s.add_argument("--context", action="store_true")
    s.add_argument("--namespace")
    s.add_argument("--touch", action="store_true")
    s.add_argument("--all", dest="all", action="store_true")
    _add_common_json(s); s.set_defaults(func=cmd_search)

    u = sub.add_parser("update")
    u.add_argument("id")
    u.add_argument("--title"); u.add_argument("--body")
    u.add_argument("--file"); u.add_argument("--append")
    u.add_argument("--add-tags", nargs="*", default=[])
    u.add_argument("--add-tag", action="append", default=[])
    u.add_argument("--add-entities", nargs="*", default=[])
    u.add_argument("--importance", type=int)
    u.add_argument("--status"); u.add_argument("--source"); u.add_argument("--summary")
    u.add_argument("--layer", choices=LAYERS)
    _add_common_json(u); u.set_defaults(func=cmd_update)

    l = sub.add_parser("link"); l.add_argument("id"); l.add_argument("to_id")
    l.add_argument("--unlink", action="store_true"); _add_common_json(l); l.set_defaults(func=cmd_link)

    gc = sub.add_parser("gc")
    gc.add_argument("--stale-days", type=int, default=None)
    gc.add_argument("--archive-stale", type=int, default=180)
    gc.add_argument("--min-importance", type=int, default=2)
    gc.add_argument("--keep-superseded", action="store_true")
    gc.add_argument("--keep-expired", action="store_true")
    gc.add_argument("--dry-run", action="store_true", default=True)
    gc.add_argument("--apply", action="store_true")
    _add_common_json(gc); gc.set_defaults(func=cmd_gc)

    r = sub.add_parser("reindex")
    r.add_argument("--embed", choices=["none", "ollama", "local"], default=None,
                   help="重建时同时灌向量（hybrid 用）")
    r.add_argument("--embed-model", default=None)
    _add_common_json(r); r.set_defaults(func=cmd_reindex)
    st = sub.add_parser("stats"); _add_common_json(st); st.set_defaults(func=cmd_stats)

    ct = sub.add_parser("context")
    ct.add_argument("query")
    ct.add_argument("-k", "--k", type=int, default=10)
    ct.add_argument("--budget", type=int, default=2000)
    ct.add_argument("--budget-unit", default="tokens", choices=["tokens", "chars"])
    ct.add_argument("--touch", action="store_true")
    _add_common_json(ct); ct.set_defaults(func=cmd_context)

    ad = sub.add_parser("adopt", help="记录采纳信号（adopted=True，供 H 校准）")
    ad.add_argument("id")
    ad.add_argument("--query", default="")
    ad.add_argument("--rank", type=int)
    ad.add_argument("--namespace")
    _add_common_json(ad); ad.set_defaults(func=cmd_adopt)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    store, index, mgr, recall = _make(args.root, args.lock_timeout, args.embed,
                                      args.lexical_tokenizer, args.embed_model)
    try:
        return args.func(store, index, mgr, recall, args)
    except TimeoutError as e:
        print(json.dumps({"schema_version": "mem.error.v1", "error": str(e), "code": 4}, ensure_ascii=False))
        return 4
    except ValueError as e:
        # R448：非法字段（如 confidence 枚举）→ exit 2，不 traceback
        print(json.dumps({"schema_version": "mem.error.v1", "error": str(e), "code": 2}, ensure_ascii=False))
        return 2
    except (OSError, sqlite3.Error) as e:
        print(json.dumps({"schema_version": "mem.error.v1", "error": str(e), "code": 5}, ensure_ascii=False))
        return 5
    finally:
        index.close()


if __name__ == "__main__":
    sys.exit(main())
