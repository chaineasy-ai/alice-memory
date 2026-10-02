"""``mem`` CLI —— 契约 §4.6 CLI（add/search/get/update/link/gc/reindex/stats/context）。

所有子命令输出 JSON，顶层带 ``schema_version``（如 ``mem.search.v1``）；字段只增不删。
根路径：``--root`` / ``$MEM_HOME``（别名 ``$MEM_ROOT``）；默认命名空间 ``$MEM_AGENT``。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .embed import get_embedder
from .index import MemoryIndex
from .lifecycle import GcPolicy, MemoryManager
from .model import LAYERS, TYPES
from .recall import RecallEngine
from .store import MemoryStore, SHARED

SCHEMA = {"add": "mem.add.v1", "search": "mem.search.v1", "get": "mem.get.v1",
          "update": "mem.update.v1", "link": "mem.link.v1", "gc": "mem.gc.v1",
          "reindex": "mem.reindex.v1", "stats": "mem.stats.v1",
          "init": "mem.init.v1", "context": "mem.context.v1"}


def _emit(cmd: str, payload: dict, exit_code: int = 0) -> int:
    payload = {"schema_version": SCHEMA[cmd], **payload}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return exit_code


def _make(root: str, lock_timeout: float, embed_backend: str):
    store = MemoryStore(root, lock_timeout=lock_timeout)
    store.init()
    index = MemoryIndex(store.indexdir / "index.sqlite")
    embedder = get_embedder(embed_backend)
    mgr = MemoryManager(store, index, embedder)
    recall = RecallEngine(store, index, embedder)
    return store, index, mgr, recall


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


def _ns(path) -> str:
    if not path:
        return SHARED
    p = str(path)
    import re
    m = re.search(r"/memory/agents/([^/]+)/", p)
    return f"agents/{m.group(1)}" if m else SHARED


def cmd_init(store, index, mgr, recall, args):
    return _emit("init", {"root": str(store.root), "ok": True})


def cmd_add(store, index, mgr, recall, args):
    ns = args.namespace or os.environ.get("MEM_AGENT") or SHARED
    if args.file:
        from .model import Note
        note = Note.from_text(Path(args.file).read_text(encoding="utf-8"))
        title, body = note.title, note.body
    else:
        title, body = args.title or (args.text or "").strip().split("\n", 1)[0], args.text or ""
    if not title:
        return _emit("add", {"error": "缺少标题/内容"}, 1)
    res = mgr.add(title.strip(), body.strip(), layer=args.layer, type=args.type,
                  tags=args.tags, entities=args.entities, importance=args.importance,
                  confidence=args.confidence, source=args.source,
                  source_type=args.source_type, links=args.link, summary=args.summary,
                  owner=args.owner, scope=args.scope, domain=args.domain,
                  expires=args.expires, note_id=args.id,
                  namespace=ns, force=args.force)
    return _emit("add", {"event": res.status, "message": res.message,
                         "related": res.related, "note": _note_json(res.note)})


def cmd_search(store, index, mgr, recall, args):
    mode = "lexical" if args.mode == "auto" and not any(index.iter_vectors()) else args.mode
    hits = recall.search(args.query, k=args.k, mode=args.mode, layer=args.layer,
                         type=args.type, tags=args.tags, namespace=args.namespace,
                         include_archived=args.include_archived, touch=args.touch)
    results = [{
        "id": h.note.id, "score": h.score, "path": h.note.path, "title": h.note.title,
        "layer": h.note.layer, "namespace": _ns(h.note.path),
        "snippet": (h.note.body or "")[:120].replace("\n", " "),
        "why": h.reason or "lexical", "components": h.components,
    } for h in hits]
    out: dict[str, Any] = {"mode": mode, "count": len(results), "results": results}
    if args.context:
        out["context"] = recall.context(args.query, token_budget=args.budget_tokens, k=args.k)
    return _emit("search", out)


def cmd_get(store, index, mgr, recall, args):
    note = store.find_by_id(args.id)
    if not note:
        return _emit("get", {"error": f"未找到记忆: {args.id}"}, 1)
    return _emit("get", {"note": _note_json(note)})


def cmd_update(store, index, mgr, recall, args):
    body = None
    if args.file:
        from .model import Note
        note = Note.from_text(Path(args.file).read_text(encoding="utf-8"))
        body = note.body
    elif args.append:
        cur = store.find_by_id(args.id)
        body = ((cur.body if cur else "") + "\n" + args.append).strip() if cur else args.append
    try:
        note = mgr.update(args.id, title=args.title, body=body, add_tags=args.add_tags,
                          add_entities=args.add_entities, importance=args.importance,
                          status=args.status, source=args.source, summary=args.summary,
                          layer=args.layer)
    except KeyError as e:
        return _emit("update", {"error": str(e)}, 1)
    return _emit("update", {"note": _note_json(note)})


def cmd_link(store, index, mgr, recall, args):
    try:
        note = mgr.link(args.id, args.to_id, unlink=args.unlink)
    except KeyError as e:
        return _emit("link", {"error": str(e)}, 1)
    return _emit("link", {"id": note.id, "links": note.links, "unlinked": args.unlink})


def cmd_gc(store, index, mgr, recall, args):
    policy = GcPolicy(stale_days=args.archive_stale, archive_superseded=not args.keep_superseded,
                      archive_expired=not args.keep_expired)
    res = mgr.gc(policy, apply=args.apply)
    return _emit("gc", {"dry_run": not args.apply, "count": len(res["candidates"]), **res})


def cmd_reindex(store, index, mgr, recall, args):
    n = mgr.reindex()
    return _emit("reindex", {"indexed": n, "db": str(index.db_path)})


def cmd_stats(store, index, mgr, recall, args):
    notes = list(store.iter_notes())
    by_layer: dict[str, int] = {}
    by_ns: dict[str, int] = {}
    for n in notes:
        by_layer[n.layer] = by_layer.get(n.layer, 0) + 1
        by_ns[_ns(n.path)] = by_ns.get(_ns(n.path), 0) + 1
    return _emit("stats", {"total": len(notes), "indexed": index.count(),
                           "by_layer": by_layer, "by_namespace": by_ns,
                           "active": sum(1 for n in notes if n.status == "active"),
                           "archived": sum(1 for n in notes if n.status == "archived")})


def cmd_context(store, index, mgr, recall, args):
    text = recall.context(args.query, token_budget=args.budget_tokens, k=args.k)
    return _emit("context", {"context": text, "tokens": __import__("alice_memory.recall", fromlist=["estimate_tokens"]).estimate_tokens(text)})


def build_parser() -> argparse.ArgumentParser:
    default_root = os.environ.get("MEM_HOME") or os.environ.get("MEM_ROOT") or "./memory-data"
    p = argparse.ArgumentParser(prog="mem", description="C-Land markdown memory (v0)")
    p.add_argument("--root", default=default_root)
    p.add_argument("--namespace", default=None,
                   help="读写命名空间；写默认 $MEM_AGENT→agents/<id>，读默认全部")
    p.add_argument("--lock-timeout", type=float, default=5.0)
    p.add_argument("--embed", default="none", choices=["none", "ollama"])
    p.add_argument("--json", action="store_true", help="JSON 输出（默认即 JSON）")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(func=cmd_init)

    a = sub.add_parser("add")
    a.add_argument("--text", default="")
    a.add_argument("--file")
    a.add_argument("--title", default="")
    a.add_argument("--type", default="note", choices=TYPES)
    a.add_argument("--layer", default="semantic", choices=LAYERS)
    a.add_argument("--tags", nargs="*", default=[])
    a.add_argument("--entities", nargs="*", default=[])
    a.add_argument("--link", nargs="*", default=[])
    a.add_argument("--importance", type=int, default=5)
    a.add_argument("--confidence", type=float)
    a.add_argument("--source", default="")
    a.add_argument("--source-type", default="unknown")
    a.add_argument("--summary", default="")
    a.add_argument("--owner", default="")
    a.add_argument("--scope", default="agent")
    a.add_argument("--domain", default="")
    a.add_argument("--expires")
    a.add_argument("--id")
    a.add_argument("--namespace")
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_add)

    g = sub.add_parser("get"); g.add_argument("id"); g.set_defaults(func=cmd_get)

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--mode", default="auto", choices=["auto", "lexical", "hybrid"])
    s.add_argument("--layer"); s.add_argument("--type")
    s.add_argument("--tags", nargs="*", default=[])
    s.add_argument("--budget-tokens", type=int, default=2000)
    s.add_argument("--context", action="store_true")
    s.add_argument("--namespace")
    s.add_argument("--include-archived", action="store_true")
    s.add_argument("--touch", action="store_true")
    s.set_defaults(func=cmd_search)

    u = sub.add_parser("update")
    u.add_argument("id")
    u.add_argument("--title"); u.add_argument("--file"); u.add_argument("--append")
    u.add_argument("--add-tags", nargs="*", default=[])
    u.add_argument("--add-entities", nargs="*", default=[])
    u.add_argument("--importance", type=int)
    u.add_argument("--status"); u.add_argument("--source"); u.add_argument("--summary")
    u.add_argument("--layer", choices=LAYERS)
    u.set_defaults(func=cmd_update)

    l = sub.add_parser("link"); l.add_argument("id"); l.add_argument("to_id")
    l.add_argument("--unlink", action="store_true"); l.set_defaults(func=cmd_link)

    gc = sub.add_parser("gc")
    gc.add_argument("--dedup", action="store_true")
    gc.add_argument("--archive-stale", type=int, default=180)
    gc.add_argument("--keep-superseded", action="store_true")
    gc.add_argument("--keep-expired", action="store_true")
    gc.add_argument("--dry-run", action="store_true", default=True)
    gc.add_argument("--apply", action="store_true")
    gc.set_defaults(func=cmd_gc)

    sub.add_parser("reindex").set_defaults(func=cmd_reindex)
    sub.add_parser("stats").set_defaults(func=cmd_stats)

    c = sub.add_parser("context")
    c.add_argument("query"); c.add_argument("--k", type=int, default=10)
    c.add_argument("--budget-tokens", type=int, default=2000)
    c.set_defaults(func=cmd_context)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    store, index, mgr, recall = _make(args.root, args.lock_timeout, args.embed)
    try:
        return args.func(store, index, mgr, recall, args)
    finally:
        index.close()


if __name__ == "__main__":
    sys.exit(main())
