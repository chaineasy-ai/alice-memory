"""存储层 MemoryStore：文件即真源 + 命名空间 + flock + 原子写。

契约：`markdown-memory-原理调研.md` §4.2（目录）、§4.7（命名空间/flock）、
§4.6（D1 `.md` 唯一真源、索引派生可重建）。

目录::

    <root>/memory/                      # 共享命名空间（shared）
    │   ├── core/ semantic/ episodic/ entity/ procedural/ inbox/ archive/
    │   └── agents/<agent-id>/<layer>/  # 私有命名空间
    <root>/.mem/index.sqlite            # 派生索引（可删，reindex 重建）
    <root>/.mem/locks/<ns>.lock         # flock 锁文件

写：同命名空间 `flock` 互斥 + 同目录 tmp + fsync + os.replace 原子落盘。
读：无锁、递归全部命名空间，不阻塞写者。
"""
from __future__ import annotations

import fcntl
import os
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from .model import LAYERS, Note

SHARED = "shared"


class _NamespaceLock:
    """可重入的按命名空间 flock（同线程重入不重复加锁）。"""

    def __init__(self, lock_dir: Path, timeout: float = 5.0):
        self.lock_dir = lock_dir
        self.timeout = timeout
        self._local = threading.local()

    def _state(self) -> dict:
        if not hasattr(self._local, "held"):
            self._local.held = {}
        return self._local.held

    def __enter__(self):
        return self

    def acquire(self, namespace: str):
        held = self._state()
        if namespace in held:  # (fh, depth)
            fh, depth = held[namespace]
            held[namespace] = (fh, depth + 1)
            return
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        path = self.lock_dir / f"{namespace.replace('/', '-')}.lock"
        fh = open(path, "a+")
        deadline = time.time() + self.timeout
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() >= deadline:
                    fh.close()
                    raise TimeoutError(f"获取命名空间锁超时: {namespace}")
                time.sleep(0.02)
        held[namespace] = (fh, 1)

    def release(self, namespace: str):
        held = self._state()
        if namespace not in held:
            return
        fh, depth = held[namespace]
        if depth > 1:
            held[namespace] = (fh, depth - 1)
            return
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()
        except OSError:
            pass
        del held[namespace]

    class _Ctx:
        def __init__(self, mgr, ns):
            self.mgr, self.ns = mgr, ns

        def __enter__(self):
            self.mgr.acquire(self.ns)
            return self

        def __exit__(self, *exc):
            self.mgr.release(self.ns)
            return False

    def lock(self, namespace: str):
        return self._Ctx(self, namespace)


class MemoryStore:
    def __init__(self, root: str | os.PathLike, *, lock_timeout: float = 5.0):
        self.root = Path(root)
        self.memdir = self.root / "memory"
        self.indexdir = self.root / ".mem"
        self.lock_dir = self.indexdir / "locks"
        self._locks = _NamespaceLock(self.lock_dir, lock_timeout)

    # -- 初始化 -----------------------------------------------------------
    def init(self) -> None:
        (self.indexdir / "locks").mkdir(parents=True, exist_ok=True)
        for layer in LAYERS:
            (self.memdir / layer).mkdir(parents=True, exist_ok=True)

    def lock(self, namespace: str = SHARED):
        return self._locks.lock(namespace)

    # -- 路径 / 命名空间 --------------------------------------------------
    @staticmethod
    def normalize_namespace(namespace: Optional[str], default: str = SHARED) -> str:
        ns = (namespace or default or SHARED).strip().strip("/")
        if ns in ("", SHARED, "shared"):
            return SHARED
        if ns.startswith("agents/"):
            return ns
        return f"agents/{ns}"

    def ns_dir(self, namespace: str) -> Path:
        namespace = self.normalize_namespace(namespace)
        if namespace == SHARED:
            return self.memdir
        return self.memdir / namespace  # memory/agents/<id>

    def path_for(self, note: Note, namespace: str = SHARED) -> Path:
        layer = note.layer if note.layer in LAYERS else "semantic"
        return self.ns_dir(namespace) / layer / f"{note.id}.md"

    def namespace_of(self, path: str | os.PathLike) -> str:
        rel = Path(path).resolve().relative_to(self.memdir.resolve())
        parts = rel.parts
        if parts[:1] == ("agents",) and len(parts) >= 2:
            return f"agents/{parts[1]}"
        return SHARED

    # -- 读 ---------------------------------------------------------------
    def read(self, path: str | os.PathLike) -> Optional[Note]:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            return Note.from_text(text, path=str(path))
        except Exception:
            return None

    def iter_paths(self, namespace: Optional[str] = None) -> Iterable[Path]:
        if namespace:
            base = self.ns_dir(namespace)
            yield from sorted(base.rglob("*.md")) if base.exists() else []
            return
        if not self.memdir.exists():
            return
        yield from sorted(self.memdir.rglob("*.md"))

    def iter_notes(self, namespace: Optional[str] = None) -> Iterable[Note]:
        for p in self.iter_paths(namespace):
            note = self.read(p)
            if note:
                yield note

    def find_by_id(self, note_id: str, namespace: Optional[str] = None) -> Optional[Note]:
        for note in self.iter_notes(namespace):
            if note.id == note_id:
                return note
        return None

    # -- 写 ---------------------------------------------------------------
    def write(self, note: Note, namespace: Optional[str] = None, *, locked: bool = False) -> Path:
        ns = self.normalize_namespace(namespace) if namespace else SHARED
        if note.path and not namespace:
            # 保持原命名空间（更新既有笔记时）
            try:
                ns = self.namespace_of(note.path)
            except ValueError:
                ns = SHARED
        path = self.path_for(note, ns)
        ctx = self.lock(ns) if not locked else _nullcontext()
        with ctx:
            path.parent.mkdir(parents=True, exist_ok=True)
            note.path = str(path)
            tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(note.to_text())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        return path

    def move(self, note: Note, layer: str, namespace: Optional[str] = None) -> Path:
        """迁移到另一个层（如 archive）。保留 id，非破坏性。"""
        note.layer = layer
        return self.write(note, namespace)

    # -- 访问日志（半衰期/去重校准用，append-only JSONL） ------------------
    def append_access(self, record: dict) -> None:
        import json as _json
        self.indexdir.mkdir(parents=True, exist_ok=True)
        line = (_json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(self.indexdir / "access.log",
                     os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)          # O_APPEND 单次写对小行在 Linux 上原子
        finally:
            os.close(fd)

    def delete(self, note: Note) -> None:
        if note.path:
            Path(note.path).unlink(missing_ok=True)


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
