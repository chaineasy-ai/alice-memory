"""alice-memory：C-Land markdown 文件式记忆系统。

契约：`cland-research/docs/markdown-memory-原理调研.md` §4.6/§4.7。
`.md` 是唯一真源；SQLite 索引为派生、可重建；分层 core/semantic/episodic/entity/
procedural/inbox/archive；命名空间 shared + agents/<id>；写 = flock + 原子写。
"""
from .model import Note
from .store import MemoryStore
from .index import MemoryIndex, content_hash
from .lifecycle import MemoryManager, WriteResult, GcPolicy
from .recall import RecallEngine, LexicalRetriever, HybridRetriever

__version__ = "0.1.0"
__all__ = ["Note", "MemoryStore", "MemoryIndex", "content_hash", "MemoryManager",
           "WriteResult", "GcPolicy", "RecallEngine", "LexicalRetriever", "HybridRetriever"]
