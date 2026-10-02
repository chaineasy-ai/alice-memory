"""可选向量嵌入后端。

默认 ``none``（只用 BM25 全文）；可选：

- ``ollama``：调用本机 Ollama ``/api/embeddings``（零付费；需先 ``ollama pull``
  一个嵌入模型，如 ``nomic-embed-text`` 或 ``bge-m3``）。
- ``sentence-transformers``：本地 Python 模型（需额外安装）。

所有后端在不可用时**优雅降级**为 None，不阻断核心流程。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional, Protocol

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> Optional[list[float]]: ...

    def embed_batch(self, texts: list[str]) -> list[Optional[list[float]]]: ...


class NullEmbedder:
    name = "none"
    dim = 0

    def embed(self, text: str) -> Optional[list[float]]:
        return None

    def embed_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        return [None for _ in texts]


class OllamaEmbedder:
    """通过 Ollama 本地 HTTP API 生成嵌入（零付费）。"""

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, url: str = DEFAULT_OLLAMA_URL,
                 timeout: float = 30.0):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{model}"
        self.dim = 0  # 首次调用后确定

    def embed(self, text: str) -> Optional[list[float]]:
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/api/embeddings", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None
        vec = data.get("embedding")
        if isinstance(vec, list) and vec:
            self.dim = len(vec)
            return [float(x) for x in vec]
        return None

    def embed_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        return [self.embed(t) for t in texts]

    def available(self) -> bool:
        return self.embed("ping") is not None


def get_embedder(backend: str = "none", model: str = DEFAULT_OLLAMA_MODEL,
                 url: str = DEFAULT_OLLAMA_URL) -> Embedder:
    """按名称获取嵌入后端；不可用/未知则返回 NullEmbedder。"""
    backend = (backend or "none").lower()
    if backend in ("none", "", "off"):
        return NullEmbedder()
    if backend == "ollama":
        return OllamaEmbedder(model=model, url=url)
    if backend in ("sentence-transformers", "st"):
        try:  # pragma: no cover - 依赖可选安装
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            return NullEmbedder()

        class _STEmbedder:
            name = f"st:{model}"

            def __init__(self) -> None:
                self._m = SentenceTransformer(model or "all-MiniLM-L6-v2")
                self.dim = int(self._m.get_sentence_embedding_dimension())

            def embed(self, text: str):
                return [float(x) for x in self._m.encode(text, normalize_embeddings=True)]

            def embed_batch(self, texts):
                return [self.embed(t) for t in texts]

        try:
            return _STEmbedder()
        except Exception:
            return NullEmbedder()
    return NullEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    """纯 Python 余弦相似度（零依赖；向量规模小时足够快）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))
