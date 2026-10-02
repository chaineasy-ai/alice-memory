"""Frontmatter 读写：YAML frontmatter 的解析与序列化。

优先使用 PyYAML（生产/测试依赖）；不可用时降级到受限 YAML 子集解析器，
保证核心功能零依赖也能跑。受支持子集：标量、行内列表 ``[a, b]``、块状列表 ``- x``。
"""

from __future__ import annotations

import json
import re
from typing import Any

try:  # pragma: no cover - 取决于环境
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # pragma: no cover
    _HAS_YAML = False


_FM_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """把文件文本拆成 (frontmatter dict, body)。无 frontmatter 时返回 ({}, 原文)。"""
    m = _FM_RE.match(text or "")
    if not m:
        return {}, text or ""
    raw = m.group(1)
    body = text[m.end():]
    return _parse_yaml(raw), body


def _parse_yaml(raw: str) -> dict[str, Any]:
    if _HAS_YAML:
        data = yaml.safe_load(raw)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError("frontmatter 必须是 YAML 映射")
        return data
    return _parse_restricted(raw)


def _parse_restricted(raw: str) -> dict[str, Any]:
    """受限 YAML 子集解析：仅顶层 key，标量/行内列表/块状列表。"""
    data: dict[str, Any] = {}
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if line.startswith((" ", "\t")):
            i += 1
            continue
        m = re.match(r"^([\w\-]+):\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key, value = m.group(1), m.group(2).strip()
        if value == "":
            # 可能是块状列表
            items: list[Any] = []
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith("- "):
                items.append(_scalar(lines[j].lstrip()[2:].strip()))
                j += 1
            if items:
                data[key] = items
                i = j
                continue
            data[key] = None
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            data[key] = [_scalar(x.strip()) for x in inner.split(",")] if inner else []
        else:
            data[key] = _scalar(value)
        i += 1
    return data


def _scalar(text: str) -> Any:
    if text.startswith(('"', "'")) and text.endswith(('"', "'")) and len(text) >= 2:
        return text[1:-1]
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    try:
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        if re.fullmatch(r"-?\d+\.\d+", text):
            return float(text)
    except ValueError:
        pass
    return text


def render(meta: dict[str, Any], body: str) -> str:
    """把 frontmatter dict + body 渲染为完整 Markdown 文本。"""
    fm = _dump_yaml(meta)
    body = (body or "").lstrip("\n")
    return f"---\n{fm}---\n\n{body}" if body else f"---\n{fm}---\n"


def _dump_yaml(meta: dict[str, Any]) -> str:
    if _HAS_YAML:
        return yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return _dump_restricted(meta)


def _dump_restricted(meta: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in meta.items():
        if isinstance(value, (list, tuple)):
            if all(not isinstance(v, (dict, list)) for v in value):
                lines.append(f"{key}: [{', '.join(_dump_scalar(v) for v in value)}]")
            else:
                lines.append(f"{key}:")
                for v in value:
                    lines.append(f"  - {_dump_scalar(v)}")
        elif isinstance(value, dict):
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        else:
            lines.append(f"{key}: {_dump_scalar(value)}")
    return "\n".join(lines) + "\n"


def _dump_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or re.search(r"[:#\[\]{},&*!|>'\"%@`\n]", text) or text.strip() != text:
        return json.dumps(text, ensure_ascii=False)
    return text
