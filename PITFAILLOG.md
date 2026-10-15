---
title: "PITFAILLOG — alice-memory 踩坑记录"
summary: "记录 alice-memory（mem CLI / 记忆存储 / 检索）的踩坑（根因与机制），动手前先查；新坑按四段（症状/根因/修复/预防）倒序追加"
read_when:
  - "改 alice-memory 写入/去重/取代/命名空间逻辑前"
  - "记录 alice-memory 新旧坑时"
scope:
  - alice-memory
status: "active"
updated: "2026-10-16"
---
# PITFAILLOG — alice-memory 踩坑记录

> 记录**为什么**踩坑（根因/机制）；CHANGELOG 只记录改了什么。
> 动手前先查本文件；出现已记录症状时，先按「修复」方向验证，不要重新排查。
> 章节内按日期倒序（最新在上）。

---

## 写入 / 命名空间

### [add/supersede] 同题不同正文 → 取代写回**丢失命名空间**，落回 `shared`
- **日期**：2026-10-16 · **模块**：`lifecycle.py`（`add` → `_find_duplicate` → `_supersede`）· `store.py::path_for`
- **症状**：以 `MEM_AGENT=hopper`（写者命名空间 `agents/hopper`）调用
  ```bash
  mem add "T" "body one" --layer procedural   # => added      agents/hopper  ✅
  mem add "T" "body two" --layer procedural   # => superseded shared          ❌ 期望 agents/hopper
  mem add "T" "body one" --layer procedural   # => duplicate  agents/hopper  ✅
  ```
  同 **title**、不同正文的第 2 条在「取代（supersede）」路径下**落进了顶层 `shared/`**，而非写者的 `agents/<id>/`。
- **根因**：`_supersede` 的落盘命名空间**不是**由写者传入，而是从 `new.path` 反推；而 `new` 此时**尚未落盘**、`new.path is None` → `path_for(new)` 用默认 `namespace=SHARED` → 落回 `shared`。**取代路径从未把写者命名空间传下去**。
- **影响**：破坏「私有命名空间隔离」（契约 §4.7 / CI-026 / CI-033）。实测：PITFAILLOG 冷启动批导入（`#463`）中 **5 条同题坑**（「[接口暴露] 测试用 Redis 通用读写端点…6 服务复制传播」来自 5 个 chainpay 仓）**全部泄漏进 `shared/`**。
- **修复**（commit `91b521c`）：`add` → `_find_duplicate(note, ns)` → `_supersede(old, new, ns)`，取代时**沿用写者 `ns`**；新增回归 `test_supersede_preserves_namespace`（`pytest tests/ -q` → **31 passed**；CLI 复跑 → `superseded ns=agents/hopper` ✅）。
- **预防**：
  - 任何「写回 / 取代 / 归档」路径，**命名空间必须显式传递**，禁止从「尚未落盘的 note.path」反推（默认值会静默兜底到 `SHARED`）。
  - 涉及路径/命名空间的写入逻辑，回归用例必须**断言落盘命名空间**（不只看 event）。
- **来源**：`base/cland-crawler#463` 单 agent 记忆试点（hopper 发现 + sage 独立复现与修复，2026-10-16）；`alice-memory/src/alice_memory/lifecycle.py`、`store.py::path_for`；`https://github.com/chaineasy-ai/alice-memory` commit `91b521c`。
