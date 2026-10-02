# alice-memory

C-Land 自研的 **markdown 文件式记忆系统**（#230 M3 MVP）。
`.md` 是唯一真源，SQLite 索引全部派生、可重建；不依赖 Obsidian 桌面 app、不依赖付费服务。

> 契约（冻结）：`cland-research/docs/markdown-memory-原理调研.md` §4.6/§4.7/§4.8
> 计划：`cland-research/docs/markdown-memory-研究计划-20261002.md`
> 评测口径：`cland-research/docs/markdown-memory-评测-指标口径与基线.md`

## 1. 定位与机制映射（复现参考实现）

| 参考实现 | 借鉴的核心机制 | 在本系统的落地 |
|---|---|---|
| **MemGPT / Letta** | 分层记忆（main context / external context）+ **自编辑记忆**（append/replace）+ 检索工具 | 分层目录 `core/semantic/episodic/...`（core=常驻）；`mem add/update` 自编辑；`mem search/get` 检索工具；`mem context` 装配可注入 prompt 的块 |
| **mem0** | 写入时 **ADD / UPDATE / NOOP 判定 + 内容指纹去重 + 关联链接** | `MemoryManager.add` → `added/duplicate/superseded`；`content_hash`(md5) 幂等；`links`/`supersedes` 保留历史 |
| **Zep / Graphiti** | **时序知识**（事实带 valid 区间）+ 图谱 | frontmatter `created/updated/last_accessed` + 链接邻接表；召回含 recency 衰减 |
| **Obsidian** | `.md` + **`[[wikilinks]]`** + backlinks + 图谱 | frontmatter `links` + 链接邻接表 `neighbors(in/out)`；召回 1 跳图加成 |
| **Generative Agents** | recency × importance × relevance 打分 | 召回 `w_rel·rel + w_rec·recency + w_imp·importance + w_graph·graph`，`recency=0.5^(age/half_life)` |

> 复现结论（供 #230 原理报告）：MemGPT 的「分层 + 自编辑」可用**目录 + CLI**无损表达；
> mem0 的可迁移内核是**写入判定（ADD/UPDATE/NOOP）+ 去重**，其向量库/LLM 抽取不是必需；
> 文件式方案以「可人读/可审计/可移植」换「大规模/强一致」，适合 agent 长记忆与团队知识沉淀。

## 2. 目录布局（契约 §4.2 / §4.7）

```
<root>/
├── memory/                              # 真源（.md，git 管理）
│   ├── core/ semantic/ episodic/ entity/ procedural/ inbox/ archive/   # 共享命名空间
│   └── agents/<agent-id>/<layer>/       # 私有命名空间（仅该 agent 写）
└── .mem/                                # 派生（建议 gitignore；可删，reindex 重建）
    ├── index.sqlite                     # 元数据 + FTS5(trigram+BM25) + links 邻接表
    └── locks/<ns>.lock                  # flock 锁文件
```

## 3. frontmatter schema（契约 §4.6）

必填 7：`id / title / type / layer / created / updated / status`；
生命周期：`importance(1-10) / last_accessed / access_count / links / supersedes / superseded_by / expires`；
溯源：`source / source_type / confidence / summary / owner / scope / domain`；未知字段保留在 `extra`。

```markdown
---
id: mem-20261002-3de551
title: bge-m3 中文选型
type: note
layer: semantic
status: active
importance: 8
tags: [embedding, rag]
links: []
source: alice-research-hub/docs/p40-stack/03-embedding-selection.md
source_type: doc
created: 2026-10-02T23:30:00+08:00
updated: 2026-10-02T23:30:00+08:00
---

bge-m3 在中文 embedding 上完胜 nomic。
```

## 4. CLI

```bash
export PYTHONPATH=src            # 或 pip install -e .
export MEM_HOME=~/.mem           # 记忆根（别名 MEM_ROOT）
export MEM_AGENT=hopper          # 默认写命名空间 → agents/hopper

mem init
mem add --title "..." --text "..." --layer semantic --tags a b --importance 7
mem add --file note.md --type entity            # 从 md（含 frontmatter）导入
mem search "查询" --k 5 --mode lexical --json    # 词法（v0）；hybrid 为 v0.1
mem search "查询" --context --budget-tokens 2000  # 召回 + 装配上下文
mem get <id>
mem update <id> --append "补充" --add-tags x
mem link <id> <to-id> [--unlink]
mem gc --dry-run | --apply [--archive-stale 180]
mem reindex            # 从 .md 全量重建索引（D1 红线可验证）
mem stats
```

**JSON 契约（D11）**：所有输出顶层含 `schema_version`（`mem.search.v1` 等），字段只增不删。

```json
{"schema_version":"mem.search.v1","mode":"lexical","count":2,
 "results":[{"id":"mem-...","score":1.95,"path":"memory/semantic/mem-....md",
   "title":"...","layer":"semantic","namespace":"shared","snippet":"...",
   "why":"lexical","components":{"rel":1.0,"recency":0.99,"importance":0.7,"graph":0.0}}]}
```

## 5. 检索（可插拔，契约 §4.4）

- **v0 `lexical`**：SQLite **FTS5 trigram + BM25**（不手搓 grep）；`<3` 字 CJK 查询用 LIKE 兜底。
- **v0.1 `hybrid`**：FTS5 + bge-m3(`:11435`) 向量 → **RRF** 融合 → 可选 bge-reranker。
  `Retriever` 接口抽象，不改存储/CLI；`--mode auto`（默认）：库中无向量或 embedder 不可达 → lexical。
- 重排：`w_rel·rel + w_rec·recency + w_imp·importance + w_graph·graph`。

## 6. 并发（契约 §4.7）

- **读**：无锁、递归全部命名空间，不阻塞写者。
- **写**：按命名空间 `flock` 互斥（`shared` 一把；每 agent 一把）+ 同目录 tmp + `fsync` + `os.replace` 原子写。
- 私有命名空间 `agents/<id>/` 天然无竞争；跨 agent 沉淀 = 写私有后 `promote` 到共享。
- 验收测试：`test_concurrent_adds_same_namespace`（6 进程并发 `mem add` → 计数 = 6）。

## 7. 测试 / 复现

```bash
PYTHONPATH=src python3 -m pytest tests/ -q     # 20 项：模型/存储/索引/生命周期/召回/CLI/并发/§4.8
./bootstrap.sh test                            # 同上（封装）
```

实测环境：Python 3.13 + SQLite（FTS5 trigram 可用）、零付费、无外部服务。
500 语料 lexical 召回 p50 11.9ms。

## 8. 集成契约（§4.8）

- **I1 读不回写**：`search` 默认不更新 `last_accessed/access_count`；`--touch` 才 best-effort。
- **I2 context**：确定性文本块，固定分隔头 `<!-- mem: id=<id> layer=<layer> source=<source> -->`；
  稳定排序（score desc, id asc）；`--budget-tokens N` / `--budget-unit tokens|chars`；无命中降为空串（不报错）。
- **I3 退出码**：`0` 成功 ｜ `2` 参数错 ｜ `3` 未找到 ｜ `4` 锁超时 ｜ `5` IO/索引错 ｜ `6` 降级（hybrid→lexical，仍返回结果）。
- **I4 幂等**：`add` 默认内容 hash NOOP；`--idempotency-key K` 存入 frontmatter，同 K 重放 → NOOP 并回既有 id。
- **I5 stats**：`mem stats --json` 固定字段 `total/by_namespace/by_layer/by_status/gc_candidates/index_schema/vectors/links`。

## 9. 局限（诚实标注）

- 只实现 **v0 lexical**；`hybrid` 已留接口，向量后端（Ollama bge-m3）**待本机服务启动后验证**。
- SQLite FTS5 **trigram 对 <3 字查询失效**（已用 LIKE 兜底，未做 CJK 单字切分索引）；
  2 字/单字 query 的召回质量**待 curie 冻结题集复测**。
- 跨机/多机并发不承诺（仅同机 flock）；`index.sqlite` 并发写靠 SQLite busy-timeout。
- 未含 LLM 事实抽取（mem0 的抽取式写入为可选增强，非 v0 契约）。
- `owner` 只读共享 / promote 审核策略为预留字段，未实现强制。
