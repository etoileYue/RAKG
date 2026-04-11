# RAKG Web Backend 后端分析

本文档基于当前代码（`src/web/backend`）进行静态分析，聚焦后端架构、执行流程、接口设计、数据模型与风险点。

## 1. 后端概览

- 框架：FastAPI（`app/main.py`）
- 运行入口：`run.py`（`uvicorn.run("app.main:app", host="0.0.0.0", port=8000)`）
- 任务执行：单进程内串行执行器（`SerialTaskExecutor` + 内存队列 `deque` + 1 个后台线程）
- 存储：SQLite（`web_tasks.db`，封装在 `app/db.py`）
- 算法依赖：复用主工程 `src.kgAgent.NER_Agent`

核心能力分两类：

1. KG 构建任务（异步提交，后台串行执行）
2. QA 查询（同步调用）

## 2. 目录与职责

```text
src/web/backend
├── run.py                        # 服务启动
├── requirements.txt              # fastapi/uvicorn/pydantic
├── web_tasks.db                  # SQLite 任务库
└── app
    ├── main.py                   # 路由、生命周期、路径安全校验
    ├── schemas.py                # 请求/响应模型（Pydantic）
    ├── config.py                 # 环境变量配置、仓库路径常量
    ├── db.py                     # 数据访问层（任务表/日志表）
    ├── serial_executor.py        # 串行队列、取消、恢复、状态迁移
    └── services
        ├── kg_service.py         # KG 构建流程封装
        └── qa_service.py         # QA 查询封装（带索引缓存 key）
```

## 3. 启动与生命周期

- 进程启动后初始化 `Database`、`SerialTaskExecutor`、`QAService`
- FastAPI `lifespan`：
  - 启动时写系统日志 `RAKG web backend started`
  - 关闭时调用 `executor.shutdown()` 并记录停止日志

说明：

- `main.py` 会将仓库根目录加入 `sys.path`，用于导入 `src/*` 模块
- `SerialTaskExecutor` 在构造阶段会启动后台线程，并执行未完成任务恢复

## 4. KG 构建任务链路

### 4.1 提交阶段（API）

`POST /api/v1/tasks/kg-build`

- 输入模型：`KGBuildRequest`
  - `input_type`: `text` 或 `json_path`
  - 可选 `output_dir`、`existing_kg`
- `existing_kg` 会在路由层进行仓库内路径校验与后缀校验（必须 `.json`）
- 执行器 `submit_kg_task`：
  - 新建 `task_id`
  - 写 `tasks` 表，初始状态 `queued`
  - 入内存队列并唤醒 worker

### 4.2 执行阶段（worker）

执行器 `_run_task` 状态迁移：

1. `queued -> running`
2. 调用 `KGBuildService.run(...)`
3. 结束后进入终态 `succeeded/failed/canceled`

`KGBuildService.run` 关键流程：

1. 解析输出目录（默认 `data/web/tasks/<task_id>`）
2. 准备输入 JSON（`text` 会生成 `input_topics.json`）
3. 读取 topic 列表
4. 创建四类输出子目录：
   - `ner_data`
   - `rel_data`
   - `sim_data`
   - `RAKG_graph_re`
5. 遍历 topic，逐个调用 `NER_Agent.process(...)`
6. 汇总 `process_summary.json`
7. 返回 `output_payload`（包含 `graph_paths/latest_graph_path/summary` 等）

取消机制：

- `queued`：可立即从队列移除并标记 `canceled`
- `running`：仅打标 `cancel_requested`，在业务循环检查点触发中断

## 5. QA 查询链路

`POST /api/v1/qa/query`

- 入参：`kg_path/question/max_hop/seed_top_k/max_context_items`
- `QAService.ask`：
  - 解析图谱路径
  - 用 `path + mtime` 生成缓存 key（`web_qa:<md5>:<mtime>`）
  - 调用 `NER_Agent.initialize_qa_graph_index(...)`
  - 调用 `answer_question_with_kg(...)`
- 返回统一字段：
  - `formatted_answer`
  - `answer`
  - `retrieved_context`
  - `evidence_sources`
  - `graph_paths`
  - `intermediate.llm_output_raw`

## 6. 数据模型（SQLite）

数据库初始化在 `Database._init_schema()`，并启用 `PRAGMA journal_mode = WAL`。

### 6.1 tasks

核心字段：

- `task_id`（主键）
- `task_type`
- `status`（`queued/running/succeeded/failed/canceled`）
- `progress`（0~1）
- `message`
- `input_payload`（JSON）
- `output_payload`（JSON）
- `error_message`
- `created_at/started_at/finished_at`
- `cancel_requested`（0/1）

### 6.2 task_logs

- `id`（自增）
- `task_id`（可空）
- `level/source/message/created_at`

日志和状态更新都通过数据库层封装，写操作受 `_write_lock` 保护。

## 7. API 面清单

- `POST /api/v1/tasks/kg-build`
- `GET /api/v1/tasks/{task_id}`
- `GET /api/v1/tasks`
- `POST /api/v1/tasks/{task_id}/cancel`
- `POST /api/v1/qa/query`
- `GET /api/v1/logs`
- `GET /api/v1/system/health`
- `GET /api/v1/artifacts/kg`
- `GET /api/v1/artifacts/kg/candidates`
- `GET /`

## 8. 恢复与容错策略

服务重启时：

- 旧 `running` 任务：标记为 `failed`（进程内执行上下文丢失）
- 旧 `queued` 任务：重新入队
- 若 `queued` 同时被标记取消：转为 `canceled`

异常处理：

- KG worker 捕获异常后落库 `failed + error_message`
- QA 路由对 `FileNotFoundError` 返回 400，其它异常返回 500

## 9. 当前实现的主要风险点

1. 吞吐上限明显
- 仅 1 个 worker 串行执行，长任务会阻塞后续任务。

2. 运行中取消非即时
- `running` 任务依赖业务循环中的取消检查点，不是强中断。

3. 路径边界策略不一致
- `GET /artifacts/kg` 强制仓库内路径。
- 但 `KGBuildService._prepare_input_json` 的 `json_path`、`_resolve_output_dir`，以及 `QAService._resolve_kg_path` 对绝对路径限制较弱，存在越界读写风险（若服务暴露给非受信任调用方）。

4. 可观测性偏基础
- 目前是任务级日志与状态，缺少统一指标（耗时分布、失败分类、队列堆积告警）。

5. 安全能力缺口
- 未见鉴权、限流、审计策略；默认更适合内网或单租户受控环境。

## 10. 建议优化方向

1. 增加统一路径 allowlist 策略
- 对 `json_path`、`output_dir`、`kg_path` 与 artifact 读取采用一致的仓库内或白名单目录限制。

2. 增加并发配置能力
- 支持可控 worker 数（例如环境变量配置），并配套资源上限保护。

3. 增强任务可观测性
- 补充关键指标：排队时长、执行时长、topic 成功率、失败类型统计。

4. 完善自动化测试
- 覆盖数据库状态迁移、取消语义、恢复逻辑、关键 API 参数校验与错误码。

