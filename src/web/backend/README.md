# RAKG Web Backend 说明文档

本文档基于当前目录下代码实现（`src/web/backend`）整理，目标是快速说明后端架构、执行流程、接口能力与已知注意点。

## 1. 模块结构与职责

```text
src/web/backend
├── run.py                      # 启动入口（uvicorn）
├── requirements.txt            # 后端依赖
├── web_tasks.db                # SQLite 任务与日志库（运行期生成/更新）
└── app
    ├── main.py                 # FastAPI 路由与生命周期
    ├── config.py               # 环境变量配置与路径常量
    ├── schemas.py              # Pydantic 请求/响应模型
    ├── db.py                   # SQLite 访问封装
    ├── serial_executor.py      # 串行任务执行器（单 worker）
    └── services
        ├── kg_service.py       # KG 构建服务（封装 NER_Agent.process）
        └── qa_service.py       # QA 查询服务（封装图索引+问答）
```

## 2. 运行模型总览

- Web 框架：FastAPI
- 任务执行模型：单进程、单线程 worker 串行执行（内存队列 `deque`）
- 持久化：SQLite（任务表 + 日志表）
- 算法调用：复用仓库主逻辑 `src.kgAgent.NER_Agent`
- 任务类型：当前仅支持 `kg_build`

## 3. 关键流程

### 3.1 KG 构建任务流程（异步提交 + 后台串行执行）

1. 前端调用 `POST /api/v1/tasks/kg-build`
2. `SerialTaskExecutor.submit_kg_task`：
   - 生成 `task_id`
   - 写入 `tasks` 表（状态 `queued`）
   - 入内存队列
3. worker 线程取任务后置为 `running`
4. `KGBuildService.run`：
   - 解析输入（文本或 JSON 文件）
   - 创建输出目录与子目录
   - 按 topic 调用 `NER_Agent.process(...)`
   - 写 `process_summary.json`
   - 返回输出路径、图谱路径等 `output_payload`
5. 执行器更新任务状态：
   - 成功 -> `succeeded`
   - 异常 -> `failed`
   - 取消 -> `canceled`

### 3.2 QA 查询流程（同步）

1. 前端调用 `POST /api/v1/qa/query`
2. `QAService.ask`：
   - 解析/校验 `kg_path`
   - 构造缓存 key（路径 + mtime 的 md5）
   - 调用 `initialize_qa_graph_index(...)`
   - 调用 `answer_question_with_kg(...)`
3. 返回标准化字段：
   - `answer`
   - `formatted_answer`
   - `retrieved_context`
   - `evidence_sources`
   - `graph_paths`

## 4. 状态机与取消语义

任务状态定义：`queued | running | succeeded | failed | canceled`

- `queued` 取消：立即从内存队列移除并置 `canceled`
- `running` 取消：仅标记 `cancel_requested`，在执行阶段轮询检查后中断
- 重启恢复：
  - 旧 `running` 会被标记为 `failed`（进程内执行上下文丢失）
  - 旧 `queued` 会重新入队（若已标记取消则置 `canceled`）

## 5. 数据存储（SQLite）

数据库文件：`src/web/backend/web_tasks.db`

### 5.1 `tasks` 表

- 核心字段：
  - `task_id`（主键）
  - `task_type`
  - `status`
  - `progress`
  - `message`
  - `input_payload`（JSON 字符串）
  - `output_payload`（JSON 字符串）
  - `error_message`
  - `created_at/started_at/finished_at`
  - `cancel_requested`（0/1）

### 5.2 `task_logs` 表

- 字段：
  - `id`（自增）
  - `task_id`（可空，系统日志可无 task）
  - `level`
  - `source`
  - `message`
  - `created_at`

## 6. API 清单（`/api/v1`）

- `POST /tasks/kg-build`：创建图谱构建任务
- `GET /tasks/{task_id}`：查询任务详情
- `GET /tasks`：分页任务列表（支持 `task_type/status` 过滤）
- `POST /tasks/{task_id}/cancel`：取消任务
- `POST /qa/query`：图谱问答
- `GET /logs`：分页日志查询（支持 `task_id/keyword`）
- `GET /system/health`：系统健康与模型配置摘要
- `GET /artifacts/kg?path=...`：读取图谱 JSON（限制仓库内路径）
- `GET /`：根健康提示

## 7. 配置项（环境变量）

- `RAKG_WEB_APP_NAME`：应用名
- `RAKG_WEB_ENV`：运行环境标记
- `RAKG_WEB_DB_PATH`：SQLite 路径（默认 `src/web/backend/web_tasks.db`）
- `RAKG_WEB_QUEUE_POLL_SEC`：队列轮询间隔（默认 `0.2`）
- `RAKG_WEB_CORS_ORIGINS`：CORS 白名单（逗号分隔）
- `RAKG_WEB_RELOAD`：是否启用 uvicorn reload（`run.py`）

## 8. 与主仓库算法层的耦合点

- `KGBuildService`、`QAService` 直接依赖 `src.kgAgent.NER_Agent`
- 模型配置由 `src/config.py` 提供，并通过 `/system/health` 暴露摘要
- 后端在 `main.py` 启动时将仓库根目录加入 `sys.path`，保证可导入 `src/*`

## 9. 已识别的实现特征与风险

1. 单 worker 串行执行
   - 优点：实现简单、状态一致性较好
   - 代价：多任务并发吞吐受限

2. 取消粒度受限
   - `running` 任务依赖业务循环中的 `is_cancel_requested()` 检查点
   - 若单 topic 处理耗时长，取消不会立即生效

3. 路径安全策略不完全一致
   - `GET /artifacts/kg` 限制了“仓库内路径”
   - 但 `KGBuildService._resolve_output_dir` 与 `QAService._resolve_kg_path` 支持绝对路径且未统一限制仓库边界
   - 若对外暴露，建议统一增加 allowlist 或仓库内约束

4. 配置安全
   - `src/config.py` 当前包含明文 API Key（仓库级问题）
   - 建议改为环境变量注入并从代码中移除密钥

## 10. 可维护性建议（下一步）

1. 增加后端单元测试与接口测试
   - `db.py`（状态迁移、分页、过滤）
   - `serial_executor.py`（恢复逻辑、取消逻辑）
   - `main.py`（接口参数校验、错误码）

2. 统一路径访问策略
   - 所有外部传入路径默认限制在仓库目录或配置白名单内

3. 引入可选并发策略
   - 在保证资源可控前提下，支持 N worker 并发执行 KG 任务

4. 强化可观测性
   - 增加任务耗时指标、topic 级耗时统计、失败分类统计

