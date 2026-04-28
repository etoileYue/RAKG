# db.py 说明文档

本文档对应文件：`src/web/backend/app/db.py`。

## 1. 文件整体作用

`db.py` 是一个轻量的 SQLite 数据访问层（DAO），核心职责如下：

- 初始化数据库与表结构（任务表、日志表、索引）。
- 为任务系统提供统一的增删改查接口（当前实现主要是增、改、查）。
- 负责任务日志写入与日志分页查询。
- 处理数据库行与业务对象之间的转换（JSON 字段反序列化、布尔字段转换）。
- 通过线程锁保证写操作在多线程下的安全性。

该模块主要围绕 `Database` 类工作，外加一个时间工具函数 `utc_now_iso()`。

---

## 2. 函数分析

### 2.1 `utc_now_iso() -> str`

功能：

- 返回当前 UTC 时间的 ISO 8601 字符串。

用途：

- 作为任务创建时间、日志时间等时间戳的统一格式来源。

---

### 2.2 `Database.__init__(self, db_path: str)`

功能：

- 保存数据库文件路径。
- 自动创建数据库文件所在目录（不存在时）。
- 初始化线程写锁 `self._write_lock`。
- 调用 `_init_schema()` 完成数据库结构初始化。

注意点：

- 启动即初始化 schema，避免运行时首次操作失败。

---

### 2.3 `Database._connect(self) -> Iterator[sqlite3.Connection]`

功能：

- 提供连接上下文管理器。
- 每次进入时新建连接；退出时确保关闭连接。
- 设置 `conn.row_factory = sqlite3.Row`，使查询结果可按字段名访问。

作用：

- 封装连接生命周期，降低连接泄漏风险。

---

### 2.4 `Database._init_schema(self) -> None`

功能：

- 通过 `executescript` 执行多条 SQL：
  - 开启 `WAL` 日志模式。
  - 创建 `tasks` 表（若不存在）。
  - 创建 `task_logs` 表（若不存在）。
  - 创建任务和日志相关索引（若不存在）。
- 最后提交事务。

作用：

- 统一数据库初始化入口，保证应用可重复启动且幂等。

---

### 2.5 `Database.create_task(self, task_id, task_type, input_payload) -> None`

功能：

- 插入一条新任务记录。
- 默认字段：
  - `status='queued'`
  - `progress=0`
  - `message='queued'`
  - `output_payload='{}'`
- `input_payload` 会序列化为 JSON 字符串。

并发处理：

- 写操作在 `self._write_lock` 下执行，避免并发写冲突。

---

### 2.6 `Database.log(self, level, source, message, task_id=None) -> None`

功能：

- 向 `task_logs` 插入一条日志。
- `level` 会被标准化为大写（如 `info -> INFO`）。
- 支持 `task_id` 为空（系统级日志）。

并发处理：

- 写操作同样受写锁保护。

---

### 2.7 `Database.get_task(self, task_id) -> dict[str, Any] | None`

功能：

- 按主键 `task_id` 查询单个任务。
- 若存在，调用 `_row_to_task()` 转为业务字典。
- 不存在则返回 `None`。

---

### 2.8 `Database.get_tasks_by_status(self, statuses) -> list[dict[str, Any]]`

功能：

- 按多个状态批量查询任务（`WHERE status IN (...)`）。
- 按 `created_at ASC` 升序返回。

注意点：

- `statuses` 为空时直接返回空列表。

---

### 2.9 `Database.list_tasks(self, page, page_size, task_type, status) -> dict[str, Any]`

功能：

- 任务分页查询。
- 支持可选过滤条件：`task_type`、`status`。
- 返回结构：
  - `total`：总数
  - `page`：当前页
  - `page_size`：页大小
  - `items`：当前页任务列表

实现方式：

- 先查总数，再按 `LIMIT/OFFSET` 查询当前页。
- 按 `created_at DESC` 倒序。

---

### 2.10 `Database.list_tasks_by_filters(self, *, task_type=None, status=None) -> list[dict[str, Any]]`

功能：

- 非分页的筛选查询。
- 支持 `task_type`、`status` 可选过滤。
- 按 `created_at DESC` 返回全部匹配项。

---

### 2.11 `Database.update_task_state(...) -> None`

功能：

- 动态更新任务状态相关字段，只更新调用方传入的字段。
- 可更新字段包括：
  - `status`
  - `progress`
  - `message`
  - `error_message`
  - `output_payload`（会 JSON 序列化）
  - `started_at`
  - `finished_at`
  - `cancel_requested`

关键逻辑：

- `progress` 会被夹紧到 `[0.0, 1.0]`。
- 若没有任何字段要更新，直接返回。
- 最终拼接 `UPDATE tasks SET ... WHERE task_id = ?` 执行。

并发处理：

- 受写锁保护。

---

### 2.12 `Database.fetch_logs(self, page, page_size, task_id, keyword) -> dict[str, Any]`

功能：

- 日志分页查询。
- 支持按 `task_id` 过滤。
- 支持按 `message LIKE '%keyword%'` 模糊过滤。
- 返回结构同分页接口：`total/page/page_size/items`。

排序：

- 按 `id DESC` 返回最新日志在前。

---

### 2.13 `Database._row_to_task(row) -> dict[str, Any]`（静态方法）

功能：

- 将 `sqlite3.Row` 转换为业务字典。
- 对字段做类型/结构转换：
  - `input_payload`：JSON 字符串 -> `dict`
  - `output_payload`：JSON 字符串 -> `dict`
  - `cancel_requested`：整数 -> `bool`

作用：

- 统一任务查询结果格式，减少上层重复解析逻辑。

---

## 3. 表结构说明

### 3.1 `tasks` 表

用途：

- 存储任务的生命周期状态与输入输出数据。

字段：

| 字段名 | 类型 | 约束/默认值 | 说明 |
|---|---|---|---|
| `task_id` | `TEXT` | `PRIMARY KEY` | 任务唯一标识 |
| `task_type` | `TEXT` | `NOT NULL` | 任务类型 |
| `status` | `TEXT` | `NOT NULL` | 任务状态（如 queued/running/succeeded/failed/canceled） |
| `progress` | `REAL` | `NOT NULL DEFAULT 0` | 任务进度，约定 0~1 |
| `message` | `TEXT` | `NOT NULL DEFAULT ''` | 任务状态说明 |
| `input_payload` | `TEXT` | `NOT NULL` | 输入参数（JSON 字符串） |
| `output_payload` | `TEXT` | `NOT NULL DEFAULT '{}'` | 输出结果（JSON 字符串） |
| `error_message` | `TEXT` | 可空 | 错误信息 |
| `created_at` | `TEXT` | `NOT NULL` | 创建时间（ISO UTC） |
| `started_at` | `TEXT` | 可空 | 开始执行时间 |
| `finished_at` | `TEXT` | 可空 | 结束时间 |
| `cancel_requested` | `INTEGER` | `NOT NULL DEFAULT 0` | 是否请求取消（0/1） |

---

### 3.2 `task_logs` 表

用途：

- 存储任务运行日志与系统日志。

字段：

| 字段名 | 类型 | 约束/默认值 | 说明 |
|---|---|---|---|
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | 自增主键 |
| `task_id` | `TEXT` | 可空 | 所属任务 ID，系统日志可为空 |
| `level` | `TEXT` | `NOT NULL` | 日志级别（INFO/WARN/ERROR 等） |
| `source` | `TEXT` | `NOT NULL` | 日志来源模块 |
| `message` | `TEXT` | `NOT NULL` | 日志内容 |
| `created_at` | `TEXT` | `NOT NULL` | 日志时间（ISO UTC） |

---

### 3.3 索引

- `idx_tasks_status_created`：`tasks(status, created_at DESC)`
  - 作用：加速按状态筛选并按创建时间排序的查询。
- `idx_task_logs_task_created`：`task_logs(task_id, created_at DESC)`
  - 作用：加速按任务筛选日志并按时间排序的查询。

---

## 4. 查询语句说明（按代码出现顺序）

以下列出 `db.py` 中所有 SQL 语句，并说明其用途。

### 4.1 `_init_schema()` 中的 SQL

1. `PRAGMA journal_mode = WAL;`
- 用途：设置 SQLite 为 WAL 模式，改善并发读写能力。

2. `CREATE TABLE IF NOT EXISTS tasks (...)`
- 用途：创建任务表；若已存在则跳过。

3. `CREATE TABLE IF NOT EXISTS task_logs (...)`
- 用途：创建日志表；若已存在则跳过。

4. `CREATE INDEX IF NOT EXISTS idx_tasks_status_created ON tasks(status, created_at DESC);`
- 用途：为任务状态+创建时间查询建立索引。

5. `CREATE INDEX IF NOT EXISTS idx_task_logs_task_created ON task_logs(task_id, created_at DESC);`
- 用途：为日志任务ID+创建时间查询建立索引。

### 4.2 `create_task()` 中的 SQL

6. `INSERT INTO tasks(...) VALUES(?, ?, 'queued', 0, 'queued', ?, '{}', ?)`
- 用途：创建新任务。
- 参数绑定：`task_id`, `task_type`, `json(input_payload)`, `created_at`。
- 特点：使用占位符防止 SQL 注入。

### 4.3 `log()` 中的 SQL

7. `INSERT INTO task_logs(task_id, level, source, message, created_at) VALUES(?, ?, ?, ?, ?)`
- 用途：插入日志记录。
- 参数绑定：`task_id`, `level.upper()`, `source`, `message`, `created_at`。

### 4.4 `get_task()` 中的 SQL

8. `SELECT * FROM tasks WHERE task_id = ?`
- 用途：按主键读取单个任务。

### 4.5 `get_tasks_by_status()` 中的 SQL

9. 
```sql
SELECT * FROM tasks
WHERE status IN (?, ?, ...)
ORDER BY created_at ASC
```
- 用途：按状态集合读取任务列表。
- 说明：占位符数量由 `statuses` 长度动态生成。

### 4.6 `list_tasks()` 中的 SQL

10. `SELECT COUNT(1) AS cnt FROM tasks {where_clause}`
- 用途：分页前先计算总数。
- `where_clause` 可能为空，或包含：
  - `task_type = ?`
  - `status = ?`

11. 
```sql
SELECT * FROM tasks
{where_clause}
ORDER BY created_at DESC
LIMIT ? OFFSET ?
```
- 用途：获取当前页任务。
- 参数绑定：过滤条件参数 + `page_size` + `offset`。

### 4.7 `list_tasks_by_filters()` 中的 SQL

12. 
```sql
SELECT * FROM tasks
{where_clause}
ORDER BY created_at DESC
```
- 用途：非分页筛选任务列表。
- `where_clause` 同样由 `task_type/status` 动态拼接。

### 4.8 `update_task_state()` 中的 SQL

13. `UPDATE tasks SET <动态字段列表> WHERE task_id = ?`
- 用途：按任务 ID 更新部分字段。
- 说明：
  - 仅更新传入参数对应字段（partial update）。
  - 绑定参数顺序与 `fields` 构造顺序一致。
  - `output_payload` 更新时会先 JSON 序列化。

### 4.9 `fetch_logs()` 中的 SQL

14. `SELECT COUNT(1) AS cnt FROM task_logs {where_clause}`
- 用途：日志分页前统计总数。
- 可选过滤：
  - `task_id = ?`
  - `message LIKE ?`

15. 
```sql
SELECT id, task_id, level, source, message, created_at
FROM task_logs
{where_clause}
ORDER BY id DESC
LIMIT ? OFFSET ?
```
- 用途：查询当前页日志。
- 排序策略：按自增 `id` 倒序，最新日志优先。

---

## 5. 补充说明

- 当前实现未对 `task_logs.task_id` 建立外键约束；这意味着日志可独立存在，不强制引用有效任务。
- 所有写操作（创建任务、写日志、更新任务）都通过线程锁串行化，适合单进程多线程场景。
- 时间字段统一使用字符串存储，格式为 ISO UTC，便于跨语言与 API 输出。
