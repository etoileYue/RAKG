# RAKG Web (V1)

基于 `src/web/plan.md` 的实现，包含：
- FastAPI 后端（任务队列 + SQLite + KG/QA/日志/健康接口）
- React 前端工作台（KG 构建、QA 检索、日志中心、系统状态）

## 目录结构

- `src/web/backend/`：FastAPI 服务
- `src/web/frontend/`：React + Vite 前端

## 启动后端

```bash
# 在仓库根目录
pip install -r src/web/backend/requirements.txt
python src/web/backend/run.py
```

默认监听 `http://127.0.0.1:8000`，API 前缀为 `/api/v1`。
如需热重载可设置：`RAKG_WEB_RELOAD=true`。

## 启动前端

```bash
cd src/web/frontend
npm install
npm run dev
```

默认监听 `http://127.0.0.1:5173`。

如果后端地址不是默认值，可设置：

```bash
VITE_API_BASE=http://127.0.0.1:8000/api/v1
```

## 关键接口

- `POST /api/v1/tasks/kg-build`
- `GET /api/v1/tasks/{task_id}`
- `GET /api/v1/tasks`
- `POST /api/v1/tasks/{task_id}/cancel`
- `POST /api/v1/qa/query`
- `GET /api/v1/logs`
- `GET /api/v1/system/health`
- `GET /api/v1/artifacts/kg?path=...`（前端读取图谱 JSON）

## 数据与持久化

- SQLite：`src/web/backend/web_tasks.db`
- 默认任务输出目录：`data/web/tasks/<task_id>/`
