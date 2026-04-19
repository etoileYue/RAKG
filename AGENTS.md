# Repository Guidelines
## 项目介绍
- 该项目为知识图谱与RAG融合架构，使用知识图谱增强RAG，同时附有前后端。具体架构流程可参考`docs/RAKG_架构流程总结.md`
## 项目结构与模块组织
- `src/`：核心 Python 代码。`src/construct/` 放主入口（KG 构建、QA 检索、naive baseline），`src/pipeline/` 放可组合流水线算子，`src/eval/` 放评测与可视化脚本。`src/kgAgent.py`主要编排构图过程。
- `src/web/`：Web 工作台。`backend/` 为 FastAPI + SQLite 任务队列，`frontend/` 为 React + Vite。
- `dataset/`：数据转换工具与单元测试（当前主要是 Multi-Doc-QA-Chinese 转换）。
- `data/`、`logs/`：运行产物与样例数据目录。提交前避免新增大体积中间文件。
