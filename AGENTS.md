# Repository Guidelines
## 项目介绍
- 该项目为知识图谱与RAG融合架构，使用知识图谱增强RAG，同时附有前后端。具体架构流程可参考`docs/RAKG_架构流程总结.md`
## 项目结构与模块组织
- `src/`：核心 Python 代码。`src/construct/` 放主入口（KG 构建、QA 检索、naive baseline），`src/pipeline/` 放可组合流水线算子，`src/eval/` 放评测与可视化脚本。`src/kgAgent.py`主要编排构图过程。
- `src/web/`：Web 工作台。`backend/` 为 FastAPI + SQLite 任务队列，`frontend/` 为 React + Vite。
- `dataset/`：数据转换工具与单元测试（当前主要是 Multi-Doc-QA-Chinese 转换）。
- `data/`、`logs/`：运行产物与样例数据目录。提交前避免新增大体积中间文件。
## 论文撰写
- 该项目为本科毕业设计项目，最终需要撰写毕业论文。
- `docs/论文/初稿.md`: 论文初稿（待完成），对于初稿写作/修改要求可以直接读写该文件。
- `docs/论文/大纲.md`: 论文大纲
- 补充：在论文撰写过程中，不将本项目命名为"RAKG"，论文中不要出现"RAKG"作为名字。
- 如果撰写过程中出现需要用户补充的内容（如图片、数据等等），可以在初稿中以"//TODO"开头提示用户。