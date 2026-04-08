 ## RAKG 网页端接入实施计划（FastAPI + React，V1: KG/QA/日志）

  ### Summary

  - 新增独立后端服务（FastAPI）作为现有脚本能力的 API 封装层，不改动
    核心算法 pipeline。
  - 新增独立前端（React）作为工作台，覆盖 KG 构建 + QA 检索 + 日志查
    看，采用异步任务+轮询。
  - 前后端分离部署；任务与历史记录持久化到 SQLite；任务执行策略为单
    机串行队列。
  - 配置改为环境变量驱动（密钥不入库），前端仅只读展示当前模型配置与
    健康状态。

  ### Key Changes

  - 后端服务层
      - 新建 backend 子工程（FastAPI），封装现有 NER_Agent、
        qa_retrieval_entry、logger 能力。
      - 引入任务管理模块：task_id、状态机（queued/running/succeeded/
        failed/canceled）、开始/结束时间、错误信息、输出路径。
      - 引入串行执行器（进程内 worker）：一次仅执行一个重任务，避免
        LLM/GPU 资源竞争。
      - 引入 SQLite 存储层：保存任务元数据、输入摘要、结果索引、日志
        索引。
      - 将 src/config.py / src/llm_provider.py 改为优先读环境变量
        （保留默认值兜底），去除硬编码密钥依赖。
  - 前端工作台（React）
      - 页面 1：KG 构建（文本输入/JSON 路径输入、提交任务、查看进
        度、查看结果 JSON 与基础图谱）。
      - 页面 2：QA 检索（选择 KG 文件或任务产物、输入问题、查看答案
        与上下文证据）。
      - 页面 3：日志中心（按任务/时间查看日志，支持分页与关键字过
        滤）。
      - 页面 4：系统状态（后端健康、模型配置只读、队列状态）。
  - 图谱展示
      - V1 使用基础网络图组件 + JSON 面板双视图；支持节点点击查看属
        性、关系 hover。
  - 兼容策略
      - 现有命令行入口（run.sh、src/construct/*）保持可用；Web 层仅
        做编排与状态管理，不重写核心算法。

  ### Public APIs / Interfaces

  - POST /api/v1/tasks/kg-build
      - 入参：input_type(text|json_path)、text/topic 或 json_path、
        output_dir（可选）。
      - 出参：task_id。
  - GET /api/v1/tasks/{task_id}
      - 出参：任务状态、进度、错误、产物索引（如 graph_path、
        summary_path）。
  - GET /api/v1/tasks
      - 出参：任务列表（分页、按类型/状态过滤）。
  - POST /api/v1/tasks/{task_id}/cancel
      - 出参：取消结果（仅 queued/running 可取消）。
  - POST /api/v1/qa/query
      - 入参：kg_path、question、max_hop、seed_top_k、
        max_context_items。
      - 出参：formatted_answer、retrieved_context、中间检索信息。
  - GET /api/v1/logs
      - 入参：task_id?、keyword?、page、page_size。
      - 出参：日志行与来源文件。
  - GET /api/v1/system/health
      - 出参：服务健康、队列状态、模型配置摘要（不返回密钥）。

  ### Test Plan

  - 单元测试
      - 任务状态流转、取消逻辑、SQLite CRUD、配置读取优先级（env >
        default）。
  - 集成测试
      - KG 构建任务提交到完成的全链路；QA 查询接口对已有 KG 的返回正
        确性；日志查询分页与过滤。
  - 前端 E2E
      - 提交 KG 任务 -> 轮询完成 -> 查看图谱/JSON。
      - 选择 KG 发起 QA -> 展示答案。
      - 日志页按任务筛选与关键字检索。
  - 回归验证
      - 保证 run.sh KG/qa/naive/eval 原命令行流程不受 Web 改造影响。

  ### Assumptions

  - V1 范围固定为 KG + QA + 日志；评测模块延后至 V1.1。
  - 无鉴权（内网使用场景），后续若上公网再补登录与权限体系。
  - 前后端分离部署在同一环境网络内；SQLite 作为单机持久化存储。
  - 异步任务采用串行执行；若后续吞吐不足，再升级为受控并发或外部队列
    （Redis/Celery）。