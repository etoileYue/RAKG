# LightRAG 前后端架构与知识图谱落地说明

## 1. 文档目标与范围

本文档面向工程实现，说明 LightRAG 项目的前后端结构与协作方式，重点解释 `lightrag_webui` 中可交互知识图谱（Knowledge Graph）的落地方案。  
本文不展开业务语义或算法细节，重点是代码层面的组织、数据流和交互实现。

## 2. 项目整体架构

LightRAG 采用单仓库多模块结构：

- 后端：`lightrag/api`（FastAPI 服务入口与路由）
- 前端：`lightrag_webui`（React + TypeScript + Vite）
- Core：`lightrag/`（RAG 核心、存储抽象、LLM/Embedding/Rerank 绑定）

在部署上，前端构建产物会输出到后端目录 `lightrag/api/webui`，由 FastAPI 统一挂载与服务。

## 3. 后端架构（FastAPI）

### 3.1 启动与应用组装

后端入口为 `lightrag/api/lightrag_server.py`，核心职责：

- 解析配置与环境变量（结合 `lightrag/api/config.py`）
- 初始化 `LightRAG` 实例
- 注册各业务路由
- 配置 CORS、鉴权依赖、异常处理
- 挂载 `/webui` 静态资源和 `/docs` 文档资源

### 3.2 路由分层

后端路由按领域拆分：

- 文档管理：`lightrag/api/routers/document_routes.py`（`/documents/*`）
- 查询能力：`lightrag/api/routers/query_routes.py`（`/query`, `/query/stream`, `/query/data`）
- 图谱能力：`lightrag/api/routers/graph_routes.py`（`/graphs`, `/graph/*`）
- Ollama 兼容层：`lightrag/api/routers/ollama_api.py`（挂载到 `/api/*`）

### 3.3 鉴权与会话机制

鉴权逻辑在 `lightrag/api/utils_api.py` 的 `get_combined_auth_dependency`：

- 支持 Bearer Token 与 `X-API-Key` 组合策略
- 支持白名单路径
- 支持 token 自动续期（返回 `X-New-Token`）
- 对高频轮询路径（如健康检查、分页、流水线状态）跳过续期，降低开销

### 3.4 图谱后端存储结构

图谱存储走 `BaseGraphStorage` 抽象，支持多实现：

- `NetworkXStorage`（默认）
- `Neo4JStorage`
- `PGGraphStorage`
- `MongoGraphStorage`
- `MemgraphStorage`
- `OpenSearchGraphStorage`

默认 `NetworkXStorage` 会落盘到 GraphML 文件（`graph_<namespace>.graphml`）。

API 输出统一为 `KnowledgeGraph` 结构：

- `nodes: [{ id, labels, properties }]`
- `edges: [{ id, source, target, type, properties }]`
- `is_truncated: bool`

## 4. 前端架构（`lightrag_webui`）

### 4.1 技术栈

- React 19 + TypeScript
- Vite 7（构建）
- Tailwind CSS 4 + Radix UI
- Zustand（全局状态）
- Sigma.js + graphology（图渲染与图模型）
- Axios + Fetch（HTTP/流式请求）
- Bun（测试与推荐安装路径）

### 4.2 模块组织

核心目录：

- `src/features`: 页面级模块（Documents/Knowledge Graph/Retrieval/API）
- `src/components`: 复用组件与领域组件
- `src/stores`: 全局状态（settings/backend/auth/graph）
- `src/api`: API 适配层
- `src/hooks`: 复用逻辑（图谱数据与行为）

### 4.3 路由与主框架

- 入口：`src/main.tsx`
- 路由：`src/AppRouter.tsx`（`HashRouter`）
- 主应用：`src/App.tsx`（Tab 切换四个主页面）

页面 Tab：

- `documents`
- `knowledge-graph`
- `retrieval`
- `api`

### 4.4 状态管理分工

- `stores/settings.ts`：用户偏好与运行参数（主题、语言、当前 tab、图谱深度、图谱最大节点数等）
- `stores/state.ts`：后端健康状态与认证状态
- `stores/graph.ts`：图谱交互状态、图数据结构与编辑更新能力

## 5. 前后端协作关系

`src/api/lightrag.ts` 是统一 API 入口，封装了：

- `axiosInstance` + 请求/响应拦截器
- Bearer token 注入与自动续期接收
- guest token 失效后的静默刷新
- 流式查询 `queryTextStream`（`/query/stream`，NDJSON 解码）

主要页面与 API 映射：

- 文档页：`/documents/*`
- 图谱页：`/graphs` + `/graph/*`
- 检索页：`/query`、`/query/stream`
- 登录页：`/auth-status`、`/login`
- API 页：iframe 打开 `/docs`

## 6. 重点：前端知识图谱的落地实现

本节回答“可交互知识图谱如何真正跑起来”。

### 6.1 总体落地思路

前端图谱实现采用“双图模型 + 事件驱动 UI”：

- 业务数据模型：`RawGraph`
- 渲染数据模型：`sigmaGraph`（graphology `UndirectedGraph`）
- 交互状态：Zustand（选中、聚焦、布局、加载、版本号、节点操作触发器等）

这种设计让“渲染性能”与“业务可编辑性”解耦。

### 6.2 图数据结构（前端）

定义在 `src/stores/graph.ts`：

1. `RawGraph`

- `nodes: RawNodeType[]`
- `edges: RawEdgeType[]`
- `nodeIdMap: Record<string, number>`
- `edgeIdMap: Record<string, number>`
- `edgeDynamicIdMap: Record<string, number>`

`RawGraph` 负责可编辑语义数据与 O(1) 查找。

2. `sigmaGraph`

- 使用 graphology `UndirectedGraph`
- 节点属性包含 `x/y/size/color/label/border...`
- 边属性包含 `size/originalWeight/label/type...`

`sigmaGraph` 负责渲染、动画、相机与事件处理。

### 6.3 数据加载链路

核心在 `src/hooks/useLightragGraph.tsx`。

链路如下：

1. 用户通过 `GraphLabels` 选择标签，更新 `queryLabel`
2. `useLightragGraph` 触发 `queryGraphs(label, max_depth, max_nodes)`
3. 返回数据转为 `RawGraph`，补充：
   - 随机初始坐标
   - 节点度数与节点尺寸
   - 节点颜色（按 `entity_type`）
4. 由 `createSigmaGraph` 构建 `sigmaGraph`
5. 更新 `graph store`，`GraphViewer` 自动渲染

边界处理：

- 空图：创建占位节点（`Graph Is Empty`）
- 截断图：根据 `is_truncated` 提示用户
- 并发保护：`isFetching`、`graphDataFetchAttempted`、`fetchInProgressRef` 防重复拉取

### 6.4 渲染与交互引擎

容器在 `src/features/GraphViewer.tsx`：

- `SigmaContainer` 承载 WebGL 渲染
- `GraphControl` 负责事件注册和样式 reducer
- 控制组件（缩放、布局、全屏、设置、图例）挂在侧边工具栏

`GraphControl` 是交互核心：

- 注册 `clickNode / clickEdge / enterNode / leaveNode / clickStage`
- 根据 `selected/focused` 动态计算节点与边显示状态
- 支持未选中节点置灰、边高亮/隐藏、标签显示策略
- 随主题变化动态调整颜色

### 6.5 图内搜索与聚焦

图内搜索在 `src/components/graph/GraphSearch.tsx`：

- 基于 MiniSearch 建本地索引（按节点 label）
- 支持前缀、模糊、中间匹配
- 搜索命中后写入 graph store 的选中/聚焦状态

聚焦在 `src/components/graph/FocusOnNode.tsx`：

- `gotoNode(node)` 实现镜头跳转
- 同步高亮节点
- 无选中时重置视角

### 6.6 节点扩展与裁剪（强交互能力）

节点扩展/裁剪逻辑在 `useLightragGraph.tsx`：

1. 扩展（Expand）

- 触发源：属性面板“扩展节点”按钮
- 动作：以当前节点 label 调 `/graphs` 拉取深度 2 的扩展子图
- 合并策略：
  - 仅并入与当前图可连接的新节点/边
  - 保留已有节点坐标
  - 为新增节点计算极坐标分布位置
  - 重算节点尺寸和边粗细
  - 更新 `RawGraph` 与 `sigmaGraph` 双模型

2. 裁剪（Prune）

- 触发源：属性面板“裁剪节点”按钮
- 动作：删除目标节点，同时删除会因此孤立的节点
- 保护：禁止一次删除所有节点（防止用户误操作清空视图）

### 6.7 属性面板与图编辑回写

属性面板在 `src/components/graph/PropertiesView.tsx`，编辑入口在 `EditablePropertyRow.tsx`。

编辑行为：

- 节点属性更新：`/graph/entity/edit`
- 边属性更新：`/graph/relation/edit`
- 名称冲突检测：`/graph/entity/exists`

更新策略：

- 后端成功后，调用 graph store 的 `updateNodeAndSelect` / `updateEdgeAndSelect`
- 同步修改本地 `RawGraph` 与 `sigmaGraph`，尽量避免整图重载

实体改名与合并：

- `entity_id` 改名可触发 merge（可选）
- 合并成功后展示 `MergeDialog`，支持以合并后的实体为新起点刷新图
- 同步更新搜索历史与标签下拉状态

### 6.8 图谱设置与运行时参数

`src/components/graph/Settings.tsx` 提供运行时控制：

- 最大深度 `graphQueryMaxDepth`
- 最大节点数 `graphMaxNodes`（受后端 `max_graph_nodes` 上限约束）
- 边粗细范围、节点标签显示、边标签显示、是否隐藏未选中边等

这些参数通过 `settings store` 持久化，并驱动图谱刷新或样式变更。

### 6.9 类型-颜色映射

`src/utils/graphColor.ts` 做了实体类型归一与颜色映射：

- 同义词归并（含中英文）
- 预定义类型固定色
- 未知类型从扩展色池分配
- 颜色映射写回 `typeColorMap`，图例组件 `Legend` 自动呈现

## 7. 当前实现的工程特点与注意点

### 7.1 优势

- 双模型设计（业务/渲染）降低了编辑与渲染耦合
- 图谱交互能力完整（搜索、聚焦、布局、扩展、裁剪、编辑、合并）
- 状态集中在 store，组件职责清晰

### 7.2 注意点

- `useLightragGraph` 当前通过 `PropertiesView` 引入，图数据加载与属性面板挂载存在耦合；若后续调整属性面板显示策略，需要评估对初始加载的影响。
- `useLightragGraph` 与 `DocumentManager` 体量较大，后续可按“数据加载/图编辑/节点操作”继续拆分，降低维护复杂度。

## 8. 关键文件索引

后端：

- `lightrag/api/lightrag_server.py`
- `lightrag/api/config.py`
- `lightrag/api/utils_api.py`
- `lightrag/api/routers/document_routes.py`
- `lightrag/api/routers/query_routes.py`
- `lightrag/api/routers/graph_routes.py`
- `lightrag/api/routers/ollama_api.py`
- `lightrag/kg/networkx_impl.py`
- `lightrag/types.py`

前端：

- `lightrag_webui/src/App.tsx`
- `lightrag_webui/src/AppRouter.tsx`
- `lightrag_webui/src/api/lightrag.ts`
- `lightrag_webui/src/stores/settings.ts`
- `lightrag_webui/src/stores/state.ts`
- `lightrag_webui/src/stores/graph.ts`
- `lightrag_webui/src/features/GraphViewer.tsx`
- `lightrag_webui/src/hooks/useLightragGraph.tsx`
- `lightrag_webui/src/components/graph/GraphControl.tsx`
- `lightrag_webui/src/components/graph/GraphLabels.tsx`
- `lightrag_webui/src/components/graph/GraphSearch.tsx`
- `lightrag_webui/src/components/graph/PropertiesView.tsx`
- `lightrag_webui/src/components/graph/EditablePropertyRow.tsx`
- `lightrag_webui/src/utils/graphColor.ts`
- `lightrag_webui/src/utils/SearchHistoryManager.ts`
