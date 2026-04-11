# RAKG Frontend Analysis

## 1. 分析范围
- 目录：`src/web/frontend`
- 时间：2026-04-11
- 目标：梳理当前前端实现、模块职责、数据流、可维护性与风险

## 2. 技术栈与工程配置
- 框架：React 18 + TypeScript 5 + Vite 5
- 样式：Tailwind CSS + 少量全局组件类（`styles.css`）
- 图谱可视化：`graphology` + `sigma`
- 构建脚本：`dev` / `build` / `preview`（无 `test`、无 `lint`）
- API 代理：Vite dev server 将 `/api`、`/docs`、`/openapi.json`、`/redoc` 转发到 `VITE_PROXY_TARGET`（默认 `http://127.0.0.1:8000`）

构建验证结果（本地）：
- `npm run build` 成功
- 产物体积约：`index-DYptNpal.js 240.85 kB (gzip 68.50 kB)`，`sigma.esm-*.js 91.30 kB (gzip 24.71 kB)`

## 3. 当前前端架构

### 3.1 顶层结构
- 入口：`src/main.tsx`
- 应用壳：`src/App.tsx`
  - 使用本地 `activeTab` 切换 4 个主页面：
    - `DocumentsPage`
    - `KnowledgeGraphPage`
    - `RetrievalPage`
    - `ApiDocsPage`
  - 使用 URL query `tab` 做轻量同步（可分享当前 tab）
  - 右侧抽屉：`SystemDrawer`（日志 + 系统状态）

### 3.2 数据访问层
- `src/api.ts` 提供统一 `request<T>()`
  - 基础路径：`VITE_API_BASE`，默认 `/api/v1`
  - 统一 JSON headers
  - 非 2xx 时解析文本/`detail` 并抛错
- `src/types.ts` 定义任务、日志、QA、图谱等数据结构

### 3.3 页面职责
- `DocumentsPage.tsx`：提交 KG 构建任务、任务筛选列表、任务详情与取消、已有 KG 候选加载
- `KnowledgeGraphPage.tsx`：KG 候选选择、图谱加载、交互渲染（`SigmaGraphPanel`）
- `RetrievalPage.tsx`：基于 KG 路径发起 QA 检索，展示格式化答案与上下文
- `ApiDocsPage.tsx`：iframe 内嵌 `/docs`
- `SystemDrawer.tsx`：日志分页检索 + 健康状态轮询（5 秒）

### 3.4 图谱渲染链路
- `readGraphArtifact(path)` 取回原始图
- `src/lib/graph.ts` 归一化节点/边（兼容数组式与对象式 relation）
- `SigmaGraphPanel.tsx` 负责：
  - 动态加载 Sigma
  - 节点上色、度数映射尺寸
  - 搜索过滤、邻居聚焦、点击节点详情、相机动画

## 4. 代码现状评估

### 4.1 优点
- 模块边界清晰：`pages`、`components`、`api`、`types` 分工明确
- 业务闭环完整：构建 -> 浏览图谱 -> QA -> 系统与日志观测
- TypeScript `strict: true`，基础类型安全较好
- 前端可直接联调后端 docs/openapi，开发效率高
- Sigma 交互能力较完整，适配“内部工作台”场景

### 4.2 主要风险与技术债
1. 存在“旧版页面/组件”未接入当前入口
- `KGBuildPage.tsx`、`QAPage.tsx`、`SystemPage.tsx`、`LogsPage.tsx` 与 `GraphView.tsx` 在当前 `App.tsx` 中未被引用。
- 这些文件仍保留大量旧样式类名（如 `page-grid`、`json-view`），当前 `styles.css` 不再定义对应规则。
- 风险：新成员误用旧文件，造成维护分叉。

2. 请求生命周期控制较弱
- 多处异步请求/轮询未使用 `AbortController`。
- 组件快速切换时存在“过期请求回写状态”的可能（虽有部分定时器清理）。

3. 轮询策略分散
- `DocumentsPage`、`SystemDrawer` 各自管理轮询节奏与终止条件，缺少共享 hook。
- 后续页面增长时会重复实现，调参成本高。

4. 类型语义仍偏宽
- 图谱、QA 上下文等字段仍大量使用 `Record<string, unknown>`。
- 对复杂 payload 的 IDE 约束与重构收益有限。

5. 工程质量门禁缺失
- 未配置 `lint`、`test` 脚本。
- 回归主要依赖手工联调，长期迭代风险较高。

## 5. 建议的改进优先级

### P0（建议先做）
1. 处理未接入旧代码：删除或迁移到 `legacy/` 并在 README 说明，避免双轨实现。
2. 为关键请求引入取消机制：统一支持 `AbortController`，避免过期回写。
3. 抽象轮询 hook：如 `useTaskPolling` / `useIntervalWhenVisible`，统一重试与停止策略。

### P1
1. 增加工程门禁：`eslint` + 基础单测（API 层与核心交互）。
2. 收紧领域类型：优先细化 `TaskDetail.output_payload`、`QAResponse.intermediate/retrieved_context`。
3. 将公共查询参数与错误处理模型统一（错误码、可读消息、原始 detail）。

### P2
1. 评估引入数据请求库（如 TanStack Query）以统一缓存、重试、取消、轮询。
2. 图谱大数据场景优化：分层加载、节点上限控制、细节面板惰性渲染。

## 6. 结论
- 当前前端已经具备较完整的 RAKG 工作台能力，结构清晰、可用性良好。
- 主要问题不是“功能缺失”，而是“历史代码分叉 + 异步治理不足 + 工程门禁薄弱”。
- 若优先完成 P0/P1，整体可维护性和迭代稳定性会明显提升。
