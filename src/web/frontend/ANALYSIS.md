# RAKG Frontend 分析文档

## 1. 分析范围
- 目录：`src/web/frontend`
- 技术栈：React 18 + TypeScript + Vite 5
- 关注点：页面架构、模块职责、状态与数据流、可维护性与迭代风险

## 2. 项目结构与职责

```
src/web/frontend
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig*.json
└── src
    ├── main.tsx                # 应用入口
    ├── App.tsx                 # 顶层标签页壳层
    ├── api.ts                  # 后端 API 封装
    ├── types.ts                # 共享数据类型
    ├── styles.css              # 全局样式
    ├── components
    │   └── GraphView.tsx       # 图谱可视化组件
    └── pages
        ├── KGBuildPage.tsx     # KG 构建任务提交/轮询/图谱预览
        ├── QAPage.tsx          # QA 查询
        ├── LogsPage.tsx        # 日志检索与分页
        └── SystemPage.tsx      # 系统状态与任务概览
```

整体是一个单页工作台（Workbench）应用，采用“顶层标签页 + 页面组件 + 公共 API 层”的轻量结构，没有引入路由和外部状态管理库。

## 3. 构建与运行配置

### 3.1 依赖情况
- 运行时依赖非常少：`react`、`react-dom`
- 开发依赖：`typescript`、`vite`、`@vitejs/plugin-react`、React 类型包

结论：  
依赖面小，启动快，维护成本低，适合内部工具型前端。

### 3.2 TS 配置
- `strict: true`，类型约束较严格
- `moduleResolution: bundler`，与 Vite 配套
- `noEmit: true`，仅作类型检查，由 Vite 打包

结论：  
类型安全基础较好，但 `types.ts` 中仍有较多 `Record<string, unknown>`，会限制类型系统的收益。

## 4. 页面架构与交互流

## 4.1 顶层壳层（`App.tsx`）
- 用 `activeTab` 本地状态切换四个页面：KG 构建、QA、日志、系统
- 条件渲染策略简单直接，避免了路由复杂度

优点：
- 结构直观，开发成本低
- 无路由跳转开销

限制：
- 页面状态不会持久化到 URL（刷新丢失上下文）
- 不利于分享具体页面链接

## 4.2 KG 构建页（`pages/KGBuildPage.tsx`）
- 提交任务：`createKGBuildTask`
- 轮询任务详情：`getTask`（成功后自动读图谱）
- 任务取消：`cancelTask`
- 图谱读取：`readGraphArtifact`
- 图谱展示：`GraphView`

关键机制：
- 通过 `taskId` 触发 `useEffect` 轮询
- 非终态每 1.8s 继续轮询，异常时 2.4s 重试
- 终态集合：`succeeded / failed / canceled`

优点：
- 任务型流程完整（提交-跟踪-取消-结果展示）
- 有异常兜底与重试

风险：
- 轮询逻辑分散且页面内耦合，后续页面增多时复用性不足
- 图谱 JSON 直接 `pre` 渲染，大数据量时可能卡顿

## 4.3 QA 页（`pages/QAPage.tsx`）
- 初始化拉取已完成 KG 任务：`listTasks(status=succeeded, task_type=kg_build)`
- 可从任务中自动回填 `kg_path`（通过 `getTask`）
- 提交查询：`qaQuery`

优点：
- 与 KG 构建页形成业务闭环（构建后直接检索）
- 参数暴露完整，便于调参

风险：
- 对 `kgPath` 缺少前端校验（空值、路径格式）
- `result` 结构展示主要依赖 JSON，缺少信息分层

## 4.4 日志页（`pages/LogsPage.tsx`）
- 查询参数：`task_id`、`keyword`、`page_size`
- 分页：上一页/下一页 + 总页数展示
- 数据接口：`getLogs`

优点：
- 功能闭环完整，易用性尚可

风险：
- `useEffect` 仅依赖 `page/pageSize`，筛选条件变更后需手动点击“查询”
- `load` 函数每次重建，后续逻辑扩展时可能引入闭包状态问题

## 4.5 系统页（`pages/SystemPage.tsx`）
- 并发请求健康信息与最近任务：`Promise.all([getHealth, listTasks])`
- 5 秒自动轮询 + 手动刷新

优点：
- 系统观测入口集中，能快速判断队列与任务状态

风险：
- 固定轮询频率对后端压力不可调
- 页面销毁后虽清理定时器，但缺少请求级取消机制（AbortController）

## 5. API 层分析（`api.ts`）

统一 `request<T>` 做了这些事：
- 拼接 `API_BASE`（默认 `http://127.0.0.1:8000/api/v1`）
- 设置 `Content-Type: application/json`
- 非 2xx 时抛错（`response.text()`）
- 2xx 时解析 JSON 返回

优点：
- 抽象统一，调用方式一致
- 错误处理模式统一

可改进点：
- 缺少超时控制和取消信号
- 缺少鉴权头注入扩展位（如 token）
- 错误对象结构未标准化（前端 UI 只能显示纯文本）

## 6. 图谱渲染组件分析（`components/GraphView.tsx`）

### 6.1 数据标准化
- 支持两种 relation 结构：
  - 数组式：`[source, relation, target, description?]`
  - 对象式：`{ source, relation, target, ... }`
- 实体缺失时自动补 `Unknown` 节点，保证边可渲染

### 6.2 布局与交互
- 固定圆环布局（非力导向）
- SVG 渲染边和节点，点击节点显示右侧详情
- 节点名超长截断展示

优点：
- 零外部图形依赖，简单稳定
- 对“脏数据”容错较好（关系存在但实体缺失）

风险：
- 节点多时可读性会快速下降（边重叠严重）
- 当前布局不可拖拽、不可缩放
- `selectedNode` 仅本地状态，无 URL/外部同步

## 7. 样式与 UI 体系（`styles.css`）
- CSS 变量定义主题色与背景层次
- 核心布局：`page-grid`、`panel`、`graph-layout`
- 响应式断点：`1024px` 与 `760px`

评价：
- 视觉风格统一，信息密度适中
- 表单、卡片、日志、JSON 区块均有一致样式规范
- 对移动端有基本适配

改进点：
- 组件样式全部全局化，缺少作用域隔离
- 缺少深色主题或高对比模式

## 8. 工程质量评估

### 8.1 当前优势
- 结构清晰，代码量适中，学习成本低
- 业务流程完整，覆盖 KG 构建/QA/日志/系统监控主路径
- TypeScript 严格模式启用，减少低级错误
- 依赖少，升级与部署成本低

### 8.2 主要风险（按优先级）
1. 轮询/请求控制未抽象：多个页面各自处理异步与重试，易重复实现。  
2. 错误处理粒度粗：主要是字符串展示，难以做分类告警和引导。  
3. 类型域模型偏弱：大量 `Record<string, unknown>`，业务字段语义不够明确。  
4. 大图谱性能风险：JSON 与 SVG 在大数据场景可能出现渲染瓶颈。  
5. 缺少自动化测试：页面交互、API 协议和关键流程没有回归保护。  

## 9. 建议的迭代路线

### 第一阶段（低成本高收益）
- 抽离通用请求 Hook（加载态、错误态、取消、重试策略）
- 将轮询逻辑封装为 `usePollingTask(taskId, options)`
- 为关键输入增加前端校验（KG 路径、QA 参数范围）
- 统一错误模型（code/message/detail）

### 第二阶段（可维护性增强）
- 增加 React Router（保留 tab 样式但支持 URL）
- 细化类型定义，减少 `unknown` 泛型对象
- 将 `GraphView` 增加缩放/拖拽或分层过滤能力

### 第三阶段（质量保障）
- 引入单元测试和页面级集成测试
- 对大数据量图谱进行性能压测与懒渲染策略

## 10. 结论
该前端实现符合“内部工作台”定位：轻量、直接、可用。  
如果后续面向更大数据规模或更多角色协作，优先应补齐“请求/轮询抽象 + 类型建模 + 测试覆盖”三项基础能力，以降低迭代风险并提升长期可维护性。
