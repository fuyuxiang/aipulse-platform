import type { PageConfig } from '../models/types';

const pages: Array<[string, string, string, string, PageConfig['designer']?]> = [
  // 首页
  ['/dashboard', '运行大盘', '/observability/dashboard', '首页'],

  // 开发 — 智能体
  ['/agents', '智能体列表', '/agents', '开发'],
  ['/agents/create', '智能体创建向导', '/agents', '开发', 'agent'],

  // 开发 — 工作流
  ['/workflows', '工作流列表', '/workflows', '开发'],
  ['/workflows/designer', '工作流编排', '/workflows', '开发', 'workflow'],

  // 开发 — 提示词
  ['/prompt-studio', '提示词工作台', '/prompt-templates', '开发'],

  // 开发 — 工具
  ['/tools', '工具中心', '/tools', '开发'],

  // 开发 — 知识库
  ['/knowledge', '知识库', '/knowledge-bases', '开发'],

  // 开发 — 记忆
  ['/memory', '记忆管理', '/memories', '开发'],

  // 运行 — 对话
  ['/chat', '对话', '/chat/sessions', '运行'],

  // 运行 — 多智能体
  ['/multi-agent', '多智能体协同', '/agent-teams', '运行'],

  // 运行 — 调度
  ['/scheduler', '调度中心', '/scheduler/jobs', '运行'],

  // 运行 — 市场
  ['/marketplace', '应用市场', '/marketplace/listings', '运行'],

  // 治理 — 模型
  ['/models', '模型管理', '/models', '治理'],
  ['/model-routing', '模型路由', '/model-routing-policies', '治理'],

  // 治理 — 安全与护栏
  ['/security', '安全策略', '/security/content-policies', '治理'],
  ['/guardrails', '安全护栏', '/guardrails/policies', '治理'],

  // 治理 — 评测
  ['/evaluation', '质量评测', '/evaluation/datasets', '治理'],

  // 治理 — 审计
  ['/audit', '审计日志', '/audit-logs', '治理'],

  // 治理 — 成本
  ['/cost-analytics', '成本分析', '/cost/summary', '治理'],

  // 治理 — 监控
  ['/trace', '链路追踪', '/observability/traces', '治理'],
  ['/alerts', '告警管理', '/alerts/rules', '治理'],

  // 设置 — 租户与权限
  ['/tenants', '租户管理', '/tenants', '设置'],
  ['/users', '用户与角色', '/users', '设置'],

  // 设置 — 系统
  ['/system', '系统配置', '/system/configs', '设置'],
];

export const pageConfigs: PageConfig[] = pages.map(([path, title, api, group, designer]) => ({ path, title, api, group, designer }));
