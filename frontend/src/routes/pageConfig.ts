import type { PageConfig } from '../models/types';

const pages: Array<[string, string, string, string, PageConfig['designer']?]> = [
  // ────────────── Home ──────────────
  ['/home', '工作空间', '/observability/dashboard', '首页'],

  // ────────────── Build ──────────────
  ['/build/agents', '智能体', '/agents', '构建'],
  ['/build/agents/create', '智能体创建向导', '/agents', '构建', 'agent'],
  ['/build/workflows', '工作流', '/workflows', '构建'],
  ['/build/workflows/designer', '工作流编排', '/workflows', '构建', 'workflow'],
  ['/build/tools', '工具', '/tools', '构建'],
  ['/build/knowledge', '知识库', '/knowledge-bases', '构建'],
  ['/build/prompts', '提示词', '/prompt-templates', '构建'],

  // ────────────── Playground ──────────────
  ['/playground', '调试', '/chat/sessions', '调试'],

  // ────────────── Eval ──────────────
  ['/eval', '评测', '/evaluation/datasets', '评测'],

  // ────────────── Observe ──────────────
  ['/observe/trace', '链路追踪', '/observability/traces', '观测'],
  ['/observe/cost', '成本分析', '/cost/summary', '观测'],
  ['/observe/alerts', '告警', '/alerts/rules', '观测'],
  ['/observe/audit', '审计日志', '/audit-logs', '观测'],

  // ────────────── Deploy ──────────────
  ['/deploy/channels', '发布渠道', '/marketplace/listings', '发布'],
  ['/deploy/api-keys', 'API 凭证', '/system/configs', '发布'],
  ['/deploy/versions', '版本发布', '/agents', '发布'],

  // ────────────── Settings ──────────────
  ['/settings/models', '模型', '/models', '设置'],
  ['/settings/model-routing', '模型路由', '/model-routing-policies', '设置'],
  ['/settings/guardrails', '安全护栏', '/guardrails/policies', '设置'],
  ['/settings/security', '安全策略', '/security/content-policies', '设置'],
  ['/settings/scheduler', '调度', '/scheduler/jobs', '设置'],
  ['/settings/tenants', '租户', '/tenants', '设置'],
  ['/settings/organizations', '组织', '/orgs', '设置'],
  ['/settings/users', '用户', '/users', '设置'],
  ['/settings/roles', '角色', '/roles', '设置'],
  ['/settings/system', '系统', '/system/configs', '设置'],
];

export const pageConfigs: PageConfig[] = pages.map(([path, title, api, group, designer]) => ({ path, title, api, group, designer }));

// 旧路径 → 新路径重定向（阶段1兼容）
export const legacyRedirects: Record<string, string> = {
  '/dashboard': '/home',
  '/agents': '/build/agents',
  '/agents/create': '/build/agents/create',
  '/workflows': '/build/workflows',
  '/workflows/designer': '/build/workflows/designer',
  '/tools': '/build/tools',
  '/knowledge': '/build/knowledge',
  '/memory': '/build/agents',
  '/prompt-studio': '/build/prompts',
  '/chat': '/playground',
  '/multi-agent': '/playground?mode=multi',
  '/evaluation': '/eval',
  '/trace': '/observe/trace',
  '/cost-analytics': '/observe/cost',
  '/alerts': '/observe/alerts',
  '/audit': '/observe/audit',
  '/marketplace': '/deploy/channels',
  '/scheduler': '/settings/scheduler',
  '/models': '/settings/models',
  '/model-routing': '/settings/model-routing',
  '/guardrails': '/settings/guardrails',
  '/security': '/settings/security',
  '/tenants': '/settings/tenants',
  '/users': '/settings/users',
  '/roles': '/settings/roles',
  '/organizations': '/settings/organizations',
  '/system': '/settings/system',
};
