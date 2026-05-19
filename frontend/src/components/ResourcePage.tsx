import { createForm } from '@formily/core';
import { FormProvider, Field } from '@formily/react';
import { FormItem, Input } from '@formily/antd-v5';
import {
  ApiOutlined,
  CloudSyncOutlined,
  DeploymentUnitOutlined,
  FilterOutlined,
  PlusOutlined,
  ReloadOutlined,
  SafetyCertificateOutlined,
  SearchOutlined,
} from '@ant-design/icons';
import { Button, Card, Col, Drawer, Form, Input as AntInput, Row, Space, Table, Tag, Typography, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import React, { useEffect, useMemo, useState } from 'react';
import type { PageConfig, ResourceRecord } from '../models/types';
import { api, getToken } from '../services/http';
import { WorkflowDesigner } from '../workflow-designer/WorkflowDesigner';
import { AgentWizard } from '../agent-designer/AgentWizard';
import { FeatureWorkbench } from './FeatureWorkbench';

interface Props {
  page: PageConfig;
}

interface PageMeta {
  domain: string;
  description: string;
  owner: string;
  objective: string;
}

const PAGE_META: Record<string, PageMeta> = {
  '/agents': {
    domain: '智能体管理',
    description: '统一管理企业智能体资产，包括版本控制、发布管理和运行策略配置。',
    owner: '智能体研发中心',
    objective: '生产可用',
  },
  '/workflows': {
    domain: '工作流编排',
    description: '可视化编排工具、模型、知识库和人工审批节点，支撑端到端业务自动化。',
    owner: '流程自动化平台',
    objective: '稳定编排',
  },
  '/prompt-templates': {
    domain: '提示词管理',
    description: '管理提示词模板、变量、评测和版本，确保提示词资产可追踪可复用。',
    owner: '提示词工程组',
    objective: '质量一致',
  },
  '/tools': {
    domain: '工具中心',
    description: '接入内部系统和外部 API，提供统一的权限管控、调用管理和审计能力。',
    owner: '平台连接器组',
    objective: '安全调用',
  },
  '/knowledge-bases': {
    domain: '知识库管理',
    description: '统一管理企业知识源、索引构建、检索测试和上下文引用质量。',
    owner: '知识工程组',
    objective: '可信引用',
  },
  '/memories': {
    domain: '记忆管理',
    description: '管理长期记忆、会话沉淀、检索策略和上下文压缩。',
    owner: '上下文工程组',
    objective: '持续学习',
  },
  '/models': {
    domain: '模型管理',
    description: '统一管理模型供应商、模型版本、健康检查、路由和容错策略。',
    owner: '模型平台组',
    objective: '弹性供给',
  },
  '/model-routing-policies': {
    domain: '模型路由',
    description: '按成本、延迟、质量与合规约束智能调度模型，降低调用风险。',
    owner: '模型平台组',
    objective: '智能分流',
  },
  '/security/content-policies': {
    domain: '安全策略',
    description: '管理内容安全、工具权限、路径访问和高风险动作审批策略。',
    owner: '安全治理组',
    objective: '风险可控',
  },
  '/guardrails/policies': {
    domain: '安全护栏',
    description: '统一配置输入输出护栏、违规处置、数据脱敏与内容拦截策略。',
    owner: '安全治理组',
    objective: '合规运行',
  },
  '/evaluation/datasets': {
    domain: '质量评测',
    description: '维护评测数据集、运行记录和质量指标，支撑上线前验收。',
    owner: '质量评测组',
    objective: '可量化',
  },
  '/audit-logs': {
    domain: '审计日志',
    description: '追踪关键操作、敏感动作、发布变更和合规证据。',
    owner: '平台治理组',
    objective: '可追溯',
  },
  '/cost/summary': {
    domain: '成本分析',
    description: '分析模型、智能体、租户和业务线成本，支撑预算管理与优化。',
    owner: '成本治理组',
    objective: '成本透明',
  },
  '/alerts/rules': {
    domain: '告警管理',
    description: '配置运行告警规则、服务等级阈值和通知策略，保障业务连续性。',
    owner: '运行保障组',
    objective: '及时响应',
  },
  '/tenants': {
    domain: '租户管理',
    description: '管理企业租户、隔离域、资源配额和组织级策略。',
    owner: '平台管理员',
    objective: '多租户',
  },
  '/users': {
    domain: '用户与权限',
    description: '管理用户、角色、权限和组织成员访问边界。',
    owner: '平台管理员',
    objective: '最小权限',
  },
  '/system/configs': {
    domain: '系统配置',
    description: '维护平台级开关、运行参数、集成配置和基础设施能力。',
    owner: '平台管理员',
    objective: '可运维',
  },
};

function getPageMeta(page: PageConfig): PageMeta {
  return PAGE_META[page.api] || {
    domain: page.group,
    description: `集中管理${page.title}资源，统一维护状态、配置和权限。`,
    owner: '平台运营团队',
    objective: '统一管控',
  };
}

function statusColor(value?: string): string {
  if (!value) return 'default';
  if (['active', 'success', 'published', 'running', 'enabled'].includes(value)) return 'success';
  if (['failed', 'error', 'blocked', 'disabled'].includes(value)) return 'error';
  if (['pending', 'draft', 'gray', 'warning'].includes(value)) return 'warning';
  return 'processing';
}

export function ResourcePage({ page }: Props): JSX.Element {
  const [rows, setRows] = useState<ResourceRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [keyword, setKeyword] = useState('');
  const form = useMemo(() => createForm(), []);

  const load = async (pageNo = 1): Promise<void> => {
    setLoading(true);
    try {
      if (!getToken()) {
        setRows([]);
        setTotal(0);
        return;
      }
      const result = await api.list(page.api, pageNo, 20);
      const filtered = keyword ? result.items.filter((item) => `${item.name}${item.code}${item.status}`.includes(keyword)) : result.items;
      setRows(filtered);
      setTotal(result.total);
    } catch (error) {
      message.error(error instanceof Error ? error.message : '加载失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, [page.api]);

  const meta = getPageMeta(page);
  const activeRows = rows.filter((row) => ['active', 'success', 'published', 'running'].includes(row.status)).length;
  const enabledRows = rows.filter((row) => row.enabled).length;
  const guardedRows = rows.filter((row) => row.config || row.spec || row.metadata_json).length;

  const columns: ColumnsType<ResourceRecord> = [
    {
      title: '资源名称',
      dataIndex: 'name',
      sorter: (a, b) => a.name.localeCompare(b.name),
      render: (value: string, row) => (
        <Space direction="vertical" size={0}>
          <Typography.Text strong>{value || row.code || row.id}</Typography.Text>
          <Typography.Text type="secondary" className="text-xs">{row.description || ''}</Typography.Text>
        </Space>
      ),
    },
    { title: '编码', dataIndex: 'code', width: 180, render: (value: string) => <Tag>{value || '-'}</Tag> },
    { title: '状态', dataIndex: 'status', width: 120, render: (value: string) => <Tag color={statusColor(value)}>{value || 'unknown'}</Tag> },
    { title: '启用', dataIndex: 'enabled', width: 110, render: (value: boolean) => <Tag color={value ? 'success' : 'error'}>{value ? '启用' : '停用'}</Tag> },
    { title: '更新时间', dataIndex: 'updated_at', width: 190, render: (value: string) => <Typography.Text type="secondary">{value || '-'}</Typography.Text> },
    {
      title: '操作',
      width: 150,
      render: (_, row) => (
        <Space>
          <Button size="small" type="link" onClick={() => void api.action(`${page.api}/${row.id}/enable`, {}).then(() => load())}>启用</Button>
          <Button size="small" type="link" danger onClick={() => void api.action(`${page.api}/${row.id}/disable`, {}).then(() => load())}>停用</Button>
        </Space>
      )
    }
  ];

  const submit = async (): Promise<void> => {
    try {
      const values = await form.submit<Record<string, unknown>>();
      await api.create(page.api, {
        name: String(values.name || ''),
        code: String(values.code || ''),
        description: String(values.description || ''),
        model_type: String(values.model_type || ''),
        provider_type: String(values.provider_type || ''),
        config: values.configText ? JSON.parse(String(values.configText)) : {},
        spec: values.specText ? JSON.parse(String(values.specText)) : {}
      });
      setOpen(false);
      form.reset();
      await load();
    } catch (error) {
      message.error(error instanceof Error ? error.message : '保存失败');
    }
  };

  return (
    <div className="enterprise-page resource-page">
      <section className="resource-hero">
        <div>
          <Space wrap size={8}>
            <Tag color="processing">{meta.domain}</Tag>
            <Tag>{page.group}</Tag>
          </Space>
          <Typography.Title level={1}>{page.title}</Typography.Title>
          <Typography.Paragraph>{meta.description}</Typography.Paragraph>
        </div>
        <Space wrap>
          <Button icon={<ReloadOutlined />} onClick={() => void load()}>刷新</Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setOpen(true)}>新建资源</Button>
        </Space>
      </section>

      <Row gutter={[16, 16]} className="resource-kpi-row">
        {[
          { label: '资源总数', value: total || rows.length, icon: <DeploymentUnitOutlined />, extra: meta.objective },
          { label: '活跃资源', value: activeRows, icon: <CloudSyncOutlined />, extra: '可投入运行' },
          { label: '已启用', value: enabledRows, icon: <SafetyCertificateOutlined />, extra: '策略生效中' },
          { label: '配置覆盖', value: guardedRows, icon: <ApiOutlined />, extra: meta.owner },
        ].map((item) => (
          <Col xs={24} sm={12} xl={6} key={item.label}>
            <Card className="resource-kpi-card" bordered={false}>
              <span className="resource-kpi-icon">{item.icon}</span>
              <div>
                <Typography.Text type="secondary">{item.label}</Typography.Text>
                <strong>{item.value}</strong>
                <small>{item.extra}</small>
              </div>
            </Card>
          </Col>
        ))}
      </Row>

      <Card className="enterprise-card resource-toolbar-card" bordered={false}>
        <div className="resource-toolbar">
          <Space wrap>
            <AntInput.Search
              prefix={<SearchOutlined />}
              placeholder="筛选名称、编码、状态"
              value={keyword}
              onChange={(event) => setKeyword(event.target.value)}
              onSearch={() => void load()}
              allowClear
              className="resource-search"
            />
            <Button icon={<FilterOutlined />}>高级筛选</Button>
          </Space>
          <Typography.Text type="secondary">共 {rows.length} 条记录</Typography.Text>
        </div>
      </Card>

      {page.designer === 'workflow' ? <WorkflowDesigner onSaved={() => void load()} /> : null}
      {page.designer === 'agent' ? <AgentWizard onCreated={() => void load()} /> : null}
      <FeatureWorkbench page={page} rows={rows} onChanged={() => load()} />
      <Card className="enterprise-card table-card" bordered={false}>
        <Table<ResourceRecord>
          rowKey="id"
          columns={columns}
          dataSource={rows}
          loading={loading}
          pagination={{ total, pageSize: 20, onChange: (pageNo) => void load(pageNo) }}
        />
      </Card>
      <Drawer open={open} title={`新建${page.title}`} width={520} onClose={() => setOpen(false)} extra={<Button type="primary" onClick={() => void submit()}>保存</Button>}>
        <FormProvider form={form}>
          <Form layout="vertical">
            <Field name="name" title="名称" required decorator={[FormItem]} component={[Input]} />
            <Field name="code" title="编码" decorator={[FormItem]} component={[Input]} />
            <Field name="description" title="描述" decorator={[FormItem]} component={[Input.TextArea, { rows: 3 }]} />
            <Field name="model_type" title="模型类型" decorator={[FormItem]} component={[Input]} />
            <Field name="provider_type" title="供应商类型" decorator={[FormItem]} component={[Input]} />
            <Field name="configText" title="配置信息" decorator={[FormItem]} component={[Input.TextArea, { rows: 5, placeholder: '请输入 JSON 格式的配置' }]} />
            <Field name="specText" title="规格信息" decorator={[FormItem]} component={[Input.TextArea, { rows: 5, placeholder: '请输入 JSON 格式的规格' }]} />
          </Form>
        </FormProvider>
      </Drawer>
    </div>
  );
}
