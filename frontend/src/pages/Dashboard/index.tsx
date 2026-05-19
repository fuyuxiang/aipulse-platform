import {
  ApiOutlined,
  ApartmentOutlined,
  BarChartOutlined,
  CheckCircleOutlined,
  CloudServerOutlined,
  ControlOutlined,
  DeploymentUnitOutlined,
  DollarOutlined,
  FieldTimeOutlined,
  NodeIndexOutlined,
  SafetyCertificateOutlined,
  ThunderboltOutlined,
  ToolOutlined,
} from '@ant-design/icons';
import { Alert, Badge, Card, Col, Empty, Row, Space, Statistic, Table, Tag, Typography } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import React, { useEffect, useMemo, useState } from 'react';
import { api } from '../../services/http';

interface DashboardResponse {
  summary?: Record<string, number>;
}

interface HealthResponse {
  status?: string;
}

interface RuntimeStatusResponse {
  total?: number;
}

interface AssetRow {
  key: string;
  name: string;
  source: string;
  count: number;
  status: '已接入' | '待确认';
}

interface RuntimeSignal {
  name: string;
  status: 'success' | 'processing' | 'warning' | 'default' | 'error';
  value: string;
}

const SUMMARY_LABELS: Record<string, string> = {
  agents: 'Agent 资产',
  agent_runs: 'Agent 运行记录',
  workflow_runs: 'Workflow 运行记录',
  model_calls: '模型调用日志',
  tool_calls: '工具调用日志',
  rag_retrievals: 'RAG 检索日志',
  alerts: '告警事件',
  bad_cases: '质量缺陷',
};

const ASSET_SOURCES: Array<[keyof typeof SUMMARY_LABELS, string]> = [
  ['agents', 'agents'],
  ['agent_runs', 'agent_run_records'],
  ['workflow_runs', 'workflow_runs'],
  ['model_calls', 'model_call_logs'],
  ['tool_calls', 'tool_call_logs'],
  ['rag_retrievals', 'knowledge_retrieval_logs'],
  ['alerts', 'alert_events'],
  ['bad_cases', 'bad_cases'],
];

const assetColumns: ColumnsType<AssetRow> = [
  {
    title: '生产数据域',
    dataIndex: 'name',
    render: (value: string, row) => (
      <Space direction="vertical" size={0}>
        <Typography.Text strong>{value}</Typography.Text>
        <Typography.Text type="secondary" className="text-xs">来源表：{row.source}</Typography.Text>
      </Space>
    ),
  },
  { title: '记录数', dataIndex: 'count', width: 140, render: (value: number) => <Typography.Text strong>{value}</Typography.Text> },
  { title: '接入状态', dataIndex: 'status', width: 140, render: (value: string) => <Tag color={value === '已接入' ? 'success' : 'default'}>{value}</Tag> },
];

function metricValue(summary: Record<string, number>, key: string): number {
  const value = summary[key];
  return Number.isFinite(value) ? value : 0;
}

function formatSummaryLabel(key: string): string {
  return SUMMARY_LABELS[key] || key;
}

export function DashboardPage(): JSX.Element {
  const [summary, setSummary] = useState<Record<string, number>>({});
  const [healthStatus, setHealthStatus] = useState('');
  const [runtimeTotal, setRuntimeTotal] = useState<number | null>(null);
  const [dashboardConnected, setDashboardConnected] = useState(false);
  const [error, setError] = useState('');
  const tenantName = localStorage.getItem('aipulse_tenant') || '当前租户';

  useEffect(() => {
    let mounted = true;
    const load = async (): Promise<void> => {
      setError('');
      const [dashboardResult, healthResult, runtimeResult] = await Promise.allSettled([
        api.get<DashboardResponse>('/observability/dashboard'),
        api.get<HealthResponse>('/observability/health'),
        api.get<RuntimeStatusResponse>('/observability/runtime-status'),
      ]);

      if (!mounted) return;

      if (dashboardResult.status === 'fulfilled') {
        setSummary(dashboardResult.value.summary || {});
        setDashboardConnected(true);
      } else {
        setSummary({});
        setDashboardConnected(false);
        setError(dashboardResult.reason instanceof Error ? dashboardResult.reason.message : '观测数据加载失败');
      }

      if (healthResult.status === 'fulfilled') {
        setHealthStatus(healthResult.value.status || '');
      } else {
        setHealthStatus('');
      }

      if (runtimeResult.status === 'fulfilled') {
        setRuntimeTotal(Number(runtimeResult.value.total || 0));
      } else {
        setRuntimeTotal(null);
      }
    };
    void load();
    return () => {
      mounted = false;
    };
  }, []);

  const metricCards = [
    {
      key: 'agents',
      title: 'Agent 资产',
      description: '来自 agents',
      icon: <DeploymentUnitOutlined />,
      tone: 'blue',
    },
    {
      key: 'agent_runs',
      title: 'Agent 运行记录',
      description: '来自 agent_run_records',
      icon: <NodeIndexOutlined />,
      tone: 'green',
    },
    {
      key: 'workflow_runs',
      title: 'Workflow 运行记录',
      description: '来自 workflow_runs',
      icon: <BarChartOutlined />,
      tone: 'amber',
    },
    {
      key: 'alerts',
      title: '告警事件',
      description: '来自 alert_events',
      icon: <SafetyCertificateOutlined />,
      tone: 'red',
    },
  ];

  const assetRows = useMemo<AssetRow[]>(
    () =>
      ASSET_SOURCES.map(([key, source]) => ({
        key,
        name: formatSummaryLabel(key),
        source,
        count: metricValue(summary, key),
        status: dashboardConnected ? '已接入' : '待确认',
      })),
    [dashboardConnected, summary],
  );

  const runtimeSignals: RuntimeSignal[] = [
    {
      name: 'Backend API Health',
      status: healthStatus === 'healthy' ? 'success' : healthStatus ? 'warning' : 'default',
      value: healthStatus || '未返回',
    },
    {
      name: 'Runtime Instances',
      status: runtimeTotal === null ? 'default' : runtimeTotal > 0 ? 'processing' : 'warning',
      value: runtimeTotal === null ? '未返回' : `${runtimeTotal} 个`,
    },
    {
      name: 'Observability Dashboard',
      status: dashboardConnected ? 'success' : 'error',
      value: dashboardConnected ? '已连接' : '未连接',
    },
  ];

  const liveMetricItems = Object.entries(summary);
  const alertCount = metricValue(summary, 'alerts');
  const badCaseCount = metricValue(summary, 'bad_cases');

  return (
    <div className="enterprise-page dashboard-page">
      <section className="dashboard-hero">
        <div>
          <Space size={8} wrap>
            <Tag color="processing">AIPulse Command Center</Tag>
            <Tag color={dashboardConnected ? 'success' : 'error'}>{dashboardConnected ? '生产数据已连接' : '生产数据未连接'}</Tag>
          </Space>
          <Typography.Title level={1}>智能体生产运行指挥中心</Typography.Title>
        </div>
        <div className="hero-control-panel">
          <div className="hero-control-title">
            <CloudServerOutlined />
            <span>生产控制面</span>
          </div>
          <div className="hero-control-grid">
            <span>租户</span>
            <strong>{tenantName}</strong>
            <span>权限</span>
            <strong>observability:read</strong>
            <span>数据源</span>
            <strong>/api/v1</strong>
          </div>
        </div>
      </section>

      {error ? <Alert className="mb-4" type="error" showIcon message="生产观测数据未加载" description={error} /> : null}

      <Row gutter={[16, 16]} className="metric-row">
        {metricCards.map((item) => (
          <Col xs={24} sm={12} xl={6} key={item.key}>
            <Card className={`metric-card metric-card-${item.tone}`} bordered={false}>
              <div className="metric-card-top">
                <span className="metric-icon">{item.icon}</span>
                <Tag bordered={false}>{item.description}</Tag>
              </div>
              <Statistic title={item.title} value={metricValue(summary, item.key)} />
            </Card>
          </Col>
        ))}
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={16}>
          <Card
            className="enterprise-card"
            title={
              <Space>
                <ControlOutlined />
                <span>生产能力闭环</span>
              </Space>
            }
          >
            <div className="lifecycle-strip">
              {[
                { title: '开发态', text: 'Agent / Workflow / Prompt', icon: <ApartmentOutlined /> },
                { title: '发布态', text: '版本、灰度、市场分发', icon: <ThunderboltOutlined /> },
                { title: '运行态', text: '会话、调度、多智能体', icon: <ApiOutlined /> },
                { title: '治理态', text: '护栏、审计、成本、权限', icon: <SafetyCertificateOutlined /> },
              ].map((item, index) => (
                <div className="lifecycle-step" key={item.title}>
                  <span className="lifecycle-index">{index + 1}</span>
                  <span className="lifecycle-icon">{item.icon}</span>
                  <strong>{item.title}</strong>
                  <small>{item.text}</small>
                </div>
              ))}
            </div>
            <Table<AssetRow>
              className="mt-4"
              rowKey="key"
              size="middle"
              columns={assetColumns}
              dataSource={assetRows}
              pagination={false}
            />
          </Card>
        </Col>

        <Col xs={24} xl={8}>
          <Card
            className="enterprise-card runtime-card"
            title={
              <Space>
                <FieldTimeOutlined />
                <span>运行连接状态</span>
              </Space>
            }
            extra={<Badge status={dashboardConnected ? 'success' : 'error'} text={dashboardConnected ? 'Connected' : 'Disconnected'} />}
          >
            <Space direction="vertical" size={14} className="w-full">
              {runtimeSignals.map((signal) => (
                <div className="runtime-signal" key={signal.name}>
                  <Badge status={signal.status} text={signal.name} />
                  <Typography.Text strong>{signal.value}</Typography.Text>
                </div>
              ))}
            </Space>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} className="mt-4">
        <Col xs={24} xl={15}>
          <Card
            className="enterprise-card"
            title={
              <Space>
                <CheckCircleOutlined />
                <span>生产风险入口</span>
              </Space>
            }
          >
            <Row gutter={[12, 12]}>
              <Col xs={24} md={12}>
                <div className="production-risk-card">
                  <SafetyCertificateOutlined />
                  <Typography.Text type="secondary">告警事件</Typography.Text>
                  <strong>{alertCount}</strong>
                  <small>来自 alert_events</small>
                </div>
              </Col>
              <Col xs={24} md={12}>
                <div className="production-risk-card">
                  <ToolOutlined />
                  <Typography.Text type="secondary">质量缺陷</Typography.Text>
                  <strong>{badCaseCount}</strong>
                  <small>来自 bad_cases</small>
                </div>
              </Col>
            </Row>
            {alertCount === 0 && badCaseCount === 0 ? (
              <Empty className="mt-4" image={Empty.PRESENTED_IMAGE_SIMPLE} description="当前生产观测接口未返回告警或质量缺陷" />
            ) : null}
          </Card>
        </Col>
        <Col xs={24} xl={9}>
          <Card
            className="enterprise-card live-summary-card"
            title={
              <Space>
                <DollarOutlined />
                <span>实时指标透视</span>
              </Space>
            }
          >
            {liveMetricItems.length > 0 ? (
              <div className="live-summary-grid">
                {liveMetricItems.map(([key, value]) => (
                  <div key={key}>
                    <span>{formatSummaryLabel(key)}</span>
                    <strong>{value}</strong>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-live-summary">
                <Typography.Text strong>暂无生产观测数据</Typography.Text>
                <Typography.Text type="secondary">请确认当前账号拥有 observability:read 权限，并且后端已经写入运行数据。</Typography.Text>
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
}
