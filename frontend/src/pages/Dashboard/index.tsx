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
  status: '正常' | '未接入';
}

interface RuntimeSignal {
  name: string;
  status: 'success' | 'processing' | 'warning' | 'default' | 'error';
  value: string;
}

const SUMMARY_LABELS: Record<string, string> = {
  agents: '智能体',
  agent_runs: '运行记录',
  workflow_runs: '工作流执行',
  model_calls: '模型调用',
  tool_calls: '工具调用',
  rag_retrievals: '知识检索',
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
    title: '数据类别',
    dataIndex: 'name',
    render: (value: string) => (
      <Typography.Text strong>{value}</Typography.Text>
    ),
  },
  { title: '记录数', dataIndex: 'count', width: 140, render: (value: number) => <Typography.Text strong>{value}</Typography.Text> },
  { title: '状态', dataIndex: 'status', width: 140, render: (value: string) => <Tag color={value === '正常' ? 'success' : 'default'}>{value}</Tag> },
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
      title: '智能体',
      description: '已注册智能体总数',
      icon: <DeploymentUnitOutlined />,
      tone: 'blue',
    },
    {
      key: 'agent_runs',
      title: '运行记录',
      description: '智能体执行次数',
      icon: <NodeIndexOutlined />,
      tone: 'green',
    },
    {
      key: 'workflow_runs',
      title: '工作流执行',
      description: '工作流运行次数',
      icon: <BarChartOutlined />,
      tone: 'amber',
    },
    {
      key: 'alerts',
      title: '告警事件',
      description: '待处理告警数',
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
        status: dashboardConnected ? '正常' : '未接入',
      })),
    [dashboardConnected, summary],
  );

  const runtimeSignals: RuntimeSignal[] = [
    {
      name: '服务健康状态',
      status: healthStatus === 'healthy' ? 'success' : healthStatus ? 'warning' : 'default',
      value: healthStatus === 'healthy' ? '正常' : healthStatus || '检测中',
    },
    {
      name: '运行实例',
      status: runtimeTotal === null ? 'default' : runtimeTotal > 0 ? 'processing' : 'warning',
      value: runtimeTotal === null ? '检测中' : `${runtimeTotal} 个`,
    },
    {
      name: '可观测性',
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
            <Tag color="processing">AIPulse AgentOS</Tag>
            <Tag color={dashboardConnected ? 'success' : 'error'}>{dashboardConnected ? '系统运行正常' : '系统未连接'}</Tag>
          </Space>
          <Typography.Title level={1}>运行指挥中心</Typography.Title>
        </div>
        <div className="hero-control-panel">
          <div className="hero-control-title">
            <CloudServerOutlined />
            <span>系统信息</span>
          </div>
          <div className="hero-control-grid">
            <span>租户</span>
            <strong>{tenantName}</strong>
            <span>环境</span>
            <strong>生产环境</strong>
            <span>版本</span>
            <strong>v1.0</strong>
          </div>
        </div>
      </section>

      {error ? <Alert className="mb-4" type="error" showIcon message="数据加载失败" description={error} /> : null}

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
                <span>平台能力概览</span>
              </Space>
            }
          >
            <div className="lifecycle-strip">
              {[
                { title: '开发', text: '智能体、工作流、提示词', icon: <ApartmentOutlined /> },
                { title: '发布', text: '版本管理、灰度发布', icon: <ThunderboltOutlined /> },
                { title: '运行', text: '对话、调度、多智能体协同', icon: <ApiOutlined /> },
                { title: '治理', text: '安全护栏、审计、成本管控', icon: <SafetyCertificateOutlined /> },
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
                </div>
              </Col>
              <Col xs={24} md={12}>
                <div className="production-risk-card">
                  <ToolOutlined />
                  <Typography.Text type="secondary">质量缺陷</Typography.Text>
                  <strong>{badCaseCount}</strong>
                </div>
              </Col>
            </Row>
            {alertCount === 0 && badCaseCount === 0 ? (
              <Empty className="mt-4" image={Empty.PRESENTED_IMAGE_SIMPLE} description="当前无告警或质量缺陷，系统运行正常" />
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
                <Typography.Text strong>暂无运行数据</Typography.Text>
                <Typography.Text type="secondary">系统启动后将自动采集运行指标数据。</Typography.Text>
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
}
