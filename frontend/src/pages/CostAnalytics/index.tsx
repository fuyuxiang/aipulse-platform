import { DollarOutlined } from '@ant-design/icons';
import { Button, Card, Col, Drawer, Form, Input, Progress, Row, Select, Space, Statistic, Table, Tabs, Tag, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import React, { useEffect, useState } from 'react';
import { getToken } from '../../services/http';

export function CostAnalyticsPage(): JSX.Element {
  const [summary, setSummary] = useState<any>({});
  const [budgets, setBudgets] = useState<any[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [budgetDrawerOpen, setBudgetDrawerOpen] = useState(false);
  const [form] = Form.useForm();

  const headers = { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' };

  const load = async (): Promise<void> => {
    setLoading(true);
    try {
      const [sRes, bRes, aRes] = await Promise.all([
        fetch('/api/v1/cost/summary', { headers }),
        fetch('/api/v1/cost/budgets?page=1&page_size=50', { headers }),
        fetch('/api/v1/cost/alerts?page=1&page_size=50', { headers }),
      ]);
      if (sRes.ok) setSummary(await sRes.json());
      if (bRes.ok) setBudgets((await bRes.json()).items || []);
      if (aRes.ok) setAlerts((await aRes.json()).items || []);
    } catch { /* ignore */ }
    setLoading(false);
  };

  useEffect(() => { void load(); }, []);

  const createBudget = async (): Promise<void> => {
    try {
      const values = await form.validateFields();
      await fetch('/api/v1/cost/budgets', {
        method: 'POST', headers,
        body: JSON.stringify({
          name: values.name,
          scope: values.scope,
          agent_id: values.agent_id || '',
          period: values.period,
          limit_amount: parseFloat(values.limit_amount),
          warning_threshold: parseFloat(values.warning_threshold || '0.8'),
          action_on_exceed: values.action_on_exceed,
        }),
      });
      message.success('预算创建成功');
      setBudgetDrawerOpen(false);
      form.resetFields();
      await load();
    } catch { message.error('创建失败'); }
  };

  const budgetColumns: ColumnsType<any> = [
    { title: '名称', dataIndex: 'name' },
    { title: '范围', render: (_, r) => <Tag>{r.spec?.scope || 'tenant'}</Tag> },
    { title: '周期', render: (_, r) => r.spec?.period || 'monthly' },
    { title: '限额', render: (_, r) => `$${r.spec?.limit_amount?.toFixed(2) || '0.00'}` },
    {
      title: '使用率', render: (_, r) => {
        const usage = r.spec?.current_usage || 0;
        const limit = r.spec?.limit_amount || 100;
        const percent = Math.min(100, Math.round((usage / limit) * 100));
        return <Progress percent={percent} size="small" status={percent >= 100 ? 'exception' : percent >= 80 ? 'active' : 'normal'} />;
      },
    },
    { title: '当前用量', render: (_, r) => `$${(r.spec?.current_usage || 0).toFixed(4)}` },
    { title: '超额动作', render: (_, r) => <Tag color={r.spec?.action_on_exceed === 'block' ? 'red' : 'orange'}>{r.spec?.action_on_exceed || 'alert'}</Tag> },
  ];

  const alertColumns: ColumnsType<any> = [
    { title: '名称', dataIndex: 'name' },
    { title: '类型', render: (_, r) => <Tag color={r.spec?.alert_type === 'budget_exceeded' ? 'red' : 'orange'}>{r.spec?.alert_type || ''}</Tag> },
    { title: '限额', render: (_, r) => `$${r.spec?.limit?.toFixed(2) || ''}` },
    { title: '当前', render: (_, r) => `$${r.spec?.current?.toFixed(4) || ''}` },
    { title: '时间', render: (_, r) => r.spec?.triggered_at?.slice(0, 19) || '' },
    { title: '状态', dataIndex: 'status', render: (v: string) => <Tag color={v === 'triggered' ? 'red' : 'orange'}>{v}</Tag> },
  ];

  return (
    <div className="p-5">
      <Row gutter={16} className="mb-4">
        <Col span={6}><Card><Statistic title="总成本" value={summary.total_cost || 0} prefix="$" precision={4} /></Card></Col>
        <Col span={6}><Card><Statistic title="总调用次数" value={summary.total_records || 0} /></Card></Col>
        <Col span={6}><Card><Statistic title="输入 Tokens" value={summary.total_input_tokens || 0} /></Card></Col>
        <Col span={6}><Card><Statistic title="输出 Tokens" value={summary.total_output_tokens || 0} /></Card></Col>
      </Row>

      {summary.by_agent?.length > 0 && (
        <Row gutter={16} className="mb-4">
          <Col span={12}>
            <Card title="按 Agent 成本 Top 10" size="small">
              {summary.by_agent.slice(0, 10).map((item: any, i: number) => (
                <div key={i} className="flex justify-between py-1 border-b">
                  <span className="text-sm">{item.agent_id?.slice(0, 12) || 'unknown'}</span>
                  <span className="text-sm font-medium">${item.cost?.toFixed(4)}</span>
                </div>
              ))}
            </Card>
          </Col>
          <Col span={12}>
            <Card title="按模型成本 Top 10" size="small">
              {(summary.by_model || []).slice(0, 10).map((item: any, i: number) => (
                <div key={i} className="flex justify-between py-1 border-b">
                  <span className="text-sm">{item.model_id?.slice(0, 12) || 'unknown'}</span>
                  <span className="text-sm font-medium">${item.cost?.toFixed(4)}</span>
                </div>
              ))}
            </Card>
          </Col>
        </Row>
      )}

      {summary.daily_trend?.length > 0 && (
        <Card title="每日成本趋势" size="small" className="mb-4">
          <div className="flex items-end gap-1 h-24">
            {summary.daily_trend.map((d: any, i: number) => {
              const maxCost = Math.max(...summary.daily_trend.map((x: any) => x.cost));
              const height = maxCost > 0 ? (d.cost / maxCost) * 80 : 0;
              return (
                <div key={i} className="flex flex-col items-center flex-1">
                  <div className="bg-blue-400 w-full rounded-t" style={{ height: `${height}px` }} title={`${d.date}: $${d.cost.toFixed(4)}`} />
                  <span className="text-[10px] text-gray-400 mt-1">{d.date?.slice(5)}</span>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      <Tabs items={[
        {
          key: 'budgets', label: '预算管理',
          children: (
            <>
              <div className="mb-3"><Button type="primary" onClick={() => setBudgetDrawerOpen(true)}>创建预算</Button></div>
              <Table rowKey="id" columns={budgetColumns} dataSource={budgets} loading={loading} pagination={{ pageSize: 20 }} />
            </>
          ),
        },
        {
          key: 'alerts', label: '成本告警',
          children: <Table rowKey="id" columns={alertColumns} dataSource={alerts} loading={loading} pagination={{ pageSize: 20 }} />,
        },
      ]} />

      <Drawer open={budgetDrawerOpen} title="创建预算" width={480} onClose={() => setBudgetDrawerOpen(false)} extra={<Button type="primary" onClick={() => void createBudget()}>创建</Button>}>
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="预算名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="scope" label="范围" initialValue="tenant">
            <Select options={[{ value: 'tenant', label: '租户级' }, { value: 'agent', label: 'Agent 级' }, { value: 'user', label: '用户级' }]} />
          </Form.Item>
          <Form.Item name="agent_id" label="Agent ID (Agent 级时填写)"><Input /></Form.Item>
          <Form.Item name="period" label="周期" initialValue="monthly">
            <Select options={[{ value: 'daily', label: '每日' }, { value: 'weekly', label: '每周' }, { value: 'monthly', label: '每月' }]} />
          </Form.Item>
          <Form.Item name="limit_amount" label="限额 ($)" rules={[{ required: true }]}><Input type="number" /></Form.Item>
          <Form.Item name="warning_threshold" label="预警阈值" initialValue="0.8"><Input type="number" step="0.1" /></Form.Item>
          <Form.Item name="action_on_exceed" label="超额动作" initialValue="alert">
            <Select options={[{ value: 'alert', label: '告警' }, { value: 'block', label: '阻断' }, { value: 'throttle', label: '限流' }]} />
          </Form.Item>
        </Form>
      </Drawer>
    </div>
  );
}
