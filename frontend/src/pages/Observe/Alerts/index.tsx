import { Card, Input, Space, Table, Tag, Tabs, Typography, Button, Drawer, Form, Select, Switch, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { PlusOutlined, ReloadOutlined, AlertOutlined } from '@ant-design/icons';
import React, { useEffect, useState } from 'react';
import { api } from '../../../services/http';

interface AlertRule {
  id: string;
  name: string;
  code: string;
  status: string;
  enabled: boolean;
  description: string;
  config?: Record<string, unknown>;
  updated_at: string;
}

interface AlertEvent {
  id: string;
  name: string;
  status: string;
  description: string;
  metadata_json?: { severity?: string; rule_id?: string };
  created_at: string;
}

const SEVERITY_COLOR: Record<string, string> = { critical: 'error', high: 'volcano', medium: 'warning', low: 'default' };

export function AlertsPage(): JSX.Element {
  const [rules, setRules] = useState<AlertRule[]>([]);
  const [events, setEvents] = useState<AlertEvent[]>([]);
  const [open, setOpen] = useState(false);
  const [form] = Form.useForm();

  const load = async (): Promise<void> => {
    try {
      const [r, e] = await Promise.all([
        api.list<AlertRule>('/alerts/rules', 1, 50),
        api.list<AlertEvent>('/alerts/events', 1, 50),
      ]);
      setRules(r.items);
      setEvents(e.items);
    } catch (err) {
      message.error(err instanceof Error ? err.message : '加载失败');
    }
  };

  useEffect(() => { void load(); }, []);

  const submit = async (): Promise<void> => {
    try {
      const v = await form.validateFields();
      await api.create('/alerts/rules', {
        name: v.name,
        code: v.code,
        description: v.description,
        enabled: v.enabled !== false,
        config: { metric: v.metric, threshold: Number(v.threshold), op: v.op, channel: v.channel },
      });
      setOpen(false);
      form.resetFields();
      await load();
      message.success('已创建告警规则');
    } catch (e) {
      if ((e as { errorFields?: unknown }).errorFields) return;
      message.error(e instanceof Error ? e.message : '保存失败');
    }
  };

  const ruleCols: ColumnsType<AlertRule> = [
    { title: '规则名称', dataIndex: 'name' },
    { title: '指标', render: (_, r) => <Tag>{(r.config as { metric?: string })?.metric || '-'}</Tag> },
    { title: '阈值', render: (_, r) => `${(r.config as { op?: string })?.op || ''} ${(r.config as { threshold?: number })?.threshold ?? ''}` },
    { title: '通道', render: (_, r) => <Tag color="blue">{(r.config as { channel?: string })?.channel || '-'}</Tag> },
    { title: '状态', dataIndex: 'enabled', width: 100, render: (v) => <Tag color={v ? 'success' : 'default'}>{v ? '启用' : '停用'}</Tag> },
    { title: '更新时间', dataIndex: 'updated_at', width: 190 },
  ];

  const eventCols: ColumnsType<AlertEvent> = [
    { title: '触发时间', dataIndex: 'created_at', width: 190 },
    { title: '级别', width: 100, render: (_, r) => {
      const sev = r.metadata_json?.severity || 'medium';
      return <Tag color={SEVERITY_COLOR[sev] || 'default'}>{sev}</Tag>;
    } },
    { title: '事件', dataIndex: 'name' },
    { title: '描述', dataIndex: 'description', ellipsis: true },
    { title: '状态', dataIndex: 'status', width: 110, render: (v) => <Tag color={v === 'resolved' ? 'success' : 'warning'}>{v}</Tag> },
  ];

  return (
    <div className="enterprise-page">
      <section className="resource-hero">
        <div>
          <Space wrap><Tag color="processing">告警与响应</Tag><Tag>观测</Tag></Space>
          <Typography.Title level={1}>告警</Typography.Title>
          <Typography.Paragraph>配置告警规则与事件触达，保障 Agent 可用性、SLA 与成本边界。</Typography.Paragraph>
        </div>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => void load()}>刷新</Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setOpen(true)}>新建规则</Button>
        </Space>
      </section>

      <Card bordered={false} className="enterprise-card">
        <Tabs
          items={[
            { key: 'rules', label: <Space><AlertOutlined />告警规则 ({rules.length})</Space>,
              children: <Table<AlertRule> rowKey="id" dataSource={rules} columns={ruleCols} pagination={{ pageSize: 20 }} /> },
            { key: 'events', label: `告警事件 (${events.length})`,
              children: <Table<AlertEvent> rowKey="id" dataSource={events} columns={eventCols} pagination={{ pageSize: 20 }} /> },
          ]}
        />
      </Card>

      <Drawer open={open} onClose={() => setOpen(false)} width={520} title="新建告警规则" extra={<Button type="primary" onClick={() => void submit()}>保存</Button>}>
        <Form form={form} layout="vertical" initialValues={{ enabled: true, op: '>', channel: 'email' }}>
          <Form.Item name="name" label="规则名称" rules={[{ required: true }]}><Input placeholder="例：模型延迟告警" /></Form.Item>
          <Form.Item name="code" label="编码"><Input placeholder="alert_xxx" /></Form.Item>
          <Form.Item name="description" label="描述"><Input.TextArea rows={2} /></Form.Item>
          <Form.Item name="metric" label="监控指标" rules={[{ required: true }]}>
            <Select options={[
              { value: 'p95_latency_ms', label: 'P95 延迟（毫秒）' },
              { value: 'error_rate', label: '错误率' },
              { value: 'cost_per_hour', label: '小时成本' },
              { value: 'tool_call_failures', label: '工具调用失败次数' },
              { value: 'token_usage', label: 'Token 用量' },
            ]} />
          </Form.Item>
          <Form.Item name="op" label="比较"><Select options={[{ value: '>', label: '大于' }, { value: '<', label: '小于' }, { value: '>=', label: '大于等于' }]} /></Form.Item>
          <Form.Item name="threshold" label="阈值" rules={[{ required: true }]}><Input type="number" /></Form.Item>
          <Form.Item name="channel" label="通知渠道"><Select options={[{ value: 'email', label: 'Email' }, { value: 'webhook', label: 'Webhook' }, { value: 'sms', label: '短信' }, { value: 'feishu', label: '飞书' }]} /></Form.Item>
          <Form.Item name="enabled" label="启用" valuePropName="checked"><Switch /></Form.Item>
        </Form>
      </Drawer>
    </div>
  );
}
