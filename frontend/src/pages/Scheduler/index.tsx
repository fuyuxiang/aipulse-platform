import { Button, Card, Col, Drawer, Form, Input, Row, Select, Space, Statistic, Switch, Table, Tabs, Tag, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import React, { useEffect, useState } from 'react';
import { getToken } from '../../services/http';

export function SchedulerPage(): JSX.Element {
  const [jobs, setJobs] = useState<any[]>([]);
  const [webhooks, setWebhooks] = useState<any[]>([]);
  const [triggers, setTriggers] = useState<any[]>([]);
  const [stats, setStats] = useState<any>({});
  const [loading, setLoading] = useState(false);
  const [jobDrawerOpen, setJobDrawerOpen] = useState(false);
  const [webhookDrawerOpen, setWebhookDrawerOpen] = useState(false);
  const [triggerDrawerOpen, setTriggerDrawerOpen] = useState(false);
  const [jobForm] = Form.useForm();
  const [webhookForm] = Form.useForm();
  const [triggerForm] = Form.useForm();

  const headers = { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' };

  const load = async (): Promise<void> => {
    setLoading(true);
    try {
      const [jRes, wRes, tRes, sRes] = await Promise.all([
        fetch('/api/v1/scheduler/jobs?page=1&page_size=50', { headers }),
        fetch('/api/v1/scheduler/webhooks?page=1&page_size=50', { headers }),
        fetch('/api/v1/scheduler/triggers?page=1&page_size=50', { headers }),
        fetch('/api/v1/scheduler/stats', { headers }),
      ]);
      if (jRes.ok) setJobs((await jRes.json()).items || []);
      if (wRes.ok) setWebhooks((await wRes.json()).items || []);
      if (tRes.ok) setTriggers((await tRes.json()).items || []);
      if (sRes.ok) setStats(await sRes.json());
    } catch { /* ignore */ }
    setLoading(false);
  };

  useEffect(() => { void load(); }, []);

  const createJob = async (): Promise<void> => {
    try {
      const values = await jobForm.validateFields();
      await fetch('/api/v1/scheduler/jobs', {
        method: 'POST', headers,
        body: JSON.stringify({
          name: values.name,
          job_type: values.job_type,
          target_type: values.target_type,
          target_id: values.target_id,
          cron_expression: values.cron_expression,
          interval_seconds: values.interval_seconds ? parseInt(values.interval_seconds) : 0,
          timeout_seconds: values.timeout_seconds ? parseInt(values.timeout_seconds) : 300,
        }),
      });
      message.success('任务创建成功');
      setJobDrawerOpen(false);
      jobForm.resetFields();
      await load();
    } catch { message.error('创建失败'); }
  };

  const triggerJob = async (jobId: string): Promise<void> => {
    try {
      await fetch(`/api/v1/scheduler/jobs/${jobId}/trigger`, { method: 'POST', headers, body: '{}' });
      message.success('已触发执行');
      await load();
    } catch { message.error('触发失败'); }
  };

  const toggleJob = async (jobId: string, enabled: boolean): Promise<void> => {
    const endpoint = enabled ? 'enable' : 'disable';
    await fetch(`/api/v1/scheduler/jobs/${jobId}/${endpoint}`, { method: 'POST', headers });
    await load();
  };

  const createWebhook = async (): Promise<void> => {
    try {
      const values = await webhookForm.validateFields();
      await fetch('/api/v1/scheduler/webhooks', {
        method: 'POST', headers,
        body: JSON.stringify({ name: values.name, target_type: values.target_type, target_id: values.target_id }),
      });
      message.success('Webhook 创建成功');
      setWebhookDrawerOpen(false);
      webhookForm.resetFields();
      await load();
    } catch { message.error('创建失败'); }
  };

  const createTrigger = async (): Promise<void> => {
    try {
      const values = await triggerForm.validateFields();
      await fetch('/api/v1/scheduler/triggers', {
        method: 'POST', headers,
        body: JSON.stringify({
          name: values.name,
          event_source: values.event_source,
          event_type: values.event_type,
          target_type: values.target_type,
          target_id: values.target_id,
        }),
      });
      message.success('触发器创建成功');
      setTriggerDrawerOpen(false);
      triggerForm.resetFields();
      await load();
    } catch { message.error('创建失败'); }
  };

  const jobColumns: ColumnsType<any> = [
    { title: '名称', dataIndex: 'name' },
    { title: '类型', render: (_, r) => <Tag color="blue">{r.spec?.job_type || 'cron'}</Tag> },
    { title: '目标', render: (_, r) => `${r.spec?.target_type || ''}:${(r.spec?.target_id || '').slice(0, 8)}` },
    { title: 'Cron', render: (_, r) => r.spec?.cron_expression || '-' },
    { title: '总执行', render: (_, r) => r.spec?.total_runs || 0 },
    { title: '成功', render: (_, r) => <span className="text-green-500">{r.spec?.success_count || 0}</span> },
    { title: '失败', render: (_, r) => <span className="text-red-500">{r.spec?.failure_count || 0}</span> },
    { title: '启用', render: (_, r) => <Switch size="small" checked={r.status === 'active'} onChange={(v) => void toggleJob(r.id, v)} /> },
    {
      title: '操作', render: (_, r) => (
        <Button size="small" type="primary" onClick={() => void triggerJob(r.id)}>手动触发</Button>
      ),
    },
  ];

  const webhookColumns: ColumnsType<any> = [
    { title: '名称', dataIndex: 'name' },
    { title: '目标', render: (_, r) => `${r.spec?.target_type || ''}:${(r.spec?.target_id || '').slice(0, 8)}` },
    { title: 'URL', render: (_, r) => <code className="text-xs">{r.spec?.url_path || ''}</code> },
    { title: '调用次数', render: (_, r) => r.spec?.total_calls || 0 },
    { title: '状态', dataIndex: 'status', render: (v: string) => <Tag color={v === 'active' ? 'green' : 'default'}>{v}</Tag> },
  ];

  const triggerColumns: ColumnsType<any> = [
    { title: '名称', dataIndex: 'name' },
    { title: '事件源', render: (_, r) => r.spec?.event_source || '' },
    { title: '事件类型', render: (_, r) => <Tag>{r.spec?.event_type || ''}</Tag> },
    { title: '目标', render: (_, r) => `${r.spec?.target_type || ''}:${(r.spec?.target_id || '').slice(0, 8)}` },
    { title: '状态', dataIndex: 'status', render: (v: string) => <Tag color={v === 'active' ? 'green' : 'default'}>{v}</Tag> },
  ];

  return (
    <div className="p-5">
      <Row gutter={16} className="mb-4">
        <Col span={6}><Card><Statistic title="定时任务" value={stats.total_jobs || 0} /></Card></Col>
        <Col span={6}><Card><Statistic title="总执行次数" value={stats.total_executions || 0} /></Card></Col>
        <Col span={6}><Card><Statistic title="Webhooks" value={stats.total_webhooks || 0} /></Card></Col>
        <Col span={6}><Card><Statistic title="事件触发器" value={stats.total_triggers || 0} /></Card></Col>
      </Row>

      <Tabs items={[
        {
          key: 'jobs', label: '定时任务',
          children: (
            <>
              <div className="mb-3"><Button type="primary" onClick={() => setJobDrawerOpen(true)}>创建任务</Button></div>
              <Table rowKey="id" columns={jobColumns} dataSource={jobs} loading={loading} pagination={{ pageSize: 20 }} />
            </>
          ),
        },
        {
          key: 'webhooks', label: 'Webhooks',
          children: (
            <>
              <div className="mb-3"><Button type="primary" onClick={() => setWebhookDrawerOpen(true)}>创建 Webhook</Button></div>
              <Table rowKey="id" columns={webhookColumns} dataSource={webhooks} loading={loading} pagination={{ pageSize: 20 }} />
            </>
          ),
        },
        {
          key: 'triggers', label: '事件触发器',
          children: (
            <>
              <div className="mb-3"><Button type="primary" onClick={() => setTriggerDrawerOpen(true)}>创建触发器</Button></div>
              <Table rowKey="id" columns={triggerColumns} dataSource={triggers} loading={loading} pagination={{ pageSize: 20 }} />
            </>
          ),
        },
      ]} />

      <Drawer open={jobDrawerOpen} title="创建定时任务" width={480} onClose={() => setJobDrawerOpen(false)} extra={<Button type="primary" onClick={() => void createJob()}>创建</Button>}>
        <Form form={jobForm} layout="vertical">
          <Form.Item name="name" label="任务名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="job_type" label="类型" initialValue="cron">
            <Select options={[{ value: 'cron', label: 'Cron 定时' }, { value: 'interval', label: '固定间隔' }, { value: 'event', label: '事件触发' }]} />
          </Form.Item>
          <Form.Item name="target_type" label="目标类型" initialValue="agent">
            <Select options={[{ value: 'agent', label: 'Agent' }, { value: 'workflow', label: 'Workflow' }]} />
          </Form.Item>
          <Form.Item name="target_id" label="目标 ID" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="cron_expression" label="Cron 表达式"><Input placeholder="0 9 * * *" /></Form.Item>
          <Form.Item name="interval_seconds" label="间隔秒数"><Input type="number" /></Form.Item>
          <Form.Item name="timeout_seconds" label="超时秒数" initialValue="300"><Input type="number" /></Form.Item>
        </Form>
      </Drawer>

      <Drawer open={webhookDrawerOpen} title="创建 Webhook" width={480} onClose={() => setWebhookDrawerOpen(false)} extra={<Button type="primary" onClick={() => void createWebhook()}>创建</Button>}>
        <Form form={webhookForm} layout="vertical">
          <Form.Item name="name" label="名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="target_type" label="目标类型" initialValue="agent">
            <Select options={[{ value: 'agent', label: 'Agent' }, { value: 'workflow', label: 'Workflow' }]} />
          </Form.Item>
          <Form.Item name="target_id" label="目标 ID" rules={[{ required: true }]}><Input /></Form.Item>
        </Form>
      </Drawer>

      <Drawer open={triggerDrawerOpen} title="创建事件触发器" width={480} onClose={() => setTriggerDrawerOpen(false)} extra={<Button type="primary" onClick={() => void createTrigger()}>创建</Button>}>
        <Form form={triggerForm} layout="vertical">
          <Form.Item name="name" label="名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="event_source" label="事件源" rules={[{ required: true }]}><Input placeholder="如: agent_run, workflow_complete" /></Form.Item>
          <Form.Item name="event_type" label="事件类型" rules={[{ required: true }]}><Input placeholder="如: completed, failed" /></Form.Item>
          <Form.Item name="target_type" label="目标类型" initialValue="agent">
            <Select options={[{ value: 'agent', label: 'Agent' }, { value: 'workflow', label: 'Workflow' }]} />
          </Form.Item>
          <Form.Item name="target_id" label="目标 ID" rules={[{ required: true }]}><Input /></Form.Item>
        </Form>
      </Drawer>
    </div>
  );
}
