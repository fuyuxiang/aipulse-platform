import { Button, Card, Drawer, Form, Input, Select, Space, Switch, Table, Tag, Tabs, message, Statistic, Row, Col } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import React, { useEffect, useState } from 'react';
import { api, getToken } from '../../services/http';

export function GuardrailsPage(): JSX.Element {
  const [policies, setPolicies] = useState<any[]>([]);
  const [violations, setViolations] = useState<any[]>([]);
  const [stats, setStats] = useState<any>({});
  const [loading, setLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [testDrawerOpen, setTestDrawerOpen] = useState(false);
  const [testContent, setTestContent] = useState('');
  const [testScope, setTestScope] = useState('input');
  const [testResult, setTestResult] = useState<any>(null);
  const [form] = Form.useForm();

  const load = async (): Promise<void> => {
    setLoading(true);
    try {
      const [pRes, vRes, sRes] = await Promise.all([
        fetch('/api/v1/guardrails/policies?page=1&page_size=50', { headers: { Authorization: `Bearer ${getToken()}` } }),
        fetch('/api/v1/guardrails/violations?page=1&page_size=50', { headers: { Authorization: `Bearer ${getToken()}` } }),
        fetch('/api/v1/guardrails/stats', { headers: { Authorization: `Bearer ${getToken()}` } }),
      ]);
      if (pRes.ok) setPolicies((await pRes.json()).items || []);
      if (vRes.ok) setViolations((await vRes.json()).items || []);
      if (sRes.ok) setStats(await sRes.json());
    } catch { /* ignore */ }
    setLoading(false);
  };

  useEffect(() => { void load(); }, []);

  const createPolicy = async (): Promise<void> => {
    try {
      const values = await form.validateFields();
      await fetch('/api/v1/guardrails/policies', {
        method: 'POST',
        headers: { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: values.name,
          description: values.description,
          scope: values.scope,
          action: values.action,
          enabled_checks: values.enabled_checks,
        }),
      });
      message.success('策略创建成功');
      setDrawerOpen(false);
      form.resetFields();
      await load();
    } catch { message.error('创建失败'); }
  };

  const runTest = async (): Promise<void> => {
    try {
      const endpoint = testScope === 'input' ? '/api/v1/guardrails/check-input' : '/api/v1/guardrails/check-output';
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: testContent }),
      });
      if (res.ok) setTestResult(await res.json());
    } catch { message.error('检测失败'); }
  };

  const policyColumns: ColumnsType<any> = [
    { title: '名称', dataIndex: 'name' },
    { title: '范围', render: (_, r) => <Tag>{r.spec?.scope || 'both'}</Tag> },
    { title: '动作', render: (_, r) => <Tag color={r.spec?.action === 'block' ? 'red' : 'orange'}>{r.spec?.action || 'block'}</Tag> },
    { title: '检查项', render: (_, r) => (r.spec?.enabled_checks || []).length },
    { title: '状态', dataIndex: 'status', render: (v: string) => <Tag color={v === 'active' ? 'green' : 'default'}>{v}</Tag> },
  ];

  const violationColumns: ColumnsType<any> = [
    { title: '类型', render: (_, r) => <Tag color="red">{r.spec?.type || ''}</Tag> },
    { title: '子类型', render: (_, r) => r.spec?.subtype || '' },
    { title: '严重性', render: (_, r) => <Tag color={r.spec?.severity === 'critical' ? 'red' : r.spec?.severity === 'high' ? 'orange' : 'blue'}>{r.spec?.severity || ''}</Tag> },
    { title: '消息', render: (_, r) => r.spec?.message || '' },
    { title: '动作', dataIndex: 'status', render: (v: string) => <Tag>{v}</Tag> },
  ];

  return (
    <div className="p-5">
      <Row gutter={16} className="mb-4">
        <Col span={8}><Card><Statistic title="护栏策略数" value={stats.total_policies || 0} /></Card></Col>
        <Col span={8}><Card><Statistic title="总检查次数" value={stats.total_executions || 0} /></Card></Col>
        <Col span={8}><Card><Statistic title="违规拦截数" value={stats.total_violations || 0} valueStyle={{ color: '#cf1322' }} /></Card></Col>
      </Row>

      <div className="mb-4 flex justify-between">
        <Space>
          <Button type="primary" onClick={() => setDrawerOpen(true)}>创建策略</Button>
          <Button onClick={() => setTestDrawerOpen(true)}>护栏测试</Button>
        </Space>
      </div>

      <Tabs items={[
        { key: 'policies', label: '护栏策略', children: <Table rowKey="id" columns={policyColumns} dataSource={policies} loading={loading} pagination={{ pageSize: 20 }} /> },
        { key: 'violations', label: '违规记录', children: <Table rowKey="id" columns={violationColumns} dataSource={violations} loading={loading} pagination={{ pageSize: 20 }} /> },
      ]} />

      <Drawer open={drawerOpen} title="创建护栏策略" width={520} onClose={() => setDrawerOpen(false)} extra={<Button type="primary" onClick={() => void createPolicy()}>创建</Button>}>
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="策略名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="description" label="描述"><Input.TextArea rows={2} /></Form.Item>
          <Form.Item name="scope" label="检查范围" initialValue="both">
            <Select options={[{ value: 'both', label: '输入+输出' }, { value: 'input', label: '仅输入' }, { value: 'output', label: '仅输出' }]} />
          </Form.Item>
          <Form.Item name="action" label="违规动作" initialValue="block">
            <Select options={[{ value: 'block', label: '拦截' }, { value: 'warn', label: '警告' }, { value: 'log', label: '仅记录' }]} />
          </Form.Item>
          <Form.Item name="enabled_checks" label="启用检查" initialValue={['pii_detection', 'prompt_injection', 'content_safety']}>
            <Select mode="multiple" options={[
              { value: 'pii_detection', label: 'PII 检测' },
              { value: 'prompt_injection', label: 'Prompt 注入检测' },
              { value: 'content_safety', label: '内容安全' },
              { value: 'output_format', label: '输出格式校验' },
              { value: 'hallucination_detection', label: '幻觉检测' },
              { value: 'topic_restriction', label: '话题限制' },
            ]} />
          </Form.Item>
        </Form>
      </Drawer>

      <Drawer open={testDrawerOpen} title="护栏测试" width={600} onClose={() => { setTestDrawerOpen(false); setTestResult(null); }}>
        <Space direction="vertical" className="w-full" size="middle">
          <Select value={testScope} onChange={setTestScope} className="w-full" options={[{ value: 'input', label: '输入检测' }, { value: 'output', label: '输出检测' }]} />
          <Input.TextArea rows={5} value={testContent} onChange={(e) => setTestContent(e.target.value)} placeholder="输入要检测的内容..." />
          <Button type="primary" onClick={() => void runTest()}>执行检测</Button>
          {testResult && (
            <Card size="small" title={testResult.passed ? '✅ 通过' : '❌ 未通过'}>
              <p>动作: <Tag color={testResult.action === 'block' ? 'red' : 'orange'}>{testResult.action}</Tag></p>
              {testResult.violations?.length > 0 && (
                <div>
                  <p className="font-medium">违规详情:</p>
                  {testResult.violations.map((v: any, i: number) => (
                    <Tag key={i} color="red" className="mb-1">{v.type}: {v.message}</Tag>
                  ))}
                </div>
              )}
              {testResult.masked_content && <p className="mt-2">脱敏结果: {testResult.masked_content}</p>}
            </Card>
          )}
        </Space>
      </Drawer>
    </div>
  );
}
