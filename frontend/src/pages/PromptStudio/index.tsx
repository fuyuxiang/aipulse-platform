import { Button, Card, Col, Drawer, Form, Input, Row, Select, Space, Table, Tabs, Tag, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import React, { useEffect, useState } from 'react';
import { getToken } from '../../services/http';

export function PromptStudioPage(): JSX.Element {
  const [templates, setTemplates] = useState<any[]>([]);
  const [abTests, setAbTests] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [playgroundOpen, setPlaygroundOpen] = useState(false);
  const [abDrawerOpen, setAbDrawerOpen] = useState(false);
  const [playgroundPrompt, setPlaygroundPrompt] = useState('');
  const [playgroundResult, setPlaygroundResult] = useState<any>(null);
  const [playgroundVars, setPlaygroundVars] = useState<Record<string, string>>({});
  const [selectedTemplate, setSelectedTemplate] = useState<any>(null);
  const [form] = Form.useForm();
  const [abForm] = Form.useForm();

  const headers = { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' };

  const load = async (): Promise<void> => {
    setLoading(true);
    try {
      const [tRes, aRes] = await Promise.all([
        fetch('/api/v1/prompt-templates?page=1&page_size=50', { headers }),
        fetch('/api/v1/prompt-ab-tests?page=1&page_size=50', { headers }),
      ]);
      if (tRes.ok) setTemplates((await tRes.json()).items || []);
      if (aRes.ok) setAbTests((await aRes.json()).items || []);
    } catch { /* ignore */ }
    setLoading(false);
  };

  useEffect(() => { void load(); }, []);

  const createTemplate = async (): Promise<void> => {
    try {
      const values = await form.validateFields();
      await fetch('/api/v1/prompt-templates', {
        method: 'POST', headers,
        body: JSON.stringify({ name: values.name, content: values.content, category: values.category, description: values.description }),
      });
      message.success('模板创建成功');
      setDrawerOpen(false);
      form.resetFields();
      await load();
    } catch { message.error('创建失败'); }
  };

  const runPlayground = async (): Promise<void> => {
    try {
      const body: any = { raw_prompt: playgroundPrompt, variables: playgroundVars };
      if (selectedTemplate) body.template_id = selectedTemplate.id;
      const res = await fetch('/api/v1/prompt-playground/run', { method: 'POST', headers, body: JSON.stringify(body) });
      if (res.ok) setPlaygroundResult(await res.json());
    } catch { message.error('执行失败'); }
  };

  const createAbTest = async (): Promise<void> => {
    try {
      const values = await abForm.validateFields();
      const res = await fetch('/api/v1/prompt-ab-tests', {
        method: 'POST', headers,
        body: JSON.stringify({
          name: values.name,
          variant_a: { content: values.variant_a },
          variant_b: { content: values.variant_b },
          test_cases: JSON.parse(values.test_cases || '[]'),
        }),
      });
      if (res.ok) {
        const test = await res.json();
        await fetch(`/api/v1/prompt-ab-tests/${test.id}/run`, { method: 'POST', headers });
        message.success('A/B 测试已创建并执行');
        setAbDrawerOpen(false);
        abForm.resetFields();
        await load();
      }
    } catch { message.error('创建失败'); }
  };

  const templateColumns: ColumnsType<any> = [
    { title: '名称', dataIndex: 'name' },
    { title: '分类', render: (_, r) => <Tag>{r.spec?.category || 'general'}</Tag> },
    { title: '变量数', render: (_, r) => (r.spec?.variables || []).length },
    { title: '版本', render: (_, r) => r.spec?.version || 1 },
    { title: '使用次数', render: (_, r) => r.spec?.usage_count || 0 },
    {
      title: '操作', render: (_, r) => (
        <Space>
          <Button size="small" onClick={() => { setSelectedTemplate(r); setPlaygroundPrompt(r.spec?.content || ''); setPlaygroundOpen(true); }}>Playground</Button>
        </Space>
      ),
    },
  ];

  const abColumns: ColumnsType<any> = [
    { title: '名称', dataIndex: 'name' },
    { title: '状态', dataIndex: 'status', render: (v: string) => <Tag color={v === 'completed' ? 'green' : 'blue'}>{v}</Tag> },
    { title: '用例数', render: (_, r) => r.spec?.total_cases || 0 },
    { title: '胜者', render: (_, r) => r.spec?.summary?.winner ? <Tag color="gold">Variant {r.spec.summary.winner.toUpperCase()}</Tag> : '-' },
    { title: 'A 得分', render: (_, r) => r.spec?.summary?.avg_score_a?.toFixed(2) || '-' },
    { title: 'B 得分', render: (_, r) => r.spec?.summary?.avg_score_b?.toFixed(2) || '-' },
  ];

  return (
    <div className="p-5">
      <Tabs items={[
        {
          key: 'templates', label: 'Prompt 模板',
          children: (
            <>
              <div className="mb-4 flex justify-between">
                <Button type="primary" onClick={() => setDrawerOpen(true)}>创建模板</Button>
                <Button onClick={() => { setSelectedTemplate(null); setPlaygroundPrompt(''); setPlaygroundOpen(true); }}>Playground</Button>
              </div>
              <Table rowKey="id" columns={templateColumns} dataSource={templates} loading={loading} pagination={{ pageSize: 20 }} />
            </>
          ),
        },
        {
          key: 'ab', label: 'A/B 测试',
          children: (
            <>
              <div className="mb-4"><Button type="primary" onClick={() => setAbDrawerOpen(true)}>创建 A/B 测试</Button></div>
              <Table rowKey="id" columns={abColumns} dataSource={abTests} loading={loading} pagination={{ pageSize: 20 }} />
            </>
          ),
        },
      ]} />

      <Drawer open={drawerOpen} title="创建 Prompt 模板" width={600} onClose={() => setDrawerOpen(false)} extra={<Button type="primary" onClick={() => void createTemplate()}>创建</Button>}>
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="模板名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="category" label="分类" initialValue="general">
            <Select options={[{ value: 'general', label: '通用' }, { value: 'chat', label: '对话' }, { value: 'extraction', label: '信息提取' }, { value: 'summarization', label: '摘要' }, { value: 'coding', label: '编程' }]} />
          </Form.Item>
          <Form.Item name="description" label="描述"><Input.TextArea rows={2} /></Form.Item>
          <Form.Item name="content" label="模板内容 (用 {{变量名}} 标记变量)" rules={[{ required: true }]}>
            <Input.TextArea rows={10} placeholder="你是一个{{role}}，请帮我{{task}}。\n\n上下文：{{context}}" />
          </Form.Item>
        </Form>
      </Drawer>

      <Drawer open={playgroundOpen} title="Prompt Playground" width={700} onClose={() => { setPlaygroundOpen(false); setPlaygroundResult(null); }}>
        <Space direction="vertical" className="w-full" size="middle">
          {selectedTemplate && (
            <Card size="small" title={`模板: ${selectedTemplate.name}`}>
              <div className="text-xs text-gray-500">变量: {(selectedTemplate.spec?.variables || []).map((v: any) => v.name).join(', ')}</div>
            </Card>
          )}
          <Input.TextArea rows={8} value={playgroundPrompt} onChange={(e) => setPlaygroundPrompt(e.target.value)} placeholder="输入 Prompt..." />
          {selectedTemplate?.spec?.variables?.map((v: any) => (
            <Input key={v.name} addonBefore={v.name} value={playgroundVars[v.name] || ''} onChange={(e) => setPlaygroundVars({ ...playgroundVars, [v.name]: e.target.value })} />
          ))}
          <Button type="primary" onClick={() => void runPlayground()}>执行</Button>
          {playgroundResult && (
            <Card size="small" title="结果">
              <p className="whitespace-pre-wrap text-sm">{playgroundResult.response}</p>
              <div className="mt-2 text-xs text-gray-400">
                Token: {playgroundResult.token_usage?.total_tokens} | 延迟: {playgroundResult.latency_ms}ms
              </div>
            </Card>
          )}
        </Space>
      </Drawer>

      <Drawer open={abDrawerOpen} title="创建 A/B 测试" width={600} onClose={() => setAbDrawerOpen(false)} extra={<Button type="primary" onClick={() => void createAbTest()}>创建并执行</Button>}>
        <Form form={abForm} layout="vertical">
          <Form.Item name="name" label="测试名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="variant_a" label="Variant A (Prompt)" rules={[{ required: true }]}><Input.TextArea rows={4} /></Form.Item>
          <Form.Item name="variant_b" label="Variant B (Prompt)" rules={[{ required: true }]}><Input.TextArea rows={4} /></Form.Item>
          <Form.Item name="test_cases" label="测试用例 JSON" initialValue='[{"variables":{},"expected":""}]'>
            <Input.TextArea rows={5} />
          </Form.Item>
        </Form>
      </Drawer>
    </div>
  );
}
