import { Card, Input, Space, Table, Tag, Tabs, Typography, Button, Drawer, Form, Switch, Select, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { PlusOutlined, ReloadOutlined, LockOutlined, KeyOutlined, GlobalOutlined, SafetyOutlined } from '@ant-design/icons';
import React, { useEffect, useState } from 'react';
import { api } from '../../../services/http';
import type { ResourceRecord } from '../../../models/types';

interface SecurityResource extends ResourceRecord {
  config?: Record<string, unknown>;
}

const TABS: Array<{ key: string; label: string; api: string; icon: React.ReactNode; description: string; new: string }> = [
  { key: 'content', label: '内容策略', api: '/security/content-policies', icon: <SafetyOutlined />, description: '过滤违规、隐私和高风险内容', new: '新建策略' },
  { key: 'sensitive', label: '敏感词', api: '/security/sensitive-rules', icon: <LockOutlined />, description: '关键词与正则规则的脱敏与拦截', new: '新建规则' },
  { key: 'injection', label: 'Prompt 注入', api: '/security/prompt-injection-rules', icon: <SafetyOutlined />, description: '识别并拦截提示词注入攻击', new: '新建规则' },
  { key: 'ip', label: 'IP 白名单', api: '/security/ip-allowlists', icon: <GlobalOutlined />, description: '限制平台与 API 的访问来源', new: '新建白名单' },
  { key: 'secrets', label: '凭证管理', api: '/security/secrets', icon: <KeyOutlined />, description: '统一管理 API Key、密钥与回调签名', new: '新建凭证' },
];

export function SecurityPage(): JSX.Element {
  const [active, setActive] = useState(TABS[0].key);
  const [data, setData] = useState<Record<string, SecurityResource[]>>({});
  const [open, setOpen] = useState(false);
  const [form] = Form.useForm();

  const tab = TABS.find((t) => t.key === active) || TABS[0];

  const load = async (key = active): Promise<void> => {
    const cur = TABS.find((t) => t.key === key) || TABS[0];
    try {
      const res = await api.list<SecurityResource>(cur.api, 1, 100);
      setData((prev) => ({ ...prev, [key]: res.items }));
    } catch (e) { message.error(e instanceof Error ? e.message : '加载失败'); }
  };

  useEffect(() => { void load(active); }, [active]);

  const submit = async (): Promise<void> => {
    try {
      const v = await form.validateFields();
      await api.create(tab.api, { name: v.name, code: v.code, description: v.description || '', enabled: v.enabled !== false,
        config: v.payload ? JSON.parse(v.payload) : {} });
      setOpen(false); form.resetFields(); await load(); message.success('已创建');
    } catch (e) {
      if ((e as { errorFields?: unknown }).errorFields) return;
      message.error(e instanceof Error ? e.message : '保存失败');
    }
  };

  const columns: ColumnsType<SecurityResource> = [
    { title: '名称', dataIndex: 'name', render: (v, r) => <Space><strong>{v}</strong><Tag>{r.code}</Tag></Space> },
    { title: '描述', dataIndex: 'description', ellipsis: true },
    { title: '状态', dataIndex: 'enabled', width: 100, render: (v) => <Tag color={v ? 'success' : 'default'}>{v ? '启用' : '停用'}</Tag> },
    { title: '更新时间', dataIndex: 'updated_at', width: 190 },
  ];

  const rows = data[active] || [];

  return (
    <div className="enterprise-page">
      <section className="resource-hero">
        <div>
          <Space wrap><Tag color="processing">安全策略</Tag><Tag>设置</Tag></Space>
          <Typography.Title level={1}>安全</Typography.Title>
          <Typography.Paragraph>统一管理内容安全、敏感词、注入防护、网络访问与凭证。</Typography.Paragraph>
        </div>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => void load()}>刷新</Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setOpen(true)}>{tab.new}</Button>
        </Space>
      </section>

      <Card bordered={false} className="enterprise-card">
        <Tabs
          activeKey={active}
          onChange={setActive}
          items={TABS.map((t) => ({ key: t.key, label: <Space>{t.icon}{t.label}</Space>,
            children: (
              <>
                <Typography.Paragraph type="secondary" style={{ marginBottom: 12 }}>{t.description}</Typography.Paragraph>
                <Table<SecurityResource> rowKey="id" dataSource={rows} columns={columns} pagination={{ pageSize: 20 }} />
              </>
            ),
          }))}
        />
      </Card>

      <Drawer open={open} onClose={() => setOpen(false)} width={520} title={tab.new} extra={<Button type="primary" onClick={() => void submit()}>保存</Button>}>
        <Form form={form} layout="vertical" initialValues={{ enabled: true }}>
          <Form.Item name="name" label="名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="code" label="编码"><Input /></Form.Item>
          <Form.Item name="description" label="描述"><Input.TextArea rows={2} /></Form.Item>
          {active === 'content' ? (
            <Form.Item name="payload" label="策略 JSON"><Input.TextArea rows={8} placeholder='{"block_categories":["violence","sexual"],"action":"block"}' /></Form.Item>
          ) : null}
          {active === 'sensitive' ? (
            <Form.Item name="payload" label="敏感词 JSON"><Input.TextArea rows={8} placeholder='{"words":["..."],"regex":["^.+@.+$"],"action":"mask"}' /></Form.Item>
          ) : null}
          {active === 'injection' ? (
            <Form.Item name="payload" label="注入规则 JSON"><Input.TextArea rows={8} placeholder='{"patterns":["ignore previous"],"action":"block"}' /></Form.Item>
          ) : null}
          {active === 'ip' ? (
            <Form.Item name="payload" label="网段 JSON"><Input.TextArea rows={6} placeholder='{"cidrs":["10.0.0.0/8","192.168.1.0/24"]}' /></Form.Item>
          ) : null}
          {active === 'secrets' ? (
            <Form.Item name="payload" label="凭证 JSON（密文存储）"><Input.TextArea rows={6} placeholder='{"type":"api_key","value":"..."}' /></Form.Item>
          ) : null}
          <Form.Item name="enabled" label="启用" valuePropName="checked"><Switch /></Form.Item>
        </Form>
      </Drawer>
    </div>
  );
}
