import { Card, Input, Space, Table, Tag, Typography, Button, Drawer, Form, Switch, Select, message, Row, Col, Statistic } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { PlusOutlined, ReloadOutlined, SettingOutlined, CloudServerOutlined, DatabaseOutlined } from '@ant-design/icons';
import React, { useEffect, useState } from 'react';
import { api } from '../../../services/http';
import type { ResourceRecord } from '../../../models/types';

interface SystemConfig extends ResourceRecord {
  config?: { key?: string; value?: unknown; category?: string };
}

const CATEGORIES = [
  { value: 'platform', label: '平台' },
  { value: 'runtime', label: '运行时' },
  { value: 'integration', label: '集成' },
  { value: 'feature_flag', label: '功能开关' },
];

export function SystemPage(): JSX.Element {
  const [rows, setRows] = useState<SystemConfig[]>([]);
  const [open, setOpen] = useState(false);
  const [form] = Form.useForm();
  const [keyword, setKeyword] = useState('');

  const load = async (): Promise<void> => {
    try {
      const res = await api.list<SystemConfig>('/system/configs', 1, 100);
      const list = keyword ? res.items.filter((r) => `${r.name}${r.code}${r.config?.key}`.includes(keyword)) : res.items;
      setRows(list);
    } catch (e) { message.error(e instanceof Error ? e.message : '加载失败'); }
  };
  useEffect(() => { void load(); }, []);

  const submit = async (): Promise<void> => {
    try {
      const v = await form.validateFields();
      let value: unknown = v.value;
      try { value = JSON.parse(v.value); } catch { /* keep string */ }
      await api.create('/system/configs', { name: v.name, code: v.code, description: v.description || '', enabled: v.enabled !== false,
        config: { key: v.code, value, category: v.category } });
      setOpen(false); form.resetFields(); await load(); message.success('已保存配置');
    } catch (e) {
      if ((e as { errorFields?: unknown }).errorFields) return;
      message.error(e instanceof Error ? e.message : '保存失败');
    }
  };

  const columns: ColumnsType<SystemConfig> = [
    { title: '配置项', dataIndex: 'name', render: (v, r) => <Space direction="vertical" size={0}><strong>{v}</strong><Typography.Text code>{r.code || r.config?.key}</Typography.Text></Space> },
    { title: '分类', width: 130, render: (_, r) => <Tag color="blue">{r.config?.category || 'platform'}</Tag> },
    { title: '值', render: (_, r) => <Typography.Text code style={{ maxWidth: 320 }} ellipsis>{JSON.stringify(r.config?.value)}</Typography.Text> },
    { title: '描述', dataIndex: 'description', ellipsis: true },
    { title: '状态', dataIndex: 'enabled', width: 100, render: (v) => <Tag color={v ? 'success' : 'default'}>{v ? '启用' : '停用'}</Tag> },
  ];

  const byCat = rows.reduce<Record<string, number>>((acc, r) => { const c = r.config?.category || 'platform'; acc[c] = (acc[c] || 0) + 1; return acc; }, {});

  return (
    <div className="enterprise-page">
      <section className="resource-hero">
        <div>
          <Space wrap><Tag color="processing">平台配置</Tag><Tag>设置</Tag></Space>
          <Typography.Title level={1}>系统</Typography.Title>
          <Typography.Paragraph>维护平台级开关、运行参数、集成配置和功能开关。</Typography.Paragraph>
        </div>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => void load()}>刷新</Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setOpen(true)}>新建配置</Button>
        </Space>
      </section>

      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        {CATEGORIES.map((c) => (
          <Col xs={12} md={6} key={c.value}>
            <Card bordered={false} className="enterprise-card">
              <Statistic title={c.label} value={byCat[c.value] || 0} prefix={c.value === 'runtime' ? <CloudServerOutlined /> : c.value === 'integration' ? <DatabaseOutlined /> : <SettingOutlined />} />
            </Card>
          </Col>
        ))}
      </Row>

      <Card bordered={false} className="enterprise-card">
        <Space style={{ marginBottom: 12 }}>
          <Input.Search placeholder="搜索 key" value={keyword} onChange={(e) => setKeyword(e.target.value)} onSearch={() => void load()} style={{ width: 320 }} />
        </Space>
        <Table<SystemConfig> rowKey="id" dataSource={rows} columns={columns} pagination={{ pageSize: 20 }} />
      </Card>

      <Drawer open={open} onClose={() => setOpen(false)} width={520} title="新建配置" extra={<Button type="primary" onClick={() => void submit()}>保存</Button>}>
        <Form form={form} layout="vertical" initialValues={{ enabled: true, category: 'platform' }}>
          <Form.Item name="name" label="名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="code" label="Key" rules={[{ required: true }]}><Input placeholder="例：platform.maintenance_mode" /></Form.Item>
          <Form.Item name="category" label="分类"><Select options={CATEGORIES} /></Form.Item>
          <Form.Item name="value" label="值（JSON 或字符串）" rules={[{ required: true }]}><Input.TextArea rows={5} placeholder='true / 100 / "hello" / {"foo":"bar"}' /></Form.Item>
          <Form.Item name="description" label="描述"><Input.TextArea rows={2} /></Form.Item>
          <Form.Item name="enabled" label="启用" valuePropName="checked"><Switch /></Form.Item>
        </Form>
      </Drawer>
    </div>
  );
}
