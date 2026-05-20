import { Card, Input, Space, Table, Tag, Typography, Button, Drawer, Form, Select, Switch, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { PlusOutlined, ReloadOutlined, SafetyCertificateOutlined } from '@ant-design/icons';
import React, { useEffect, useState } from 'react';
import { api } from '../../../services/http';
import type { ResourceRecord } from '../../../models/types';

interface RoleRow extends ResourceRecord {
  config?: { permissions?: string[]; scope?: string };
}

const PERMISSIONS = [
  { value: 'agent:read', label: '智能体只读' },
  { value: 'agent:write', label: '智能体编辑' },
  { value: 'workflow:write', label: '工作流编辑' },
  { value: 'tool:write', label: '工具编辑' },
  { value: 'knowledge:write', label: '知识库编辑' },
  { value: 'audit:read', label: '审计查看' },
  { value: 'cost:read', label: '成本查看' },
  { value: 'settings:write', label: '设置编辑' },
];

export function RolesPage(): JSX.Element {
  const [rows, setRows] = useState<RoleRow[]>([]);
  const [open, setOpen] = useState(false);
  const [form] = Form.useForm();

  const load = async (): Promise<void> => {
    try {
      const res = await api.list<RoleRow>('/roles', 1, 50);
      setRows(res.items);
    } catch (e) { message.error(e instanceof Error ? e.message : '加载失败'); }
  };

  useEffect(() => { void load(); }, []);

  const submit = async (): Promise<void> => {
    try {
      const v = await form.validateFields();
      await api.create('/roles', { name: v.name, code: v.code, description: v.description || '', enabled: v.enabled !== false,
        config: { permissions: v.permissions || [], scope: v.scope || 'tenant' } });
      setOpen(false); form.resetFields(); await load(); message.success('已创建角色');
    } catch (e) {
      if ((e as { errorFields?: unknown }).errorFields) return;
      message.error(e instanceof Error ? e.message : '保存失败');
    }
  };

  const columns: ColumnsType<RoleRow> = [
    { title: '角色', dataIndex: 'name', render: (v, r) => <Space><strong>{v}</strong><Tag>{r.code}</Tag></Space> },
    { title: '作用域', render: (_, r) => <Tag color="blue">{r.config?.scope || 'tenant'}</Tag> },
    { title: '权限', render: (_, r) => (r.config?.permissions || []).map((p) => <Tag key={p}>{p}</Tag>) },
    { title: '描述', dataIndex: 'description', ellipsis: true },
    { title: '状态', dataIndex: 'enabled', width: 100, render: (v) => <Tag color={v ? 'success' : 'default'}>{v ? '启用' : '停用'}</Tag> },
  ];

  return (
    <div className="enterprise-page">
      <section className="resource-hero">
        <div>
          <Space wrap><Tag color="processing">访问控制</Tag><Tag>设置</Tag></Space>
          <Typography.Title level={1}>角色</Typography.Title>
          <Typography.Paragraph>定义平台角色与权限组合，支撑细粒度访问控制（RBAC）。</Typography.Paragraph>
        </div>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => void load()}>刷新</Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setOpen(true)}>新建角色</Button>
        </Space>
      </section>

      <Card bordered={false} className="enterprise-card">
        <Table<RoleRow> rowKey="id" dataSource={rows} columns={columns} pagination={{ pageSize: 20 }} />
      </Card>

      <Drawer open={open} onClose={() => setOpen(false)} width={480} title="新建角色" extra={<Button type="primary" icon={<SafetyCertificateOutlined />} onClick={() => void submit()}>保存</Button>}>
        <Form form={form} layout="vertical" initialValues={{ enabled: true, scope: 'tenant' }}>
          <Form.Item name="name" label="角色名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="code" label="编码" rules={[{ required: true }]}><Input placeholder="role_xxx" /></Form.Item>
          <Form.Item name="description" label="描述"><Input.TextArea rows={2} /></Form.Item>
          <Form.Item name="scope" label="作用域"><Select options={[{ value: 'tenant', label: '租户级' }, { value: 'org', label: '组织级' }, { value: 'project', label: '项目级' }]} /></Form.Item>
          <Form.Item name="permissions" label="权限"><Select mode="multiple" options={PERMISSIONS} /></Form.Item>
          <Form.Item name="enabled" label="启用" valuePropName="checked"><Switch /></Form.Item>
        </Form>
      </Drawer>
    </div>
  );
}
