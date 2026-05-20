import { Card, Input, Space, Table, Tag, Typography, Button, Drawer, Form, Select, Switch, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { PlusOutlined, ReloadOutlined, UserOutlined } from '@ant-design/icons';
import React, { useEffect, useState } from 'react';
import { api } from '../../../services/http';
import type { ResourceRecord } from '../../../models/types';

interface UserRow extends ResourceRecord {
  config?: { email?: string; phone?: string; role?: string; org_id?: string };
}

export function UsersPage(): JSX.Element {
  const [rows, setRows] = useState<UserRow[]>([]);
  const [total, setTotal] = useState(0);
  const [open, setOpen] = useState(false);
  const [form] = Form.useForm();
  const [keyword, setKeyword] = useState('');

  const load = async (page = 1): Promise<void> => {
    try {
      const res = await api.list<UserRow>('/users', page, 50);
      const list = keyword ? res.items.filter((r) => `${r.name}${r.code}${r.config?.email}`.includes(keyword)) : res.items;
      setRows(list);
      setTotal(res.total);
    } catch (e) {
      message.error(e instanceof Error ? e.message : '加载失败');
    }
  };

  useEffect(() => { void load(); }, []);

  const submit = async (): Promise<void> => {
    try {
      const v = await form.validateFields();
      await api.create('/users', { name: v.name, code: v.code, description: v.description || '', enabled: v.enabled !== false,
        config: { email: v.email, phone: v.phone, role: v.role, org_id: v.org_id } });
      setOpen(false); form.resetFields();
      await load(); message.success('已创建用户');
    } catch (e) {
      if ((e as { errorFields?: unknown }).errorFields) return;
      message.error(e instanceof Error ? e.message : '保存失败');
    }
  };

  const columns: ColumnsType<UserRow> = [
    { title: '用户', dataIndex: 'name', render: (v, r) => <Space><UserOutlined /><strong>{v}</strong><Tag>{r.code}</Tag></Space> },
    { title: '邮箱', render: (_, r) => r.config?.email || '-' },
    { title: '角色', render: (_, r) => <Tag color="processing">{r.config?.role || '-'}</Tag> },
    { title: '组织', render: (_, r) => r.config?.org_id || '-' },
    { title: '状态', dataIndex: 'enabled', width: 100, render: (v) => <Tag color={v ? 'success' : 'default'}>{v ? '启用' : '停用'}</Tag> },
    { title: '更新时间', dataIndex: 'updated_at', width: 190 },
  ];

  return (
    <div className="enterprise-page">
      <section className="resource-hero">
        <div>
          <Space wrap><Tag color="processing">身份与访问</Tag><Tag>设置</Tag></Space>
          <Typography.Title level={1}>用户</Typography.Title>
          <Typography.Paragraph>管理租户内的用户、角色绑定和组织归属。</Typography.Paragraph>
        </div>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => void load()}>刷新</Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setOpen(true)}>新建用户</Button>
        </Space>
      </section>

      <Card bordered={false} className="enterprise-card">
        <Space style={{ marginBottom: 12 }}>
          <Input.Search placeholder="搜索用户名 / 邮箱" value={keyword} onChange={(e) => setKeyword(e.target.value)} onSearch={() => void load()} style={{ width: 320 }} />
          <Typography.Text type="secondary">共 {total} 个用户</Typography.Text>
        </Space>
        <Table<UserRow> rowKey="id" dataSource={rows} columns={columns} pagination={{ pageSize: 20, total, onChange: (p) => void load(p) }} />
      </Card>

      <Drawer open={open} onClose={() => setOpen(false)} width={480} title="新建用户" extra={<Button type="primary" onClick={() => void submit()}>保存</Button>}>
        <Form form={form} layout="vertical" initialValues={{ enabled: true }}>
          <Form.Item name="name" label="用户名" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="code" label="登录账号" rules={[{ required: true }]}><Input placeholder="login id" /></Form.Item>
          <Form.Item name="email" label="邮箱"><Input type="email" /></Form.Item>
          <Form.Item name="phone" label="手机"><Input /></Form.Item>
          <Form.Item name="role" label="角色"><Select options={[{ value: 'admin', label: '管理员' }, { value: 'developer', label: '开发者' }, { value: 'operator', label: '运维' }, { value: 'viewer', label: '只读' }]} /></Form.Item>
          <Form.Item name="org_id" label="组织"><Input placeholder="组织编码" /></Form.Item>
          <Form.Item name="description" label="备注"><Input.TextArea rows={2} /></Form.Item>
          <Form.Item name="enabled" label="启用" valuePropName="checked"><Switch /></Form.Item>
        </Form>
      </Drawer>
    </div>
  );
}
