import { Card, Input, Space, Table, Tag, Typography, Button, Drawer, Form, Switch, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { PlusOutlined, ReloadOutlined, ApartmentOutlined } from '@ant-design/icons';
import React, { useEffect, useState } from 'react';
import { api } from '../../../services/http';
import type { ResourceRecord } from '../../../models/types';

export function OrganizationsPage(): JSX.Element {
  const [rows, setRows] = useState<ResourceRecord[]>([]);
  const [open, setOpen] = useState(false);
  const [form] = Form.useForm();

  const load = async (): Promise<void> => {
    try { setRows((await api.list('/orgs', 1, 100)).items); }
    catch (e) { message.error(e instanceof Error ? e.message : '加载失败'); }
  };
  useEffect(() => { void load(); }, []);

  const submit = async (): Promise<void> => {
    try {
      const v = await form.validateFields();
      await api.create('/orgs', { name: v.name, code: v.code, description: v.description || '', enabled: v.enabled !== false,
        parent_id: v.parent_id || '' });
      setOpen(false); form.resetFields(); await load(); message.success('已创建组织');
    } catch (e) {
      if ((e as { errorFields?: unknown }).errorFields) return;
      message.error(e instanceof Error ? e.message : '保存失败');
    }
  };

  const columns: ColumnsType<ResourceRecord> = [
    { title: '组织名称', dataIndex: 'name', render: (v, r) => <Space><ApartmentOutlined /><strong>{v}</strong><Tag>{r.code}</Tag></Space> },
    { title: '父组织', dataIndex: 'parent_id', width: 200, render: (v) => v ? <Tag>{v}</Tag> : <Typography.Text type="secondary">顶级</Typography.Text> },
    { title: '描述', dataIndex: 'description', ellipsis: true },
    { title: '状态', dataIndex: 'enabled', width: 100, render: (v) => <Tag color={v ? 'success' : 'default'}>{v ? '启用' : '停用'}</Tag> },
    { title: '更新时间', dataIndex: 'updated_at', width: 190 },
  ];

  return (
    <div className="enterprise-page">
      <section className="resource-hero">
        <div>
          <Space wrap><Tag color="processing">组织治理</Tag><Tag>设置</Tag></Space>
          <Typography.Title level={1}>组织</Typography.Title>
          <Typography.Paragraph>维护租户内的组织树、部门归属与资源可见性边界。</Typography.Paragraph>
        </div>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => void load()}>刷新</Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setOpen(true)}>新建组织</Button>
        </Space>
      </section>

      <Card bordered={false} className="enterprise-card">
        <Table<ResourceRecord> rowKey="id" dataSource={rows} columns={columns} pagination={{ pageSize: 20 }} />
      </Card>

      <Drawer open={open} onClose={() => setOpen(false)} width={460} title="新建组织" extra={<Button type="primary" onClick={() => void submit()}>保存</Button>}>
        <Form form={form} layout="vertical" initialValues={{ enabled: true }}>
          <Form.Item name="name" label="组织名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="code" label="编码" rules={[{ required: true }]}><Input placeholder="org_xxx" /></Form.Item>
          <Form.Item name="parent_id" label="父组织编码"><Input placeholder="留空表示顶级" /></Form.Item>
          <Form.Item name="description" label="描述"><Input.TextArea rows={2} /></Form.Item>
          <Form.Item name="enabled" label="启用" valuePropName="checked"><Switch /></Form.Item>
        </Form>
      </Drawer>
    </div>
  );
}
