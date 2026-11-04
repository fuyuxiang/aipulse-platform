import { useAccess } from "../../../access/context";
import { checkedFetch } from '../../../services/http';
import { Button, Card, Drawer, Form, Input, message, Select, Space, Table, Tag, Typography } from 'antd';
import React, { useEffect, useState } from 'react';
import { api, getToken } from '../../../services/http';
import type { ResourceRecord } from '../../../models/types';

interface UserRow { id: string; username: string; display_name: string; email: string; is_active: boolean }

export function UsersPage(): JSX.Element {
  const { can } = useAccess();
  const [rows, setRows] = useState<UserRow[]>([]);
  const [roles, setRoles] = useState<ResourceRecord[]>([]);
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState<UserRow | null>(null);
  const [keyword, setKeyword] = useState('');
  const [saving, setSaving] = useState(false);
  const [form] = Form.useForm();
  const load = async (): Promise<void> => {
    try {
      const users = await api.list<UserRow>('/users', 1, 200);
      setRows(users.items);
      setRoles((await api.list('/roles', 1, 200)).items);
    } catch (e) { message.error(e instanceof Error ? e.message : '加载失败'); }
  };
  useEffect(() => { void load(); }, []);
  const edit = async (user: UserRow): Promise<void> => {
    try {
      const result = await api.get<{ role_ids: string[] }>(`/users/${user.id}/roles`);
      setEditing(user); form.setFieldsValue({ role_ids: result.role_ids }); setOpen(true);
    } catch (e) { message.error(e instanceof Error ? e.message : '加载角色失败'); }
  };
  const submit = async (): Promise<void> => {
    try {
      const values = await form.validateFields(); setSaving(true);
      let user = editing;
      if (!user) user = await api.post<UserRow>('/users', { username: values.username, password: values.password, display_name: values.display_name || '', email: values.email || '' });
      // Keep the created user in the drawer if assigning roles fails, so retry never creates a duplicate.
      setEditing(user);
      const response = await checkedFetch(`/api/v1/users/${user.id}/roles`, { method: 'PUT', headers: { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' }, body: JSON.stringify(values.role_ids || []) });
      if (!response.ok) throw new Error(await response.text());
      setOpen(false); setEditing(null); form.resetFields(); await load(); message.success('用户与角色已保存');
    } catch (e) {
      if (!(e as { errorFields?: unknown }).errorFields) message.error(e instanceof Error ? e.message : '保存失败');
    } finally { setSaving(false); }
  };
  return <div className="enterprise-page">
    <section className="resource-hero"><div><Typography.Title level={1}>用户</Typography.Title><Typography.Paragraph>创建登录账号，分配或撤销实际生效的租户角色。</Typography.Paragraph></div>
      <Button type="primary" disabled={!can("users:write")} onClick={() => { setEditing(null); form.resetFields(); setOpen(true); }}>新建用户</Button></section>
    <Card><Input.Search placeholder="搜索用户名 / 邮箱" value={keyword} onChange={e => setKeyword(e.target.value)} />
      <Table<UserRow> rowKey="id" dataSource={rows.filter(r => `${r.username} ${r.display_name} ${r.email}`.includes(keyword))} columns={[
        { title: '登录账号', dataIndex: 'username' }, { title: '显示名称', dataIndex: 'display_name' }, { title: '邮箱', dataIndex: 'email' },
        { title: '状态', render: (_, r) => <Tag>{r.is_active ? '启用' : '停用'}</Tag> },
        { title: '操作', render: (_, r) => <Button onClick={() => void edit(r)}>分配角色</Button> }
      ]} />
    </Card>
    <Drawer open={open} onClose={() => setOpen(false)} width={480} title={editing ? `分配角色：${editing.username}` : '新建用户'} extra={<Button type="primary" disabled={!can("users:write")} loading={saving} onClick={() => void submit()}>保存</Button>}>
      <Form form={form} layout="vertical">
        {!editing && <><Form.Item name="username" label="登录账号" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="password" label="初始密码" rules={[{ required: true }, { min: 8 }]}><Input.Password autoComplete="new-password" /></Form.Item>
          <Form.Item name="display_name" label="显示名称"><Input /></Form.Item><Form.Item name="email" label="邮箱"><Input type="email" /></Form.Item></>}
        <Form.Item name="role_ids" label="角色"><Select mode="multiple" showSearch optionFilterProp="label" options={roles.map(r => ({ value: r.id, label: r.name }))} /></Form.Item>
        <Space><Typography.Text type="secondary">移除全部角色后，该用户的受保护接口访问权限将立即失效。</Typography.Text></Space>
      </Form>
    </Drawer>
  </div>;
}
