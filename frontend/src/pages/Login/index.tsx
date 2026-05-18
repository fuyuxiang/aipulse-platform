import { LockOutlined, UserOutlined } from '@ant-design/icons';
import { Button, Card, Form, Input, message } from 'antd';
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { api, setTokens } from '../../services/http';

export function LoginPage(): JSX.Element {
  const navigate = useNavigate();
  const submit = async (values: { tenant: string; username: string; password: string }): Promise<void> => {
    try {
      const tokens = await api.login(values);
      setTokens(tokens);
      navigate('/dashboard');
    } catch (error) {
      message.error(error instanceof Error ? error.message : '登录失败');
    }
  };
  return (
    <div className="flex h-full items-center justify-center bg-slate-100">
      <Card title="AIPulse" className="w-[380px]">
        <Form layout="vertical" initialValues={{ tenant: 'default', username: 'admin' }} onFinish={(values) => void submit(values)}>
          <Form.Item name="tenant" label="租户" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="username" label="用户名" rules={[{ required: true }]}><Input prefix={<UserOutlined />} /></Form.Item>
          <Form.Item name="password" label="密码" rules={[{ required: true }]}><Input.Password prefix={<LockOutlined />} /></Form.Item>
          <Button type="primary" htmlType="submit" block>登录</Button>
        </Form>
      </Card>
    </div>
  );
}

