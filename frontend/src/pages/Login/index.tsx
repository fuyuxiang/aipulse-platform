import {
  ApartmentOutlined,
  CloudServerOutlined,
  DatabaseOutlined,
  LockOutlined,
  SafetyCertificateOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { Button, Card, Col, Form, Input, Row, Space, Tag, Typography, message } from 'antd';
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { api, setTokens } from '../../services/http';

export function LoginPage(): JSX.Element {
  const navigate = useNavigate();
  const submit = async (values: { tenant: string; username: string; password: string }): Promise<void> => {
    try {
      const tokens = await api.login(values);
      setTokens(tokens);
      localStorage.setItem('aipulse_tenant', values.tenant);
      localStorage.setItem('aipulse_username', values.username);
      navigate('/dashboard');
    } catch (error) {
      message.error(error instanceof Error ? error.message : '登录失败');
    }
  };
  return (
    <div className="login-shell">
      <div className="login-product-panel">
        <div className="login-brand-row">
          <div className="app-logo-mark app-logo-mark-large">AP</div>
          <div>
            <Typography.Title level={2}>AIPulse AgentOS</Typography.Title>
            <Typography.Text>企业级智能体研发、运行与治理平台</Typography.Text>
          </div>
        </div>
        <Row gutter={[12, 12]} className="login-capability-grid">
          {[
            { icon: <ApartmentOutlined />, title: '多智能体协作', text: '团队拓扑、委派策略、运行追踪' },
            { icon: <CloudServerOutlined />, title: '生产运行面', text: '调度、会话、发布、市场分发' },
            { icon: <SafetyCertificateOutlined />, title: '安全治理', text: '护栏、权限、审计、合规矩阵' },
            { icon: <DatabaseOutlined />, title: '知识与记忆', text: 'RAG、长期记忆、上下文压缩' },
          ].map((item) => (
            <Col span={12} key={item.title}>
              <div className="login-capability-card">
                <span>{item.icon}</span>
                <strong>{item.title}</strong>
                <small>{item.text}</small>
              </div>
            </Col>
          ))}
        </Row>
      </div>
      <Card
        className="login-card"
        title={
          <Space direction="vertical" size={2}>
            <Typography.Text strong>登录控制台</Typography.Text>
            <Typography.Text type="secondary">智能体基础设施平台</Typography.Text>
          </Space>
        }
        extra={<Tag color="processing">企业版</Tag>}
      >
        <Form layout="vertical" initialValues={{ tenant: 'default', username: 'admin' }} onFinish={(values) => void submit(values)}>
          <Form.Item name="tenant" label="租户" rules={[{ required: true }]}>
            <Input placeholder="default" />
          </Form.Item>
          <Form.Item name="username" label="用户名" rules={[{ required: true }]}>
            <Input prefix={<UserOutlined />} placeholder="admin" />
          </Form.Item>
          <Form.Item name="password" label="密码" rules={[{ required: true }]}>
            <Input.Password prefix={<LockOutlined />} placeholder="请输入密码" />
          </Form.Item>
          <Button type="primary" htmlType="submit" block size="large">登录平台</Button>
        </Form>
      </Card>
    </div>
  );
}
