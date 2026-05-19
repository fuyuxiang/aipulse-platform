import { Steps, Form, Input, Select, Button, Card, Space, message } from 'antd';
import React, { useState } from 'react';
import { api } from '../services/http';

const steps = ['基础信息', '模型配置', '提示词', '工具绑定', '知识库', '记忆策略', '安全策略'];

interface AgentWizardProps {
  onCreated?: () => void;
}

export function AgentWizard({ onCreated }: AgentWizardProps): JSX.Element {
  const [current, setCurrent] = useState(0);
  const [saving, setSaving] = useState(false);
  const [form] = Form.useForm();

  const submit = async (): Promise<void> => {
    setSaving(true);
    try {
      const values = await form.validateFields();
      const config = {
        model_id: values.model_id || '',
        model_type: values.model_type,
        model_config: values.model_config ? JSON.parse(values.model_config) : {},
        system_prompt: values.system_prompt || '',
        tool_policy: values.tool_policy ? JSON.parse(values.tool_policy) : {},
        memory_policy: values.memory_policy ? JSON.parse(values.memory_policy) : { enabled: true },
        guardrail_policy_ids: values.guardrail_policy_ids ? String(values.guardrail_policy_ids).split(',').map((item) => item.trim()).filter(Boolean) : [],
        knowledge_base_ids: values.knowledge_base_ids ? String(values.knowledge_base_ids).split(',').map((item) => item.trim()).filter(Boolean) : [],
      };
      await api.create('/agents', {
        name: values.name,
        code: values.code || '',
        description: values.description || '',
        model_type: values.model_type,
        config,
        spec: {
          system_prompt: values.system_prompt || '',
          tool_ids: values.tool_ids ? String(values.tool_ids).split(',').map((item) => item.trim()).filter(Boolean) : [],
          created_from: 'agent_wizard',
        },
      });
      message.success('Agent 已创建');
      form.resetFields();
      setCurrent(0);
      onCreated?.();
    } catch (error) {
      message.error(error instanceof Error ? error.message : 'Agent 创建失败');
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card className="mb-5" title="Agent 创建向导">
      <Steps current={current} items={steps.map((title) => ({ title }))} />
      <Form form={form} layout="vertical" className="mt-5" initialValues={{ model_type: 'chat_llm', memory_policy: '{\"enabled\":true}' }}>
        {current === 0 ? (
          <>
            <Form.Item name="name" label="Agent 名称" rules={[{ required: true, message: '请输入 Agent 名称' }]}><Input /></Form.Item>
            <Form.Item name="code" label="编码"><Input /></Form.Item>
            <Form.Item name="description" label="描述"><Input.TextArea rows={3} /></Form.Item>
          </>
        ) : null}
        {current === 1 ? (
          <>
            <Form.Item name="model_type" label="模型类型" rules={[{ required: true }]}>
              <Select options={['chat_llm', 'reasoning_llm', 'vision_language', 'embedding', 'rerank'].map((value) => ({ value, label: value }))} />
            </Form.Item>
            <Form.Item name="model_id" label="模型资源 ID"><Input /></Form.Item>
            <Form.Item name="model_config" label="运行时模型配置 JSON"><Input.TextArea rows={4} /></Form.Item>
          </>
        ) : null}
        {current === 2 ? <Form.Item name="system_prompt" label="系统提示词"><Input.TextArea rows={5} /></Form.Item> : null}
        {current === 3 ? <Form.Item name="tool_ids" label="工具 ID，多个用逗号分隔"><Input.TextArea rows={3} /></Form.Item> : null}
        {current === 4 ? <Form.Item name="knowledge_base_ids" label="知识库 ID，多个用逗号分隔"><Input.TextArea rows={3} /></Form.Item> : null}
        {current === 5 ? <Form.Item name="memory_policy" label="记忆策略 JSON"><Input.TextArea rows={4} /></Form.Item> : null}
        {current === 6 ? (
          <>
            <Form.Item name="guardrail_policy_ids" label="护栏策略 ID，多个用逗号分隔"><Input /></Form.Item>
            <Form.Item name="tool_policy" label="工具权限策略 JSON"><Input.TextArea rows={4} /></Form.Item>
          </>
        ) : null}
      </Form>
      <Space>
        <Button disabled={current === 0 || saving} onClick={() => setCurrent((value) => value - 1)}>上一步</Button>
        {current < steps.length - 1 ? (
          <Button type="primary" disabled={saving} onClick={() => setCurrent((value) => value + 1)}>下一步</Button>
        ) : (
          <Button type="primary" loading={saving} onClick={() => void submit()}>保存 Agent</Button>
        )}
      </Space>
    </Card>
  );
}
