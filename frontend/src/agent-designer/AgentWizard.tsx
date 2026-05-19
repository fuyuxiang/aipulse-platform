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
      message.success('智能体创建成功');
      form.resetFields();
      setCurrent(0);
      onCreated?.();
    } catch (error) {
      message.error(error instanceof Error ? error.message : '创建失败，请检查配置');
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card className="mb-5" title="智能体创建向导">
      <Steps current={current} items={steps.map((title) => ({ title }))} />
      <Form form={form} layout="vertical" className="mt-5" initialValues={{ model_type: 'chat_llm', memory_policy: '{\"enabled\":true}' }}>
        {current === 0 ? (
          <>
            <Form.Item name="name" label="智能体名称" rules={[{ required: true, message: '请输入智能体名称' }]}><Input placeholder="例如：客服助手" /></Form.Item>
            <Form.Item name="code" label="编码"><Input placeholder="唯一标识，留空自动生成" /></Form.Item>
            <Form.Item name="description" label="描述"><Input.TextArea rows={3} placeholder="描述该智能体的用途和能力" /></Form.Item>
          </>
        ) : null}
        {current === 1 ? (
          <>
            <Form.Item name="model_type" label="模型类型" rules={[{ required: true }]}>
              <Select options={[
                { value: 'chat_llm', label: '对话模型' },
                { value: 'reasoning_llm', label: '推理模型' },
                { value: 'vision_language', label: '多模态模型' },
                { value: 'embedding', label: '向量模型' },
                { value: 'rerank', label: '重排序模型' },
              ]} />
            </Form.Item>
            <Form.Item name="model_id" label="模型"><Input placeholder="选择已注册的模型资源" /></Form.Item>
            <Form.Item name="model_config" label="模型运行参数"><Input.TextArea rows={4} placeholder='{"temperature": 0.7, "max_tokens": 4096}' /></Form.Item>
          </>
        ) : null}
        {current === 2 ? <Form.Item name="system_prompt" label="系统提示词"><Input.TextArea rows={5} placeholder="定义智能体的角色、能力和行为规范" /></Form.Item> : null}
        {current === 3 ? <Form.Item name="tool_ids" label="绑定工具"><Input.TextArea rows={3} placeholder="输入工具 ID，多个以逗号分隔" /></Form.Item> : null}
        {current === 4 ? <Form.Item name="knowledge_base_ids" label="关联知识库"><Input.TextArea rows={3} placeholder="输入知识库 ID，多个以逗号分隔" /></Form.Item> : null}
        {current === 5 ? <Form.Item name="memory_policy" label="记忆策略配置"><Input.TextArea rows={4} placeholder='{"enabled": true, "scope": "session"}' /></Form.Item> : null}
        {current === 6 ? (
          <>
            <Form.Item name="guardrail_policy_ids" label="关联安全护栏"><Input placeholder="输入护栏策略 ID，多个以逗号分隔" /></Form.Item>
            <Form.Item name="tool_policy" label="工具权限配置"><Input.TextArea rows={4} placeholder='{"allowed_tools": [], "require_approval": false}' /></Form.Item>
          </>
        ) : null}
      </Form>
      <Space>
        <Button disabled={current === 0 || saving} onClick={() => setCurrent((value) => value - 1)}>上一步</Button>
        {current < steps.length - 1 ? (
          <Button type="primary" disabled={saving} onClick={() => setCurrent((value) => value + 1)}>下一步</Button>
        ) : (
          <Button type="primary" loading={saving} onClick={() => void submit()}>创建智能体</Button>
        )}
      </Space>
    </Card>
  );
}
