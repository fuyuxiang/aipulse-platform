import { Steps, Form, Input, Select, Button, Card } from 'antd';
import React, { useState } from 'react';

const steps = ['基础信息', '模型配置', '提示词', '工具绑定', '知识库', '记忆策略', '安全策略'];

export function AgentWizard(): JSX.Element {
  const [current, setCurrent] = useState(0);
  return (
    <Card className="mb-5" title="Agent 创建向导">
      <Steps current={current} items={steps.map((title) => ({ title }))} />
      <Form layout="vertical" className="mt-5">
        {current === 0 ? <Form.Item label="Agent 名称"><Input /></Form.Item> : null}
        {current === 1 ? <Form.Item label="模型类型"><Select options={['chat_llm', 'reasoning_llm', 'vision_language', 'embedding', 'rerank'].map((value) => ({ value, label: value }))} /></Form.Item> : null}
        {current === 2 ? <Form.Item label="系统提示词"><Input.TextArea rows={5} /></Form.Item> : null}
        {current > 2 ? <Form.Item label={steps[current]}><Input.TextArea rows={4} /></Form.Item> : null}
      </Form>
      <Button disabled={current === 0} onClick={() => setCurrent((value) => value - 1)}>上一步</Button>
      <Button type="primary" className="ml-2" disabled={current === steps.length - 1} onClick={() => setCurrent((value) => value + 1)}>下一步</Button>
    </Card>
  );
}

