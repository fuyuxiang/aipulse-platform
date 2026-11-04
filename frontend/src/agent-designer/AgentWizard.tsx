import {
  Alert,
  Button,
  Card,
  Collapse,
  Form,
  Input,
  message,
  Select,
  Space,
  Spin,
  Steps,
  Switch,
} from "antd";
import React, { useEffect, useState } from "react";
import { useAccess } from "../access/context";
import type { ResourceRecord } from "../models/types";
import { api } from "../services/http";
import { useUnsavedChanges } from "../components/useUnsavedChanges";
import { parseObject } from "../components/PageHeader";

const steps = [
  "基础信息",
  "模型配置",
  "提示词",
  "工具绑定",
  "知识库",
  "记忆策略",
  "安全策略",
];
export function AgentWizard({
  onCreated,
  agentId,
}: {
  onCreated?: () => void;
  agentId?: string;
}): JSX.Element {
  const { can } = useAccess();
  const [current, setCurrent] = useState(0);
  const [dirty, setDirty] = useState(false);
  const markSaved = useUnsavedChanges(dirty);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [existing, setExisting] = useState<ResourceRecord | null>(null);
  const [form] = Form.useForm();
  const [resources, setResources] = useState<Record<string, ResourceRecord[]>>(
    {},
  );
  const modelType = Form.useWatch("model_type", form) || "chat_llm";
  useEffect(() => {
    let active = true;
    const load = async () => {
      setLoading(true);
      setError("");
      try {
        const sources = [
          { key: "models", api: "/models", permission: "models:read" },
          { key: "tools", api: "/tools", permission: "tools:read" },
          {
            key: "knowledge-bases",
            api: "/knowledge-bases",
            permission: "knowledge:read",
          },
          {
            key: "guardrails",
            api: "/guardrails/policies",
            permission: "security:read",
          },
        ].filter((source) => can(source.permission));
        const results = await Promise.all(
          sources.map(async (source) => ({
            key: source.key,
            result: await api.list(source.api, 1, 200),
          })),
        );
        if (!active) return;
        setResources(
          Object.fromEntries(
            results.map((result) => [result.key, result.result.items]),
          ),
        );
        if (agentId) {
          const row = await api.get<ResourceRecord>(`/agents/${agentId}`);
          if (!active) return;
          setExisting(row);
          const config = { ...row.spec, ...row.config };
          form.setFieldsValue({
            name: row.name,
            code: row.code,
            description: row.description,
            model_type: config.model_type || row.model_type || "chat_llm",
            model_id: config.model_id,
            system_prompt: config.system_prompt,
            tool_ids: config.tool_ids || [],
            knowledge_base_ids: config.knowledge_base_ids || [],
            model_config: JSON.stringify(config.model_config || {}, null, 2),
            memory_enabled: !!(config.memory_policy as any)?.enabled,
            require_approval: !!(config.tool_policy as any)?.require_approval,
            guardrail_policy_ids: config.guardrail_policy_ids || [],
          });
        }
      } catch (reason) {
        if (active) setError((reason as Error).message);
      } finally {
        if (active) setLoading(false);
      }
    };
    void load();
    return () => {
      active = false;
    };
  }, [agentId]);
  const options = (key: string) =>
    (resources[key] || [])
      .filter((row) => row.enabled)
      .filter((row) => key !== "models" || row.model_type === modelType)
      .map((row) => ({
        value: row.id,
        label:
          row.name +
          (row.provider_type === "echo_agent_native" ? "（本地演示）" : ""),
      }));
  const submit = async () => {
    try {
      await form.validateFields();
      const values = form.getFieldsValue(true);
      if (!values.name?.trim() || !values.model_id)
        throw new Error("请填写名称并选择模型");
      const config = {
        ...existing?.config,
        model_id: values.model_id,
        model_type: values.model_type,
        tool_ids: values.tool_ids || [],
        model_config: parseObject(values.model_config),
        system_prompt: values.system_prompt || "",
        tool_policy: {
          ...(existing?.config?.tool_policy as object),
          require_approval: !!values.require_approval,
        },
        memory_policy: {
          ...(existing?.config?.memory_policy as object),
          enabled: !!values.memory_enabled,
        },
        guardrail_policy_ids: values.guardrail_policy_ids || [],
        knowledge_base_ids: values.knowledge_base_ids || [],
      };
      setSaving(true);
      const payload = {
        name: values.name.trim(),
        code: values.code || "",
        description: values.description || "",
        model_type: values.model_type,
        config,
        spec: {
          ...existing?.spec,
          system_prompt: values.system_prompt || "",
          tool_ids: values.tool_ids || [],
          created_from: "agent_wizard",
        },
      };
      if (agentId) await api.update("/agents", agentId, payload);
      else await api.create("/agents", payload);
      markSaved();
      setDirty(false);
      message.success(agentId ? "智能体已保存" : "智能体创建成功");
      onCreated?.();
    } catch (reason) {
      if (reason instanceof Error) message.error(reason.message);
    } finally {
      setSaving(false);
    }
  };
  return (
    <Card className="enterprise-card agent-wizard" bordered={false}>
      <Steps
        current={current}
        size="small"
        responsive
        items={steps.map((title) => ({ title }))}
      />
      {loading && (
        <div className="p-5">
          <Spin /> 正在加载配置…
        </div>
      )}
      {error && (
        <Alert
          className="mt-4"
          type="error"
          showIcon
          message="配置加载失败"
          description={error}
        />
      )}
      <Form
        form={form}
        layout="vertical"
        className="wizard-form"
        onValuesChange={() => setDirty(true)}
        preserve
        initialValues={{
          model_type: "chat_llm",
          memory_enabled: false,
          require_approval: false,
        }}
        disabled={loading || saving || !!error}
      >
        <div className="wizard-step-heading">
          <h3>{steps[current]}</h3>
          <p>
            {
              [
                "为智能体设置易于识别的名称和用途。",
                "选择用于生成回答的模型。本地演示模型用于验证流程。",
                "描述智能体的职责、回答要求和边界。",
                "选择智能体在执行任务时可使用的工具。此步骤可跳过。",
                "选择回答时可检索的知识库。此步骤可跳过。",
                "选择是否启用记忆，后续可在编辑时调整。",
                "按需关联安全护栏与工具审批策略。",
              ][current]
            }
          </p>
        </div>
        {current === 0 && (
          <>
            <Form.Item
              name="name"
              label="智能体名称"
              rules={[
                {
                  required: true,
                  whitespace: true,
                  message: "请输入智能体名称",
                },
              ]}
            >
              <Input placeholder="例如：客户服务助手" maxLength={120} />
            </Form.Item>
            <Form.Item name="description" label="描述">
              <Input.TextArea rows={3} placeholder="说明适合处理哪些任务" />
            </Form.Item>
            <Form.Item name="code" label="编码">
              <Input placeholder="留空自动生成" />
            </Form.Item>
          </>
        )}
        {current === 1 && (
          <>
            <Form.Item
              name="model_type"
              label="模型类型"
              rules={[{ required: true }]}
            >
              <Select
                onChange={() => form.setFieldValue("model_id", undefined)}
                options={[
                  { value: "chat_llm", label: "对话模型" },
                  { value: "reasoning_llm", label: "推理模型" },
                  { value: "vision_language", label: "多模态模型" },
                ]}
              />
            </Form.Item>
            <Form.Item
              name="model_id"
              label="模型"
              rules={[{ required: true, message: "请选择模型" }]}
            >
              <Select
                showSearch
                optionFilterProp="label"
                placeholder="选择可用模型"
                options={options("models")}
                notFoundContent="没有匹配的可用模型，请联系管理员接入"
              />
            </Form.Item>
            <Collapse
              items={[
                {
                  key: "parameters",
                  label: "高级运行参数",
                  forceRender: true,
                  children: (
                    <Form.Item
                      name="model_config"
                      label="模型运行参数"
                      rules={[
                        {
                          validator: async (_, value) => {
                            parseObject(value);
                          },
                        },
                      ]}
                    >
                      <Input.TextArea
                        rows={4}
                        placeholder='{"temperature": 0.7, "max_tokens": 4096}'
                        className="code-input"
                      />
                    </Form.Item>
                  ),
                },
              ]}
            />
          </>
        )}
        {current === 2 && (
          <Form.Item name="system_prompt" label="系统提示词">
            <Input.TextArea rows={9} placeholder="你是一名…请遵循…" />
          </Form.Item>
        )}
        {current === 3 && (
          <Form.Item name="tool_ids" label="绑定工具">
            <Select
              mode="multiple"
              showSearch
              optionFilterProp="label"
              placeholder="不绑定工具也可以开始对话"
              options={options("tools")}
            />
          </Form.Item>
        )}
        {current === 4 && (
          <Form.Item name="knowledge_base_ids" label="关联知识库">
            <Select
              mode="multiple"
              showSearch
              optionFilterProp="label"
              placeholder="选择已配置检索模型的知识库"
              options={options("knowledge-bases")}
            />
          </Form.Item>
        )}
        {current === 5 && (
          <>
            <Form.Item
              name="memory_enabled"
              label="启用记忆"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
            <Alert
              type="info"
              showIcon
              message="启用后将按智能体的记忆策略使用历史上下文；关闭时不主动读取长期记忆。"
            />
          </>
        )}
        {current === 6 && (
          <>
            <Form.Item name="guardrail_policy_ids" label="关联安全护栏">
              <Select
                mode="multiple"
                showSearch
                optionFilterProp="label"
                options={options("guardrails")}
                placeholder="可选"
              />
            </Form.Item>
            <Form.Item
              name="require_approval"
              label="工具调用需要审批"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
            <Alert
              type="info"
              showIcon
              message="需要审批的调用会进入等待状态。当前对话中的工具审批暂不支持自动续跑，请在配置策略时考虑这一限制。"
            />
          </>
        )}
      </Form>
      <div className="wizard-footer">
        <span className="secondary-text">
          步骤 {current + 1} / {steps.length}
        </span>
        <Space>
          <Button
            disabled={current === 0 || saving || loading}
            onClick={() => setCurrent((value) => value - 1)}
          >
            上一步
          </Button>
          {current < steps.length - 1 ? (
            <Button
              type="primary"
              disabled={saving || loading || !!error}
              onClick={() =>
                void form
                  .validateFields()
                  .then(() => setCurrent((value) => value + 1))
                  .catch(() => undefined)
              }
            >
              下一步
            </Button>
          ) : (
            <Button
              type="primary"
              loading={saving}
              disabled={!!error}
              onClick={() => void submit()}
            >
              {agentId ? "保存智能体" : "创建智能体"}
            </Button>
          )}
        </Space>
      </div>
    </Card>
  );
}
