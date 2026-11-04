import { PlayCircleOutlined, UploadOutlined } from "@ant-design/icons";
import {
  Alert,
  Button,
  Collapse,
  Empty,
  Form,
  Input,
  InputNumber,
  Select,
  Space,
  Typography,
  Upload,
  message,
} from "antd";
import React, { useMemo, useState } from "react";
import type { PageConfig, ResourceRecord } from "../models/types";
import { api } from "../services/http";
import { MarkdownContent } from "./MarkdownContent";
import { parseObject, StatusTag } from "./PageHeader";

interface Props {
  page: PageConfig;
  rows: ResourceRecord[];
  onChanged: () => Promise<void>;
}

interface Operation {
  key: string;
  label: string;
  needsResource: boolean;
  buildPath: (resourceId: string) => string;
}

function operationsFor(page: PageConfig): Operation[] {
  if (page.api === "/knowledge-bases") {
    return [
      {
        key: "upload",
        label: "上传文档",
        needsResource: true,
        buildPath: (id) => `/knowledge-bases/${id}/documents`,
      },
      {
        key: "retrieve",
        label: "检索测试",
        needsResource: true,
        buildPath: (id) => `/knowledge-bases/${id}/retrieve`,
      },
      {
        key: "rebuild",
        label: "重建索引",
        needsResource: true,
        buildPath: (id) => `/knowledge-bases/${id}/rebuild-index`,
      },
    ];
  }
  if (page.api === "/agents") {
    return [
      {
        key: "version",
        label: "创建版本",
        needsResource: true,
        buildPath: (id) => `/agents/${id}/versions`,
      },
      {
        key: "release",
        label: "发布",
        needsResource: true,
        buildPath: (id) => `/agents/${id}/release`,
      },
      {
        key: "gray-release",
        label: "灰度发布",
        needsResource: true,
        buildPath: (id) => `/agents/${id}/gray-release`,
      },
      {
        key: "rollback",
        label: "回滚",
        needsResource: true,
        buildPath: (id) => `/agents/${id}/rollback`,
      },
      {
        key: "debug-run",
        label: "试运行",
        needsResource: true,
        buildPath: (id) => `/agents/${id}/debug-run`,
      },
    ];
  }
  if (page.api === "/workflows") {
    return [
      {
        key: "validate",
        label: "校验",
        needsResource: true,
        buildPath: (id) => `/workflows/${id}/validate`,
      },
      {
        key: "run",
        label: "执行",
        needsResource: true,
        buildPath: (id) => `/workflows/${id}/run`,
      },
      {
        key: "publish",
        label: "发布版本",
        needsResource: true,
        buildPath: (id) => `/workflows/${id}/publish`,
      },
    ];
  }
  if (page.api === "/tools") {
    return [
      {
        key: "invoke",
        label: "调用工具",
        needsResource: true,
        buildPath: (id) => `/tools/${id}/invoke`,
      },
    ];
  }
  if (page.api === "/memories") {
    return [
      {
        key: "extract",
        label: "抽取记忆",
        needsResource: false,
        buildPath: () => "/memories/extract",
      },
      {
        key: "search",
        label: "检索记忆",
        needsResource: false,
        buildPath: () => "/memories/search",
      },
    ];
  }
  if (page.api === "/models") {
    return [
      {
        key: "version",
        label: "创建模型版本",
        needsResource: true,
        buildPath: (id) => `/models/${id}/versions`,
      },
      {
        key: "health",
        label: "健康检查",
        needsResource: true,
        buildPath: (id) => `/models/${id}/health-check`,
      },
      {
        key: "reset-circuit",
        label: "重置熔断",
        needsResource: true,
        buildPath: (id) => `/model-circuit-breakers/${id}/reset`,
      },
      {
        key: "test-chat",
        label: "对话验证",
        needsResource: true,
        buildPath: (id) => `/models/${id}/test-chat`,
      },
      {
        key: "test-embedding",
        label: "向量化验证",
        needsResource: true,
        buildPath: (id) => `/models/${id}/test-embedding`,
      },
      {
        key: "test-rerank",
        label: "重排序验证",
        needsResource: true,
        buildPath: (id) => `/models/${id}/test-rerank`,
      },
      {
        key: "test-moderation",
        label: "内容审核验证",
        needsResource: true,
        buildPath: (id) => `/models/${id}/test-moderation`,
      },
    ];
  }
  if (page.api === "/model-credentials") {
    return [
      {
        key: "credential-test",
        label: "凭证验证",
        needsResource: true,
        buildPath: (id) => `/model-credentials/${id}/test`,
      },
    ];
  }
  if (page.api === "/model-routing-policies") {
    return [
      {
        key: "route",
        label: "调度模拟",
        needsResource: false,
        buildPath: () => "/models/route",
      },
      {
        key: "invoke",
        label: "路由调用",
        needsResource: false,
        buildPath: () => "/models/invoke",
      },
    ];
  }
  if (page.api.startsWith("/security/")) {
    return [
      {
        key: "security-check",
        label: "安全检查",
        needsResource: false,
        buildPath: () => "/security/check",
      },
    ];
  }
  if (page.api === "/audit-logs") {
    return [
      {
        key: "audit-export",
        label: "审计导出",
        needsResource: false,
        buildPath: () => "/audit-logs/export",
      },
    ];
  }
  if (page.api === "/evaluation/datasets" || page.api === "/evaluation/runs") {
    return [
      {
        key: "evaluation-run",
        label: "导入回答评分",
        needsResource: false,
        buildPath: () => "/evaluation/runs",
      },
    ];
  }
  return [];
}

function parsePayload(text: string): Record<string, unknown> {
  if (!text.trim()) return {};
  const parsed = JSON.parse(text) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("JSON 必须是对象");
  }
  return parsed as Record<string, unknown>;
}

interface OperationField {
  key: string;
  label: string;
  required?: boolean;
  multiline?: boolean;
  number?: boolean;
}
function fieldsFor(key: string): OperationField[] {
  if (key === "upload")
    return [
      { key: "filename", label: "文件名", required: true },
      { key: "content", label: "文档内容", multiline: true },
    ];
  if (key === "retrieve" || key === "search")
    return [
      { key: "query", label: "检索问题", required: true, multiline: true },
      { key: "top_k", label: "返回数量", number: true },
    ];
  if (key === "run" || key === "debug-run" || key === "test-chat")
    return [
      { key: "prompt", label: "任务输入", required: true, multiline: true },
    ];
  if (
    key === "test-embedding" ||
    key === "test-moderation" ||
    key === "security-check" ||
    key === "extract"
  )
    return [
      { key: "text", label: "待处理文本", required: true, multiline: true },
    ];
  if (key === "test-rerank")
    return [
      { key: "query", label: "查询", required: true },
      {
        key: "documents",
        label: "候选文档（每行一条）",
        required: true,
        multiline: true,
      },
    ];
  if (key === "version")
    return [
      { key: "version", label: "版本号（可选）" },
      { key: "change_summary", label: "变更说明", multiline: true },
    ];
  if (key === "release")
    return [{ key: "version_id", label: "版本 ID（留空使用最新版本）" }];
  if (key === "gray-release")
    return [
      { key: "percentage", label: "流量百分比", required: true, number: true },
    ];
  if (key === "rollback")
    return [{ key: "target_version", label: "目标版本", required: true }];
  if (key === "evaluation-run")
    return [
      { key: "input", label: "样例输入", multiline: true },
      { key: "expected", label: "期望回答", required: true },
      { key: "actual", label: "实际回答", required: true, multiline: true },
    ];
  if (key === "invoke")
    return [{ key: "arguments", label: "调用参数 JSON", multiline: true }];
  return [];
}
function buildPayload(
  key: string,
  values: Record<string, any>,
  advanced: string,
): Record<string, unknown> {
  let payload = Object.fromEntries(
    Object.entries(values).filter(
      ([, value]) => value !== undefined && value !== "",
    ),
  );
  if (key === "run") payload = { context: { input: values.prompt } };
  if (key === "invoke" && typeof values.arguments === "string")
    payload.arguments = parseObject(values.arguments);
  if (key === "test-rerank")
    payload.documents = String(values.documents || "")
      .split("\n")
      .filter(Boolean);
  if (key === "evaluation-run")
    payload = {
      cases: [
        {
          input: values.input || "",
          expected: values.expected,
          actual: values.actual,
        },
      ],
    };
  return { ...payload, ...parsePayload(advanced) };
}
export function FeatureWorkbench({
  page,
  rows,
  onChanged,
}: Props): JSX.Element | null {
  const operations = useMemo(() => operationsFor(page), [page]);
  const [operationKey, setOperationKey] = useState(operations[0]?.key || "");
  const [resourceId, setResourceId] = useState(
    rows.length === 1 ? rows[0].id : "",
  );
  const [payloadText, setPayloadText] = useState("{}");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Record<string, any> | null>(null);
  const [form] = Form.useForm();
  const operation = operations.find((item) => item.key === operationKey);
  if (!operation) return null;
  const execute = async () => {
    if (operation.needsResource && !resourceId) {
      message.warning("请先选择目标资源");
      return;
    }
    try {
      const values = await form.validateFields();
      const payload = buildPayload(operationKey, values, payloadText);
      if (
        operationKey === "upload" &&
        !payload.content &&
        !payload.content_base64
      )
        throw new Error("请上传文件或填写文档内容");
      setRunning(true);
      setError("");
      setResult(null);
      const response = await api.post(operation.buildPath(resourceId), payload);
      setResult(response);
      await onChanged();
    } catch (reason) {
      if (reason instanceof Error) setError(reason.message);
    } finally {
      setRunning(false);
    }
  };
  const responseText =
    result?.final_response ||
    result?.response ||
    result?.content ||
    result?.text ||
    result?.output?.response ||
    result?.output?.content;
  const status =
    result?.status ||
    (result?.valid === false || result?.error ? "failed" : "completed");
  return (
    <div className="operation-workbench">
      <Form
        form={form}
        layout="vertical"
        initialValues={{ top_k: 5, percentage: 10 }}
        disabled={running}
      >
        <Form.Item label="操作">
          <Select
            aria-label="选择操作"
            value={operationKey}
            onChange={(value) => {
              setOperationKey(value);
              setResult(null);
              setError("");
              form.resetFields();
              setPayloadText("{}");
            }}
            options={operations.map((item) => ({
              label: item.label,
              value: item.key,
            }))}
          />
        </Form.Item>
        {operation.needsResource && (
          <Form.Item label="目标资源">
            <Select
              aria-label="目标资源"
              placeholder="请选择资源"
              value={resourceId || undefined}
              onChange={setResourceId}
              showSearch
              optionFilterProp="label"
              options={rows.map((row) => ({
                label: row.name || row.id,
                value: row.id,
              }))}
            />
          </Form.Item>
        )}
        {operationKey === "evaluation-run" && (
          <Alert
            className="mb-4"
            type="info"
            showIcon
            message="按标准化文本精确匹配评分。请提供实际回答，此操作不会自动调用智能体。"
          />
        )}
        {fieldsFor(operationKey).map((field) => (
          <Form.Item
            key={field.key}
            name={field.key}
            label={field.label}
            rules={
              field.required
                ? [{ required: true, message: `请填写${field.label}` }]
                : []
            }
          >
            {field.number ? (
              <InputNumber
                min={1}
                max={field.key === "percentage" ? 100 : 1000}
              />
            ) : field.multiline ? (
              <Input.TextArea rows={4} />
            ) : (
              <Input />
            )}
          </Form.Item>
        ))}
        {operationKey === "upload" && (
          <Upload
            maxCount={1}
            beforeUpload={async (file) => {
              if (file.size > 20 * 1024 * 1024) {
                message.error("请选择不超过 20 MB 的文档");
                return Upload.LIST_IGNORE;
              }
              const reader = new FileReader();
              reader.onload = () => {
                form.setFieldsValue({ filename: file.name, content: "" });
                setPayloadText(
                  JSON.stringify({
                    content_base64: String(reader.result).split(",")[1],
                  }),
                );
              };
              reader.readAsDataURL(file);
              return false;
            }}
            onRemove={() => {
              setPayloadText("{}");
            }}
          >
            <Button icon={<UploadOutlined />}>选择文档文件</Button>
          </Upload>
        )}
        <Collapse
          className="my-4"
          items={[
            {
              key: "payload",
              label: "高级请求参数",
              children: (
                <Form.Item label="请求 JSON">
                  <Input.TextArea
                    rows={5}
                    className="code-input"
                    value={payloadText}
                    onChange={(event) => setPayloadText(event.target.value)}
                  />
                </Form.Item>
              ),
            },
          ]}
        />
        <Button
          type="primary"
          icon={<PlayCircleOutlined />}
          loading={running}
          disabled={operation.needsResource && !rows.length}
          onClick={() => void execute()}
        >
          {operation.label}
        </Button>
      </Form>
      {error && (
        <Alert
          className="mt-4"
          type="error"
          showIcon
          message="操作失败"
          description={error}
        />
      )}
      {result && (
        <div className="operation-result mt-4">
          <Space>
            <Typography.Text strong>执行结果</Typography.Text>
            <StatusTag status={status} />
          </Space>
          {typeof responseText === "string" && (
            <div className="response-content">
              <MarkdownContent>{responseText}</MarkdownContent>
            </div>
          )}
          <Collapse
            className="mt-3"
            defaultActiveKey={
              typeof responseText === "string" ? [] : ["result"]
            }
            items={[
              {
                key: "result",
                label: "结果详情",
                children: (
                  <pre className="json-view">
                    {JSON.stringify(result, null, 2)}
                  </pre>
                ),
              },
            ]}
          />
        </div>
      )}
    </div>
  );
}
export const featureWorkbenchInternals = {
  operationsFor,
  parsePayload,
  buildPayload,
};
