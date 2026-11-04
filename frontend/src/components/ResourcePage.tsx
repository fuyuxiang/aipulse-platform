import {
  ArrowLeftOutlined,
  DeleteOutlined,
  EditOutlined,
  MoreOutlined,
  PlusOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import {
  Alert,
  Button,
  Card,
  Collapse,
  Descriptions,
  Drawer,
  Dropdown,
  Empty,
  Form,
  Input,
  InputNumber,
  message,
  Popconfirm,
  Select,
  Space,
  Switch,
  Table,
  Typography,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import React, {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { pagePermission, useAccess } from "../access/context";
import type { PageConfig, ResourceRecord } from "../models/types";
import { api } from "../services/http";
const WorkflowDesigner = lazy(() =>
  import("../workflow-designer/WorkflowDesigner").then((module) => ({
    default: module.WorkflowDesigner,
  })),
);
const AgentWizard = lazy(() =>
  import("../agent-designer/AgentWizard").then((module) => ({
    default: module.AgentWizard,
  })),
);
import {
  FeatureWorkbench,
  featureWorkbenchInternals,
} from "./FeatureWorkbench";
import { formatTime, PageHeader, parseObject, StatusTag } from "./PageHeader";

const descriptions: Record<string, string> = {
  "/agents": "配置智能体的模型、提示词与能力，创建后即可开始对话。",
  "/workflows":
    "管理可复用的业务流程，在编排器中连接节点、配置参数与验证流程。",
  "/tools": "管理智能体可调用的工具与参数，按需配置调用权限。",
  "/knowledge-bases": "管理知识库、上传文档，并验证检索结果。",
  "/models": "接入对话、向量和重排序模型，验证连接后供智能体使用。",
  "/model-routing-policies": "配置模型选择与回退规则。",
  "/evaluation/datasets": "管理评测数据，并对导入的实际回答进行评分。",
  "/tenants": "管理企业租户与访问状态。",
};

export function ResourcePage({ page }: { page: PageConfig }): JSX.Element {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const { can } = useAccess();
  const canWrite = can(
    pagePermission({ ...page, designer: undefined }).replace(":read", ":write"),
  );
  const [rows, setRows] = useState<ResourceRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [keyword, setKeyword] = useState("");
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [pageNo, setPageNo] = useState(1);
  const [editing, setEditing] = useState<ResourceRecord | null | undefined>(
    undefined,
  );
  const [detail, setDetail] = useState<ResourceRecord | null>(null);
  const [operationRow, setOperationRow] = useState<ResourceRecord | null>(null);
  const [form] = Form.useForm();
  const providerType = Form.useWatch("provider_type", form);
  const authScheme = Form.useWatch("auth_scheme", form);
  const toolType = Form.useWatch("tool_type", form);
  const [models, setModels] = useState<ResourceRecord[]>([]);
  const requestId = useRef(0);
  const load = useCallback(async (): Promise<void> => {
    const id = ++requestId.current;
    setLoading(true);
    setError("");
    try {
      const result = await api.list(page.api, pageNo, 20, {
        name: search,
        status,
      });
      if (id !== requestId.current) return;
      if (!result.items.length && pageNo > 1) {
        setPageNo((value) => value - 1);
        return;
      }
      setRows(result.items);
      setTotal(result.total);
    } catch (reason) {
      if (id === requestId.current) {
        setError((reason as Error).message);
        setRows([]);
      }
    } finally {
      if (id === requestId.current) setLoading(false);
    }
  }, [page.api, pageNo, search, status]);
  useEffect(() => {
    if (!page.designer) void load();
    return () => {
      requestId.current++;
    };
  }, [load, page.designer]);
  const label =
    (
      {
        "/agents": "智能体",
        "/workflows": "工作流",
        "/tools": "工具",
        "/knowledge-bases": "知识库",
        "/models": "模型",
        "/evaluation/datasets": "评测数据集",
      } as Record<string, string>
    )[page.api] || page.title;
  const openEditor = (row?: ResourceRecord) => {
    if (page.api === "/agents") {
      navigate(`/build/agents/create${row ? `?agent=${row.id}` : ""}`);
      return;
    }
    if (page.api === "/workflows") {
      navigate(`/build/workflows/designer${row ? `?workflow=${row.id}` : ""}`);
      return;
    }
    setEditing(row || null);
    if (page.api === "/knowledge-bases" && can("models:read")) {
      void api
        .list("/models", 1, 200)
        .then((result) =>
          setModels(result.items.filter((model) => model.enabled)),
        )
        .catch((reason) => message.error(reason.message));
    }
    form.resetFields();
    form.setFieldsValue({
      name: row?.name || "",
      code: row?.code || "",
      description: row?.description || "",
      enabled: row?.enabled ?? true,
      model_type: row?.model_type || "chat_llm",
      provider_type: row?.provider_type || "openai_compatible",
      model_id: row?.model_id || "",
      base_url: row?.config?.base_url || row?.config?.api_base || "",
      api_key_env: row?.config?.api_key_env || "",
      auth_scheme: row?.config?.auth_scheme || "bearer",
      api_key: "",
      tool_type: row?.config?.type || "echo",
      tool_url: row?.config?.url || "",
      tool_method: row?.config?.method || "GET",
      tool_template: row?.config?.template || "{value}",
      tool_operation: row?.config?.operation || "sum",
      embedding_model_id: row?.config?.embedding_model_id,
      rerank_model_id: row?.config?.rerank_model_id,
      embedding_dimensions: row?.config?.embedding_dimensions || 128,
      configText: JSON.stringify(row?.config || {}, null, 2),
      specText: JSON.stringify(row?.spec || {}, null, 2),
    });
  };
  const save = async () => {
    try {
      const values = await form.validateFields();
      const config = parseObject(values.configText);
      if (page.api === "/tools") {
        config.type = values.tool_type;
        if (values.tool_type === "http")
          Object.assign(config, {
            url: values.tool_url,
            method: values.tool_method,
          });
        if (values.tool_type === "template")
          config.template = values.tool_template;
        if (values.tool_type === "calculator")
          config.operation = values.tool_operation;
      }
      if (page.api === "/knowledge-bases")
        Object.assign(config, {
          embedding_model_id: values.embedding_model_id,
          rerank_model_id: values.rerank_model_id || "",
          embedding_dimensions: values.embedding_dimensions,
        });
      if (page.api === "/models") {
        if (values.base_url) {
          config.base_url = values.base_url;
          delete config.api_base;
        }
        if (values.api_key_env) config.api_key_env = values.api_key_env;
        else delete config.api_key_env;
        config.auth_scheme = values.auth_scheme;
        if (values.api_key) config.api_key = values.api_key;
      }
      const spec = parseObject(values.specText);
      setSaving(true);
      const payload: Partial<ResourceRecord> = {
        name: values.name.trim(),
        code: values.code?.trim() || "",
        description: values.description || "",
        enabled: values.enabled,
      };
      if (
        !editing ||
        JSON.stringify(config) !== JSON.stringify(editing.config || {})
      )
        payload.config = config;
      if (
        !editing ||
        values.specText !== JSON.stringify(editing.spec || {}, null, 2)
      )
        payload.spec = spec;
      if (page.api === "/models")
        Object.assign(payload, {
          model_type: values.model_type,
          provider_type: values.provider_type,
          model_id: values.model_id,
        });
      if (editing) await api.update(page.api, editing.id, payload);
      else await api.create(page.api, payload);
      message.success(`${label}已保存`);
      setEditing(undefined);
      await load();
    } catch (reason) {
      if (reason instanceof Error) message.error(reason.message);
    } finally {
      setSaving(false);
    }
  };
  const toggle = async (row: ResourceRecord) => {
    try {
      await api.update(page.api, row.id, {
        enabled: !row.enabled,
        status: row.enabled ? "disabled" : "active",
      });
      message.success(row.enabled ? "已停用" : "已启用");
      await load();
    } catch (reason) {
      message.error((reason as Error).message);
    }
  };
  const remove = async (row: ResourceRecord) => {
    try {
      await api.remove(page.api, row.id);
      message.success("已删除");
      await load();
    } catch (reason) {
      message.error((reason as Error).message);
    }
  };
  if (page.designer)
    return (
      <div
        className={`enterprise-page ${page.designer === "workflow" ? "editor-page" : "wizard-page"}`}
      >
        <PageHeader
          title={
            page.designer === "workflow"
              ? "工作流编排"
              : params.get("agent")
                ? "编辑智能体"
                : "创建智能体"
          }
          description={
            page.designer === "workflow"
              ? "连接节点，配置执行参数，再保存并校验流程。"
              : "按步骤配置模型、提示词和能力。可选配置可以稍后补充。"
          }
          actions={
            <Button
              icon={<ArrowLeftOutlined />}
              onClick={() =>
                navigate(
                  page.designer === "workflow"
                    ? "/build/workflows"
                    : "/build/agents",
                )
              }
            >
              返回列表
            </Button>
          }
        />
        <Suspense
          fallback={<div className="page-loading">正在加载编辑器…</div>}
        >
          {page.designer === "workflow" ? (
            <WorkflowDesigner
              workflowId={params.get("workflow") || undefined}
            />
          ) : (
            <AgentWizard
              agentId={params.get("agent") || undefined}
              onCreated={() => navigate("/build/agents")}
            />
          )}
        </Suspense>
      </div>
    );
  const operations = featureWorkbenchInternals.operationsFor(page);
  const columns: ColumnsType<ResourceRecord> = [
    {
      title: `${label}名称`,
      dataIndex: "name",
      ellipsis: true,
      render: (value, row) => (
        <div className="resource-name-cell">
          <Button type="link" onClick={() => setDetail(row)}>
            {value || row.code || row.id}
          </Button>
          <Typography.Text type="secondary" ellipsis>
            {row.description || row.code || "暂无描述"}
          </Typography.Text>
        </div>
      ),
    },
    {
      title: "状态",
      dataIndex: "status",
      width: 120,
      render: (value) => <StatusTag status={value} />,
    },
    {
      title: "启用",
      dataIndex: "enabled",
      width: 90,
      render: (value) => (
        <span className={`state-label ${value ? "is-enabled" : ""}`}>
          {value ? "已启用" : "已停用"}
        </span>
      ),
    },
    {
      title: "更新时间",
      dataIndex: "updated_at",
      width: 175,
      render: (value) => (
        <span className="secondary-text">{formatTime(value)}</span>
      ),
    },
    {
      title: "操作",
      width: canWrite ? 200 : 80,
      fixed: "right",
      render: (_, row) => (
        <Space size={4}>
          <Button type="link" onClick={() => setDetail(row)}>
            详情
          </Button>
          {canWrite && (
            <>
              <Button
                type="link"
                icon={<EditOutlined />}
                onClick={() => openEditor(row)}
              >
                编辑
              </Button>
              <Dropdown
                trigger={["click"]}
                menu={{
                  items: [
                    { key: "toggle", label: row.enabled ? "停用" : "启用" },
                    ...(operations.length
                      ? [{ key: "operations", label: "运行与操作" }]
                      : []),
                  ],
                  onClick: ({ key }) =>
                    key === "toggle" ? void toggle(row) : setOperationRow(row),
                }}
              >
                <Button
                  type="text"
                  aria-label={`更多操作 ${row.name}`}
                  icon={<MoreOutlined />}
                />
              </Dropdown>
              <Popconfirm
                title={`删除${label}？`}
                description="删除后将从当前列表移除。"
                onConfirm={() => remove(row)}
              >
                <Button
                  type="text"
                  danger
                  aria-label={`删除 ${row.name}`}
                  icon={<DeleteOutlined />}
                />
              </Popconfirm>
            </>
          )}
        </Space>
      ),
    },
  ];
  return (
    <div className="enterprise-page resource-page">
      <PageHeader
        title={page.title}
        description={
          descriptions[page.api] || `管理${page.title}，查看详情并维护配置。`
        }
        actions={
          <Space>
            <Button
              aria-label="刷新列表"
              icon={<ReloadOutlined />}
              loading={loading}
              onClick={() => void load()}
            />
            {canWrite && (
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => openEditor()}
              >{`创建${label}`}</Button>
            )}
          </Space>
        }
      />
      {error && (
        <Alert
          className="mb-4"
          type="error"
          showIcon
          message="列表加载失败"
          description={error}
          action={<Button onClick={() => void load()}>重试</Button>}
        />
      )}
      <Card className="enterprise-card table-card" bordered={false}>
        <div className="list-toolbar">
          <Space wrap>
            <Input.Search
              aria-label="按名称搜索"
              placeholder={`搜索${label}名称`}
              value={keyword}
              allowClear
              onChange={(event) => {
                setKeyword(event.target.value);
                if (!event.target.value) {
                  setSearch("");
                  setPageNo(1);
                }
              }}
              onSearch={(value) => {
                setSearch(value.trim());
                setPageNo(1);
              }}
              style={{ width: 280, maxWidth: "100%" }}
            />
            <Select
              aria-label="筛选状态"
              placeholder="全部状态"
              allowClear
              value={status || undefined}
              onChange={(value) => {
                setStatus(value || "");
                setPageNo(1);
              }}
              style={{ width: 140 }}
              options={["active", "draft", "published", "disabled"].map(
                (value) => ({
                  value,
                  label: (
                    {
                      active: "正常",
                      draft: "草稿",
                      published: "已发布",
                      disabled: "已停用",
                    } as Record<string, string>
                  )[value],
                }),
              )}
            />
          </Space>
          <span className="secondary-text">共 {total} 项</span>
        </div>
        <Table<ResourceRecord>
          rowKey="id"
          columns={columns}
          dataSource={rows}
          loading={loading}
          scroll={{ x: 760 }}
          pagination={{
            current: pageNo,
            total,
            pageSize: 20,
            showSizeChanger: false,
            onChange: setPageNo,
            showTotal: (count) => `共 ${count} 项`,
          }}
          locale={{
            emptyText: (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={
                  search || status
                    ? "没有符合条件的资源，请调整筛选条件"
                    : `还没有${label}`
                }
              >
                {canWrite && !search && !status && (
                  <Button
                    type="primary"
                    onClick={() => openEditor()}
                  >{`创建第一个${label}`}</Button>
                )}
              </Empty>
            ),
          }}
        />
      </Card>
      {canWrite && operations.length > 0 && (
        <Collapse
          className="advanced-operations"
          items={[
            {
              key: "advanced",
              label: "高级操作",
              children: (
                <FeatureWorkbench
                  key={page.api}
                  page={page}
                  rows={rows}
                  onChanged={load}
                />
              ),
            },
          ]}
        />
      )}
      <Drawer
        open={editing !== undefined}
        title={`${editing ? "编辑" : "创建"}${label}`}
        width={560}
        onClose={() => !saving && setEditing(undefined)}
        extra={
          <Button type="primary" loading={saving} onClick={() => void save()}>
            保存
          </Button>
        }
      >
        <Form form={form} layout="vertical" preserve>
          <Form.Item
            name="name"
            label="名称"
            rules={[
              { required: true, whitespace: true, message: "请输入名称" },
            ]}
          >
            <Input maxLength={120} />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input.TextArea rows={3} placeholder="简要说明用途" />
          </Form.Item>
          <Form.Item name="code" label="编码">
            <Input placeholder="留空自动生成" />
          </Form.Item>
          <Form.Item name="enabled" label="启用" valuePropName="checked">
            <Switch />
          </Form.Item>
          {page.api === "/tools" && (
            <>
              <Form.Item
                name="tool_type"
                label="工具类型"
                rules={[{ required: true }]}
              >
                <Select
                  options={[
                    { value: "echo", label: "参数回显（调试）" },
                    { value: "calculator", label: "计算器" },
                    { value: "template", label: "文本模板" },
                    { value: "http", label: "HTTP 接口" },
                  ]}
                />
              </Form.Item>
              {toolType === "calculator" && (
                <Form.Item name="tool_operation" label="默认运算">
                  <Select
                    options={[
                      { value: "sum", label: "求和" },
                      { value: "multiply", label: "乘积" },
                    ]}
                  />
                </Form.Item>
              )}
              {toolType === "template" && (
                <Form.Item
                  name="tool_template"
                  label="文本模板"
                  rules={[{ required: true }]}
                >
                  <Input.TextArea rows={3} placeholder="例如：你好，{name}" />
                </Form.Item>
              )}
              {toolType === "http" && (
                <>
                  <Form.Item
                    name="tool_url"
                    label="接口地址"
                    rules={[
                      { required: true },
                      {
                        type: "url",
                        message: "请输入完整的 HTTP 或 HTTPS 地址",
                      },
                    ]}
                  >
                    <Input placeholder="https://example.com/api" />
                  </Form.Item>
                  <Form.Item name="tool_method" label="请求方法">
                    <Select
                      options={["GET", "POST", "PUT", "PATCH", "DELETE"].map(
                        (value) => ({ value, label: value }),
                      )}
                    />
                  </Form.Item>
                </>
              )}
            </>
          )}
          {page.api === "/knowledge-bases" && (
            <>
              <Form.Item
                name="embedding_model_id"
                label="向量模型"
                rules={[{ required: true, message: "请选择向量模型" }]}
              >
                <Select
                  showSearch
                  optionFilterProp="label"
                  placeholder="选择用于文档与查询的向量模型"
                  options={models
                    .filter((model) => model.model_type === "embedding")
                    .map((model) => ({ value: model.id, label: model.name }))}
                  onChange={(id) => {
                    const dimensions = models.find((model) => model.id === id)
                      ?.config?.embedding_dimensions;
                    form.setFieldValue(
                      "embedding_dimensions",
                      dimensions || undefined,
                    );
                  }}
                />
              </Form.Item>
              <Form.Item
                name="embedding_dimensions"
                label="向量维度"
                rules={[
                  { required: true, message: "请填写模型输出的向量维度" },
                ]}
                extra="必须与所选模型的实际输出一致；本地演示模型默认 128。"
              >
                <InputNumber min={1} max={65536} />
              </Form.Item>
              <Form.Item name="rerank_model_id" label="重排序模型（可选）">
                <Select
                  allowClear
                  showSearch
                  optionFilterProp="label"
                  options={models
                    .filter((model) => model.model_type === "rerank")
                    .map((model) => ({ value: model.id, label: model.name }))}
                />
              </Form.Item>
            </>
          )}
          {page.api === "/models" && (
            <>
              <Form.Item
                name="model_type"
                label="模型类型"
                rules={[{ required: true }]}
              >
                <Select
                  options={[
                    "chat_llm",
                    "reasoning_llm",
                    "vision_language",
                    "embedding",
                    "rerank",
                  ].map((value) => ({ value, label: value }))}
                />
              </Form.Item>
              <Form.Item
                name="provider_type"
                label="供应商类型"
                rules={[{ required: true }]}
              >
                <Select
                  options={[
                    { value: "openai_compatible", label: "OpenAI 兼容接口" },
                    { value: "echo_agent_native", label: "本地演示模型" },
                  ]}
                />
              </Form.Item>
              <Form.Item
                name="model_id"
                label="模型标识"
                rules={[{ required: true }]}
              >
                <Input placeholder="服务端提供的模型名称" />
              </Form.Item>
              {providerType !== "echo_agent_native" && (
                <>
                  <Form.Item
                    name="base_url"
                    label="服务地址"
                    rules={[
                      { required: true, message: "请输入服务地址" },
                      {
                        type: "url",
                        message: "请输入完整的 HTTP 或 HTTPS 地址",
                      },
                    ]}
                  >
                    <Input placeholder="https://example.com/v1" />
                  </Form.Item>
                  <Form.Item name="auth_scheme" label="认证方式">
                    <Select
                      options={[
                        { value: "bearer", label: "API 密钥" },
                        { value: "none", label: "无需认证" },
                      ]}
                    />
                  </Form.Item>
                  {authScheme !== "none" && (
                    <>
                      <Form.Item name="api_key_env" label="密钥环境变量">
                        <Input placeholder="例如：LLM_API_KEY" />
                      </Form.Item>
                      <Form.Item name="api_key" label="或填写 API 密钥">
                        <Input.Password
                          autoComplete="new-password"
                          placeholder={
                            editing
                              ? "留空保留原密钥"
                              : "环境变量与密钥填写一项即可"
                          }
                        />
                      </Form.Item>
                    </>
                  )}
                </>
              )}
            </>
          )}
          <Collapse
            defaultActiveKey={!editing ? ["config"] : []}
            items={[
              {
                key: "config",
                label: "高级配置",
                children: (
                  <>
                    <Form.Item name="configText" label="配置 JSON">
                      <Input.TextArea rows={8} className="code-input" />
                    </Form.Item>
                    <Form.Item name="specText" label="规格 JSON">
                      <Input.TextArea rows={5} className="code-input" />
                    </Form.Item>
                  </>
                ),
              },
            ]}
          />
        </Form>
      </Drawer>
      <Drawer
        open={!!detail}
        title={detail?.name || `${label}详情`}
        width={620}
        onClose={() => setDetail(null)}
        extra={
          canWrite && (
            <Button
              onClick={() => {
                if (detail) openEditor(detail);
                setDetail(null);
              }}
            >
              编辑
            </Button>
          )
        }
      >
        {detail && (
          <>
            <Descriptions
              column={1}
              bordered
              size="small"
              items={[
                {
                  key: "description",
                  label: "描述",
                  children: detail.description || "暂无描述",
                },
                {
                  key: "status",
                  label: "状态",
                  children: <StatusTag status={detail.status} />,
                },
                {
                  key: "id",
                  label: "资源 ID",
                  children: (
                    <Typography.Text copyable>{detail.id}</Typography.Text>
                  ),
                },
                {
                  key: "updated",
                  label: "更新时间",
                  children: formatTime(detail.updated_at),
                },
              ]}
            />
            {canWrite && operations.length > 0 && (
              <div className="mt-4">
                <FeatureWorkbench
                  page={page}
                  rows={[detail]}
                  onChanged={load}
                />
              </div>
            )}
            <Collapse
              className="mt-4"
              items={[
                {
                  key: "config",
                  label: "查看完整配置",
                  children: (
                    <pre className="json-view">
                      {JSON.stringify(
                        { config: detail.config, spec: detail.spec },
                        null,
                        2,
                      )}
                    </pre>
                  ),
                },
              ]}
            />
          </>
        )}
      </Drawer>
      <Drawer
        open={!!operationRow}
        title={`${operationRow?.name} · 运行与操作`}
        width={620}
        onClose={() => setOperationRow(null)}
      >
        {operationRow && (
          <FeatureWorkbench
            key={operationRow.id}
            page={page}
            rows={[operationRow]}
            onChanged={load}
          />
        )}
      </Drawer>
    </div>
  );
}
