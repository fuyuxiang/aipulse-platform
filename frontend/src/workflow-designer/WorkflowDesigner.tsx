import { Graph, type Edge, type Node } from "@antv/x6";
import {
  ApartmentOutlined,
  CheckCircleOutlined,
  ExpandOutlined,
  PlayCircleOutlined,
  PlusOutlined,
  SaveOutlined,
} from "@ant-design/icons";
import {
  Alert,
  Button,
  Card,
  Collapse,
  Drawer,
  Form,
  Input,
  message,
  Modal,
  Select,
  Space,
  Tag,
} from "antd";
import React, { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useAccess } from "../access/context";
import { MarkdownContent } from "../components/MarkdownContent";
import type { ResourceRecord } from "../models/types";
import { api } from "../services/http";
import { useUnsavedChanges } from "../components/useUnsavedChanges";
import { parseObject, StatusTag } from "../components/PageHeader";

const nodeTypes = [
  { type: "agent", label: "Agent 节点" },
  { type: "tool", label: "工具节点" },
  { type: "model", label: "模型节点" },
  { type: "rag", label: "RAG 节点" },
  { type: "human_approval", label: "审批节点" },
  { type: "condition", label: "条件分支" },
];
const resourceKeys: Record<string, string> = {
  agent: "agents",
  model: "models",
  rag: "knowledge-bases",
  tool: "tools",
};
const configKeys: Record<string, string> = {
  agent: "agent_id",
  model: "model_id",
  rag: "knowledge_base_id",
  tool: "tool_id",
};
function visualNode(
  id: string,
  type: string,
  label: string,
  x: number,
  y: number,
  config: object = {},
) {
  return {
    id,
    x,
    y,
    width: 170,
    height: 54,
    label,
    shape: "rect",
    attrs: {
      body: {
        rx: 8,
        ry: 8,
        stroke: type === "start" || type === "end" ? "#98a2b3" : "#93b4f4",
        fill: "#ffffff",
        strokeWidth: 1.5,
      },
      label: { fill: "#344054", fontSize: 13 },
    },
    ports: {
      groups: {
        in: {
          position: "left",
          attrs: {
            circle: {
              r: 5,
              magnet: "passive",
              stroke: "#7298de",
              fill: "#fff",
            },
          },
        },
        out: {
          position: "right",
          attrs: {
            circle: { r: 5, magnet: true, stroke: "#7298de", fill: "#fff" },
          },
        },
      },
      items: [
        ...(type !== "start" ? [{ id: "in", group: "in" }] : []),
        ...(type !== "end" ? [{ id: "out", group: "out" }] : []),
      ],
    },
    data: { type, config },
  };
}
export function WorkflowDesigner({
  onSaved,
  workflowId: initialId,
}: {
  onSaved?: () => void;
  workflowId?: string;
}): JSX.Element {
  const { can } = useAccess();
  const [, setSearchParams] = useSearchParams();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const graphRef = useRef<Graph | null>(null);
  const definitionRequest = useRef(0);
  const [name, setName] = useState("新建工作流");
  const [workflowId, setWorkflowId] = useState("");
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const markSaved = useUnsavedChanges(dirty);
  const [loadingDefinition, setLoadingDefinition] = useState(!!initialId);
  const [definitionError, setDefinitionError] = useState("");
  const [error, setError] = useState("");
  const [workflows, setWorkflows] = useState<ResourceRecord[]>([]);
  const [resources, setResources] = useState<Record<string, ResourceRecord[]>>(
    {},
  );
  const [selected, setSelected] = useState<Node | Edge | null>(null);
  const [nodeForm] = Form.useForm();
  const [runOpen, setRunOpen] = useState(false);
  const [runError, setRunError] = useState("");
  const [runInput, setRunInput] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<Record<string, any> | null>(null);
  const promptBefore = useRef("");
  const loadResources = async () => {
    const keys = ["workflows", "agents", "models", "tools", "knowledge-bases"];
    const values = await Promise.allSettled(
      keys.map((key) => api.list("/" + key, 1, 200)),
    );
    values.forEach((value, index) => {
      if (value.status === "fulfilled") {
        if (keys[index] === "workflows") setWorkflows(value.value.items);
        else
          setResources((prev) => ({
            ...prev,
            [keys[index]]: value.value.items,
          }));
      }
    });
    const unavailable = values.flatMap((value, index) =>
      value.status === "rejected" ? [keys[index]] : [],
    );
    if (unavailable.length)
      setError(
        `部分资源无法加载：${unavailable.join("、")}。请检查访问权限或稍后刷新。`,
      );
  };
  const openConfig = (cell: Node | Edge) => {
    setSelected(cell);
    const data = cell.getData() || {};
    promptBefore.current = data.config?.prompt || "";
    nodeForm.setFieldsValue({
      resource_id: data.config?.[configKeys[data.type]] || undefined,
      label: cell.isNode()
        ? cell.attr("label/text") || cell.attr("text/text")
        : "",
      prompt: promptBefore.current,
      condition: data.condition || "",
      config: JSON.stringify(data.config || {}, null, 2),
    });
  };
  const loadWorkflow = async (id: string) => {
    const request = ++definitionRequest.current;
    setLoadingDefinition(true);
    setDefinitionError("");
    try {
      const row = await api.get<ResourceRecord>(`/workflows/${id}`);
      if (request !== definitionRequest.current) return;
      const definition = (row.spec?.nodes ? row.spec : row.config) as {
        nodes?: any[];
        edges?: any[];
      };
      const graph = graphRef.current;
      if (!graph) return;
      graph.clearCells();
      (definition.nodes || []).forEach((node, index) =>
        graph.addNode(
          visualNode(
            node.id,
            node.type,
            node.label || node.id,
            node.position?.x ?? 60 + index * 220,
            node.position?.y ?? 100,
            node.config || {},
          ),
        ),
      );
      (definition.edges || []).forEach((edge) =>
        graph.addEdge({
          source: { cell: edge.source, port: "out" },
          target: { cell: edge.target, port: "in" },
          data: { condition: edge.condition || "" },
          labels: edge.condition ? [edge.condition] : [],
        }),
      );
      setWorkflowId(row.id);
      setName(row.name);
      setSelected(null);
      setDirty(false);
      setResult(null);
    } catch (reason) {
      if (request === definitionRequest.current)
        setDefinitionError((reason as Error).message);
    } finally {
      if (request === definitionRequest.current) setLoadingDefinition(false);
    }
  };
  useEffect(() => {
    if (!containerRef.current) return;
    const graph: Graph = new Graph({
      container: containerRef.current,
      async: false,
      autoResize: true,
      grid: { visible: true, size: 16 },
      panning: true,
      mousewheel: { enabled: true, modifiers: ["ctrl", "meta"] },
      connecting: {
        allowBlank: false,
        allowLoop: false,
        allowMulti: false,
        snap: true,
        highlight: true,
        router: "manhattan",
        connector: "rounded",
        connectionPoint: "anchor",
        validateConnection: ({ sourceMagnet, targetMagnet }) =>
          !!sourceMagnet && targetMagnet?.getAttribute("port-group") === "in",
        createEdge: () =>
          graph.createEdge({
            attrs: {
              line: {
                stroke: "#94a3b8",
                strokeWidth: 1.5,
                targetMarker: "block",
              },
            },
          }),
      },
    });
    graphRef.current = graph;
    graph.addNode(visualNode("start", "start", "开始", 60, 120));
    graph.addNode(
      visualNode("agent", "agent", "Agent 节点", 310, 120, { agent_id: "" }),
    );
    graph.addNode(visualNode("end", "end", "结束", 560, 120));
    graph.addEdge({
      source: { cell: "start", port: "out" },
      target: { cell: "agent", port: "in" },
    });
    graph.addEdge({
      source: { cell: "agent", port: "out" },
      target: { cell: "end", port: "in" },
    });
    graph.on("node:click", ({ node }) => openConfig(node));
    graph.on("edge:click", ({ edge }) => openConfig(edge));
    [
      "cell:added",
      "cell:removed",
      "node:change:position",
      "cell:change:data",
      "edge:connected",
    ].forEach((event) => graph.on(event, () => setDirty(true)));
    void loadResources();
    return () => {
      definitionRequest.current++;
      graph.dispose();
      graphRef.current = null;
    };
  }, []);
  useEffect(() => {
    if (initialId && initialId !== workflowId) void loadWorkflow(initialId);
  }, [initialId]);
  const confirmChange = (action: () => void) =>
    dirty
      ? Modal.confirm({
          title: "当前修改尚未保存",
          content: "继续操作将放弃本次修改。",
          okText: "放弃修改",
          cancelText: "继续编辑",
          onOk: action,
        })
      : action();
  const serialize = () => ({
    nodes: (graphRef.current?.getNodes() || []).map((node) => ({
      id: node.id,
      type: String(node.getData()?.type || "agent"),
      label: String(
        node.attr("label/text") || node.attr("text/text") || node.id,
      ),
      config: node.getData()?.config || {},
      position: node.position(),
    })),
    edges: (graphRef.current?.getEdges() || [])
      .map((edge) => ({
        source: edge.getSourceCellId(),
        target: edge.getTargetCellId(),
        condition: String(edge.getData()?.condition || ""),
      }))
      .filter((edge) => edge.source && edge.target),
  });
  const save = async (): Promise<string | undefined> => {
    if (!name.trim()) {
      message.warning("请输入工作流名称");
      return;
    }
    setSaving(true);
    try {
      const definition = serialize();
      const payload = {
        name: name.trim(),
        status: "draft",
        spec: definition,
        config: definition,
      };
      const row = workflowId
        ? await api.update("/workflows", workflowId, payload)
        : await api.create("/workflows", payload);
      markSaved();
      setWorkflowId(row.id);
      setSearchParams({ workflow: row.id }, { replace: true });
      setDirty(false);
      void loadResources();
      message.success("工作流已保存");
      onSaved?.();
      return row.id;
    } catch (reason) {
      message.error((reason as Error).message);
    } finally {
      setSaving(false);
    }
  };
  const validate = async () => {
    try {
      const id = dirty || !workflowId ? await save() : workflowId;
      if (!id) return;
      const response = await api.post<{ valid?: boolean; error?: string }>(
        `/workflows/${id}/validate`,
        serialize(),
      );
      if (response.valid) message.success("工作流校验通过");
      else message.error(response.error || "工作流校验失败");
    } catch (reason) {
      message.error((reason as Error).message);
    }
  };
  const saveConfig = async () => {
    try {
      const values = await nodeForm.validateFields();
      if (!selected) return;
      if (selected.isEdge()) {
        selected.setData({ condition: values.condition || "" });
        selected.setLabels(values.condition ? [values.condition] : []);
      } else {
        const config = parseObject(values.config);
        const type = selected.getData()?.type;
        if (configKeys[type])
          config[configKeys[type]] = values.resource_id || "";
        if (values.prompt !== promptBefore.current)
          config.prompt = values.prompt;
        selected.setData({ ...selected.getData(), config });
        if (values.label?.trim())
          selected.attr("text/text", values.label.trim());
      }
      setDirty(true);
      setSelected(null);
    } catch (reason) {
      if (reason instanceof Error) message.error(reason.message);
    }
  };
  const run = async () => {
    setRunError("");
    setRunning(true);
    setResult(null);
    try {
      const id = dirty || !workflowId ? await save() : workflowId;
      if (!id) return;
      const response = await api.post<Record<string, any>>(
        `/workflows/${id}/run`,
        { context: { input: runInput } },
      );
      setResult(response);
      if (["completed", "success"].includes(response.status))
        message.success("工作流执行完成");
      else if (response.status === "waiting_approval")
        message.info("流程等待审批");
      else if (response.status === "failed") message.error("流程执行失败");
      else message.info("流程状态已更新，请查看执行结果");
    } catch (reason) {
      setRunError((reason as Error).message);
    } finally {
      setRunning(false);
    }
  };
  const waitingNode = Object.entries(result?.results || {}).find(
    ([, value]) =>
      (value as any)?.status === "waiting_approval" ||
      (value as any)?.output?.status === "waiting_approval",
  )?.[0];
  const waitingType = waitingNode
    ? graphRef.current?.getCellById(waitingNode)?.getData()?.type
    : "";
  const approvalDomain = ["human_approval", "approval"].includes(waitingType)
    ? "workflow-approvals"
    : waitingType === "tool"
      ? "tool-approval-tasks"
      : "";
  const canApprove =
    approvalDomain &&
    can(
      approvalDomain === "workflow-approvals"
        ? "workflows:approve"
        : "tools:approve",
    );
  const waitingValue = waitingNode ? result?.results?.[waitingNode] : null;
  const approvalId =
    result?.waiting?.approval_id ||
    result?.pending_approval ||
    waitingValue?.approval_id ||
    waitingValue?.output?.approval_id;
  const decide = async (approved: boolean) => {
    if (!approvalId || !result) return;
    setRunError("");
    setRunning(true);
    try {
      if (!result?.approval_decided)
        await api.post(
          `/${approvalDomain}/${approvalId}/${approved ? "approve" : "reject"}`,
          {},
        );
      if (approved) {
        setResult({ ...result, approval_decided: true });
        setResult(
          await api.post(`/workflow-runs/${result?.run_id}/resume`, {}),
        );
      } else {
        if (approvalDomain === "tool-approval-tasks")
          await api.post(`/workflow-runs/${result.run_id}/cancel`, {});
        setResult({
          ...result,
          status:
            approvalDomain === "workflow-approvals" ? "rejected" : "cancelled",
        });
      }
    } catch (reason) {
      setRunError((reason as Error).message);
    } finally {
      setRunning(false);
    }
  };
  const resourceType = resourceKeys[selected?.getData()?.type];
  return (
    <section className="workflow-editor">
      {definitionError && (
        <Alert
          type="error"
          message="工作流加载失败"
          description={definitionError}
          action={
            <Button onClick={() => initialId && void loadWorkflow(initialId)}>
              重试
            </Button>
          }
        />
      )}
      {error && <Alert type="warning" message={error} showIcon closable />}
      <div className="editor-toolbar">
        <Space wrap>
          <Input
            aria-label="工作流名称"
            disabled={loadingDefinition || !!definitionError}
            value={name}
            onChange={(event) => {
              setName(event.target.value);
              setDirty(true);
            }}
            style={{ width: 220 }}
          />
          <Tag>{dirty ? "未保存" : workflowId ? "已保存" : "新建流程"}</Tag>
          <Select
            aria-label="加载已有工作流"
            value={workflowId || undefined}
            placeholder="加载已有工作流"
            style={{ width: 210 }}
            options={workflows.map((row) => ({
              value: row.id,
              label: row.name,
            }))}
            onChange={(id) => confirmChange(() => void loadWorkflow(id))}
          />
          <Button
            onClick={() => {
              setWorkflowId("");
              setName(`${name} 副本`);
              setDirty(true);
            }}
          >
            另存为新工作流
          </Button>
        </Space>
        <Space>
          <Button
            aria-label="适应画布"
            icon={<ExpandOutlined />}
            onClick={() =>
              graphRef.current?.zoomToFit({ padding: 40, maxScale: 1 })
            }
          />
          <Button
            icon={<CheckCircleOutlined />}
            disabled={saving || loadingDefinition || !!definitionError}
            onClick={() => void validate()}
          >
            校验
          </Button>
          <Button
            icon={<PlayCircleOutlined />}
            disabled={saving || loadingDefinition || !!definitionError}
            onClick={() => {
              setResult(null);
              setRunError("");
              setRunOpen(true);
            }}
          >
            试运行
          </Button>
          <Button
            type="primary"
            icon={<SaveOutlined />}
            loading={saving}
            disabled={loadingDefinition || !!definitionError}
            onClick={() => void save()}
          >
            保存定义
          </Button>
        </Space>
      </div>
      <div
        className={`editor-workspace ${loadingDefinition || definitionError ? "editor-loading" : ""}`}
        aria-busy={loadingDefinition}
      >
        <aside className="node-palette">
          <strong>
            <ApartmentOutlined /> 节点库
          </strong>
          <p>添加节点后，从圆点拖动连线</p>
          {nodeTypes.map((item) => (
            <Button
              key={item.type}
              icon={<PlusOutlined />}
              onClick={() => {
                const count = graphRef.current?.getNodes().length || 0;
                graphRef.current?.addNode(
                  visualNode(
                    `${item.type}-${Date.now()}`,
                    item.type,
                    item.label,
                    100 + (count % 3) * 220,
                    250 + Math.floor((count - 3) / 3) * 100,
                    item.type === "model"
                      ? { model_type: "chat_llm", payload: {} }
                      : {},
                  ),
                );
              }}
            >
              {item.label}
            </Button>
          ))}
        </aside>
        <div className="canvas-container">
          <div ref={containerRef} className="workflow-canvas" />
          <div className="canvas-caption">
            拖动画布移动视野 · 点击节点配置 · Ctrl / ⌘ + 滚轮缩放
          </div>
        </div>
      </div>
      <Drawer
        open={!!selected}
        title={selected?.isEdge() ? "连接条件" : "节点配置"}
        onClose={() => setSelected(null)}
        extra={
          <Space>
            <Button
              danger
              onClick={() => {
                selected?.remove();
                setSelected(null);
                setDirty(true);
              }}
            >
              删除
            </Button>
            <Button type="primary" onClick={() => void saveConfig()}>
              应用配置
            </Button>
          </Space>
        }
      >
        <Form form={nodeForm} layout="vertical">
          {selected?.isEdge() ? (
            <Form.Item name="condition" label="分支条件">
              <Select
                options={[
                  { value: "", label: "始终执行" },
                  { value: "true", label: "条件为真" },
                  { value: "false", label: "条件为假" },
                ]}
              />
            </Form.Item>
          ) : (
            <>
              <Form.Item name="label" label="节点名称">
                <Input />
              </Form.Item>
              {resourceType && (
                <Form.Item
                  name="resource_id"
                  label="选择资源"
                  rules={[{ required: true, message: "请选择资源" }]}
                >
                  <Select
                    placeholder="请选择资源"
                    showSearch
                    optionFilterProp="label"
                    options={(resources[resourceType] || []).map((row) => ({
                      value: row.id,
                      label: row.name,
                    }))}
                  />
                </Form.Item>
              )}
              {selected?.getData()?.type === "agent" && (
                <Form.Item name="prompt" label="任务提示词">
                  <Input.TextArea rows={3} placeholder="留空时使用运行输入" />
                </Form.Item>
              )}
              <Form.Item name="config" label="高级配置 JSON">
                <Input.TextArea rows={12} className="code-input" />
              </Form.Item>
            </>
          )}
        </Form>
      </Drawer>
      <Modal
        open={runOpen}
        title="试运行工作流"
        onCancel={() => !running && setRunOpen(false)}
        footer={null}
        width={700}
      >
        <p className="secondary-text">
          试运行前会保存当前流程。运行输入将传入节点的上下文。
        </p>
        <Input.TextArea
          aria-label="工作流运行输入"
          rows={3}
          value={runInput}
          onChange={(event) => setRunInput(event.target.value)}
          placeholder="输入本次任务"
        />
        <Button
          className="mt-3"
          type="primary"
          loading={running}
          onClick={() => void run()}
        >
          开始运行
        </Button>
        {runError && (
          <Alert
            className="mt-3"
            type="error"
            showIcon
            message="执行操作失败"
            description={runError}
          />
        )}
        {result && (
          <div className="mt-4">
            <StatusTag status={result.status} />
            {result.status === "waiting_approval" && (
              <Alert
                className="mt-3"
                type="warning"
                showIcon
                message="流程等待审批"
                description={
                  approvalDomain
                    ? "审批通过后，可以从当前节点继续执行。"
                    : "此节点内部调用需要审批，当前暂不支持自动续跑。"
                }
                action={
                  canApprove ? (
                    <Space>
                      <Button
                        loading={running}
                        type="primary"
                        onClick={() => void decide(true)}
                      >
                        {result.approval_decided ? "继续执行" : "批准并继续"}
                      </Button>
                      <Button
                        loading={running}
                        danger
                        onClick={() => void decide(false)}
                      >
                        拒绝
                      </Button>
                    </Space>
                  ) : undefined
                }
              />
            )}
            {Object.entries(result.results || {})
              .filter(
                ([id, value]) =>
                  value !== null &&
                  !["start", "end", "approval", "human_approval"].includes(
                    graphRef.current?.getCellById(id)?.getData()?.type,
                  ),
              )
              .map(([id, value]) => (
                <Card
                  className="mt-3"
                  size="small"
                  key={id}
                  title={String(
                    graphRef.current?.getCellById(id)?.attr("label/text") ||
                      graphRef.current?.getCellById(id)?.attr("text/text") ||
                      id,
                  )}
                >
                  <div className="response-content">
                    <MarkdownContent>
                      {executionText(value) ||
                        "该节点返回了结构化结果，请展开执行详情查看。"}
                    </MarkdownContent>
                  </div>
                </Card>
              ))}
            <Collapse
              className="mt-3"
              items={[
                {
                  key: "details",
                  label: "执行详情",
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
      </Modal>
    </section>
  );
}

function executionText(value: any): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean")
    return String(value);
  if (Array.isArray(value))
    return value.map(executionText).filter(Boolean).join("\n\n");
  if (!value || typeof value !== "object") return "";
  for (const key of [
    "response",
    "text",
    "content",
    "agent",
    "model",
    "output",
    "result",
    "value",
    "rag",
    "sources",
  ]) {
    const text = executionText(value[key]);
    if (text) return text;
  }
  return "";
}
