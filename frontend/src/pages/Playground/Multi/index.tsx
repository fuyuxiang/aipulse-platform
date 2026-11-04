import { Graph } from "@antv/x6";
import {
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
  ReloadOutlined,
  TeamOutlined,
} from "@ant-design/icons";
import {
  Alert,
  Button,
  Card,
  Collapse,
  Descriptions,
  Drawer,
  Empty,
  Form,
  Input,
  InputNumber,
  message,
  Modal,
  Popconfirm,
  Select,
  Space,
  Switch,
  Table,
  Tag,
} from "antd";
import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useAccess } from "../../../access/context";
import { MarkdownContent } from "../../../components/MarkdownContent";
import {
  formatTime,
  PageHeader,
  StatusTag,
} from "../../../components/PageHeader";
import { api } from "../../../services/http";
import type { ResourceRecord } from "../../../models/types";

interface Member {
  id: string;
  agent_id: string;
  name: string;
  spec?: Record<string, any>;
}
interface Team extends ResourceRecord {
  spec?: Record<string, any>;
  members?: Member[];
}
const topologyLabels: Record<string, string> = {
  star: "协调者模式",
  pipeline: "流水线",
  mesh: "全连接",
};
export function MultiAgentPage(): JSX.Element {
  const { can } = useAccess();
  const canWrite = can("agents:write");
  const [teams, setTeams] = useState<Team[]>([]);
  const [total, setTotal] = useState(0);
  const [pageNo, setPageNo] = useState(1);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [editing, setEditing] = useState<Team | null | undefined>(undefined);
  const [saving, setSaving] = useState(false);
  const [detail, setDetail] = useState<Team | null>(null);
  const [runTeamValue, setRunTeamValue] = useState<Team | null>(null);
  const [runPrompt, setRunPrompt] = useState("");
  const [runResult, setRunResult] = useState<Record<string, any> | null>(null);
  const [runError, setRunError] = useState("");
  const [running, setRunning] = useState(false);
  const [form] = Form.useForm();
  const topology = Form.useWatch("topology", form) || "star";
  const [agents, setAgents] = useState<ResourceRecord[]>([]);
  const [historyTeam, setHistoryTeam] = useState<Team | null>(null);
  const [history, setHistory] = useState<ResourceRecord[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [historyPage, setHistoryPage] = useState(1);
  const [historyTotal, setHistoryTotal] = useState(0);
  const canvas = useRef<HTMLDivElement | null>(null);
  const loadTeams = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await api.list<Team>("/agent-teams", pageNo, 20, {
        name: search,
      });
      setTeams(response.items);
      setTotal(response.total);
      if (!response.items.length && pageNo > 1) setPageNo((value) => value - 1);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    void loadTeams();
  }, [pageNo, search]);
  useEffect(() => {
    void api
      .list("/agents", 1, 200)
      .then((response) => setAgents(response.items))
      .catch((reason) => setError(reason.message));
  }, []);
  const agentName = (id: string) =>
    agents.find((agent) => agent.id === id)?.name || `智能体 ${id.slice(0, 8)}`;
  useEffect(() => {
    if (!detail || !canvas.current) return;
    const graph = new Graph({
      container: canvas.current,
      autoResize: true,
      grid: true,
      panning: true,
      interacting: false,
    });
    const members = [...(detail.members || [])].sort(
      (a, b) => (a.spec?.priority || 0) - (b.spec?.priority || 0),
    );
    const add = (id: string, label: string, x: number, y: number) =>
      graph.addNode({
        id,
        x,
        y,
        width: 150,
        height: 56,
        label,
        attrs: {
          body: { rx: 8, ry: 8, fill: "#fff", stroke: "#a9bfe9" },
          label: { fill: "#344054", fontSize: 12 },
        },
      });
    if (detail.spec?.topology === "pipeline")
      members.forEach((member, index) => {
        add(member.id, agentName(member.agent_id), 35 + index * 210, 130);
        if (index > 0)
          graph.addEdge({ source: members[index - 1].id, target: member.id });
      });
    else {
      if (detail.spec?.topology === "star")
        add(
          "coordinator",
          agentName(detail.spec.coordinator_agent_id || ""),
          265,
          135,
        );
      members.forEach((member, index) => {
        const angle = (2 * Math.PI * index) / members.length;
        add(
          member.id,
          agentName(member.agent_id),
          265 + 230 * Math.cos(angle),
          135 + 110 * Math.sin(angle),
        );
        if (detail.spec?.topology === "star")
          graph.addEdge({ source: "coordinator", target: member.id });
      });
      if (detail.spec?.topology === "mesh")
        for (let i = 0; i < members.length; i++)
          for (let j = i + 1; j < members.length; j++)
            graph.addEdge({ source: members[i].id, target: members[j].id });
    }
    graph.zoomToFit({ padding: 25, maxScale: 1 });
    return () => graph.dispose();
  }, [detail, agents]);
  const edit = async (team?: Team) => {
    try {
      const value = team
        ? await api.get<Team>(`/agent-teams/${team.id}`)
        : null;
      setEditing(value);
      form.resetFields();
      form.setFieldsValue(
        value
          ? {
              name: value.name,
              ...value.spec,
              members: value.members?.map((member) => ({
                id: member.id,
                agent_id: member.agent_id,
                description: member.spec?.description || "",
              })),
            }
          : {
              topology: "star",
              max_rounds: 10,
              timeout_seconds: 300,
              shared_memory_enabled: true,
              members: [],
            },
      );
    } catch (reason) {
      message.error((reason as Error).message);
    }
  };
  const saveTeam = async () => {
    try {
      const values = await form.validateFields();
      const members = values.members || [];
      if (!members.length) throw new Error("请至少添加一名成员");
      if (
        new Set(members.map((member: any) => member.agent_id)).size !==
        members.length
      )
        throw new Error("请勿重复选择同一智能体");
      setSaving(true);
      const spec = { ...editing?.spec, ...values };
      delete spec.name;
      delete spec.members;
      if (editing)
        await api.update("/agent-teams", editing.id, {
          name: values.name,
          spec,
          members,
        } as any);
      else
        await api.create("/agent-teams", {
          ...values,
          members: members.map((member: any, priority: number) => ({
            ...member,
            role: "worker",
            priority,
          })),
        } as any);
      message.success(editing ? "团队已保存" : "团队创建成功");
      setEditing(undefined);
      await loadTeams();
    } catch (reason) {
      if (reason instanceof Error) message.error(reason.message);
    } finally {
      setSaving(false);
    }
  };
  const run = async () => {
    if (!runTeamValue || !runPrompt.trim() || running) return;
    setRunning(true);
    setRunError("");
    setRunResult(null);
    try {
      const response = await api.post(`/agent-teams/${runTeamValue.id}/run`, {
        prompt: runPrompt,
      });
      setRunResult(response);
      if (response.status === "completed") message.success("执行完成");
    } catch (reason) {
      setRunError((reason as Error).message);
    } finally {
      setRunning(false);
    }
  };
  const loadHistory = async (team: Team, number = 1) => {
    setHistoryTeam(team);
    setHistoryPage(number);
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const response = await api.list(
        `/agent-teams/${team.id}/runs`,
        number,
        10,
      );
      setHistory(response.items);
      setHistoryTotal(response.total);
    } catch (reason) {
      setHistoryError((reason as Error).message);
    } finally {
      setHistoryLoading(false);
    }
  };
  return (
    <div className="enterprise-page team-page">
      <PageHeader
        title="团队协作"
        description="为成员分配职责，选择协作方式，执行任务并回看结果。"
        actions={
          <Space>
            <Button
              icon={<ReloadOutlined />}
              aria-label="刷新团队"
              loading={loading}
              onClick={() => void loadTeams()}
            />
            {canWrite && (
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => void edit()}
              >
                创建团队
              </Button>
            )}
          </Space>
        }
      />
      {error && (
        <Alert
          className="mb-4"
          type="error"
          message="团队数据加载失败"
          description={error}
          showIcon
          action={<Button onClick={() => void loadTeams()}>重试</Button>}
        />
      )}
      <Card className="enterprise-card table-card">
        <div className="list-toolbar">
          <Input.Search
            placeholder="搜索团队名称"
            allowClear
            onSearch={(value) => {
              setSearch(value);
              setPageNo(1);
            }}
            onChange={(event) => {
              if (!event.target.value) {
                setSearch("");
                setPageNo(1);
              }
            }}
            style={{ width: 280 }}
          />
          <span className="secondary-text">共 {total} 个团队</span>
        </div>
        <Table<Team>
          rowKey="id"
          loading={loading}
          dataSource={teams}
          scroll={{ x: 720 }}
          pagination={{
            current: pageNo,
            total,
            pageSize: 20,
            showSizeChanger: false,
            onChange: setPageNo,
          }}
          columns={[
            {
              title: "团队名称",
              dataIndex: "name",
              render: (value, row) => (
                <Button
                  type="link"
                  onClick={() =>
                    void api
                      .get<Team>(`/agent-teams/${row.id}`)
                      .then(setDetail)
                      .catch((reason) => message.error(reason.message))
                  }
                >
                  {value}
                </Button>
              ),
            },
            {
              title: "协作方式",
              render: (_, row) => (
                <Tag>
                  {topologyLabels[row.spec?.topology] || row.spec?.topology}
                </Tag>
              ),
            },
            {
              title: "更新时间",
              width: 175,
              render: (_, row) => (
                <span className="secondary-text">
                  {formatTime(row.updated_at)}
                </span>
              ),
            },
            {
              title: "操作",
              width: canWrite ? 290 : 100,
              render: (_, row) => (
                <Space size={4}>
                  <Button type="link" onClick={() => void loadHistory(row)}>
                    历史记录
                  </Button>
                  {canWrite && (
                    <>
                      <Button type="link" onClick={() => void edit(row)}>
                        编辑
                      </Button>
                      <Button
                        type="primary"
                        size="small"
                        onClick={() => {
                          setRunTeamValue(row);
                          setRunPrompt("");
                          setRunResult(null);
                          setRunError("");
                        }}
                      >
                        执行
                      </Button>
                      <Popconfirm
                        title="删除团队？"
                        onConfirm={async () => {
                          try {
                            await api.remove("/agent-teams", row.id);
                            message.success("团队已删除");
                            await loadTeams();
                          } catch (reason) {
                            message.error((reason as Error).message);
                          }
                        }}
                      >
                        <Button
                          type="text"
                          danger
                          aria-label={`删除团队 ${row.name}`}
                          icon={<DeleteOutlined />}
                        />
                      </Popconfirm>
                    </>
                  )}
                </Space>
              ),
            },
          ]}
          locale={{
            emptyText: (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={search ? "没有匹配的团队" : "还没有协作团队"}
              >
                {canWrite && !search && (
                  <Button onClick={() => void edit()} type="primary">
                    创建第一个团队
                  </Button>
                )}
              </Empty>
            ),
          }}
        />
      </Card>
      <Drawer
        open={editing !== undefined}
        title={editing ? "编辑 Agent 团队" : "创建 Agent 团队"}
        width={560}
        onClose={() => !saving && setEditing(undefined)}
        extra={
          <Button
            type="primary"
            loading={saving}
            onClick={() => void saveTeam()}
          >
            {editing ? "保存" : "创建"}
          </Button>
        }
      >
        <Form form={form} layout="vertical" preserve>
          {!agents.length && (
            <Alert
              type="info"
              className="mb-4"
              message={
                <span>
                  先配置团队可使用的智能体。
                  <Link to="/build/agents">前往智能体列表</Link>
                </span>
              }
            />
          )}
          <Form.Item
            name="name"
            label="团队名称"
            rules={[
              { required: true, whitespace: true, message: "请输入团队名称" },
            ]}
          >
            <Input maxLength={120} />
          </Form.Item>
          <Form.Item name="description" label="用途">
            <Input.TextArea rows={2} />
          </Form.Item>
          <Form.Item name="topology" label="通信拓扑">
            <Select
              options={[
                { value: "star", label: "协调者模式" },
                { value: "pipeline", label: "流水线" },
                { value: "mesh", label: "全连接" },
              ]}
            />
          </Form.Item>
          <p className="secondary-text">
            {topology === "pipeline"
              ? "成员按添加顺序依次处理，上一步输出交给下一步。"
              : topology === "star"
                ? "由协调者组织成员执行，并汇总结果。"
                : "成员并行参与，平台汇总各成员输出。"}
          </p>
          {topology === "star" && (
            <Form.Item
              name="coordinator_agent_id"
              label="协调者 Agent"
              rules={[{ required: true, message: "请选择协调者" }]}
            >
              <Select
                showSearch
                optionFilterProp="label"
                options={agents.map((agent) => ({
                  value: agent.id,
                  label: agent.name,
                }))}
              />
            </Form.Item>
          )}
          <Form.List name="members">
            {(fields, { add, remove }) => (
              <>
                <div className="member-list-heading">
                  <strong>团队成员</strong>
                  <Button
                    size="small"
                    icon={<PlusOutlined />}
                    onClick={() => add()}
                  >
                    添加成员
                  </Button>
                </div>
                {fields.map((field) => (
                  <Card
                    key={field.key}
                    size="small"
                    className="member-form-card"
                    title={`成员 ${field.name + 1}`}
                    extra={
                      <Button
                        size="small"
                        type="text"
                        danger
                        onClick={() => remove(field.name)}
                      >
                        移除
                      </Button>
                    }
                  >
                    <Form.Item name={[field.name, "id"]} hidden>
                      <Input />
                    </Form.Item>
                    <Form.Item
                      name={[field.name, "agent_id"]}
                      label="Agent"
                      rules={[{ required: true, message: "请选择成员智能体" }]}
                    >
                      <Select
                        showSearch
                        optionFilterProp="label"
                        options={agents.map((agent) => ({
                          value: agent.id,
                          label: agent.name,
                        }))}
                      />
                    </Form.Item>
                    <Form.Item name={[field.name, "description"]} label="职责">
                      <Input placeholder="说明该成员承担的工作" />
                    </Form.Item>
                  </Card>
                ))}
              </>
            )}
          </Form.List>
          <Collapse
            className="mt-4"
            items={[
              {
                key: "advanced",
                label: "运行设置",
                forceRender: true,
                children: (
                  <>
                    <Form.Item name="max_rounds" label="最大轮次">
                      <InputNumber min={1} max={100} />
                    </Form.Item>
                    <Form.Item name="timeout_seconds" label="超时时间（秒）">
                      <InputNumber min={1} max={3600} />
                    </Form.Item>
                    <Form.Item
                      name="shared_memory_enabled"
                      label="启用共享记忆"
                      valuePropName="checked"
                    >
                      <Switch />
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
        width={760}
        title={detail?.name}
        onClose={() => setDetail(null)}
      >
        {detail && (
          <>
            <Descriptions
              column={2}
              items={[
                {
                  key: "topology",
                  label: "协作方式",
                  children: topologyLabels[detail.spec?.topology],
                },
                {
                  key: "members",
                  label: "成员",
                  children: detail.members?.length,
                },
                {
                  key: "description",
                  label: "用途",
                  children: detail.spec?.description || "暂无描述",
                },
              ]}
            />
            <div ref={canvas} className="team-topology" />
            {detail.members?.map((member) => (
              <div className="team-member" key={member.id}>
                <TeamOutlined />
                <div>
                  <strong>{agentName(member.agent_id)}</strong>
                  <p>{member.spec?.description || "未填写职责"}</p>
                </div>
              </div>
            ))}
          </>
        )}
      </Drawer>
      <Drawer
        open={!!historyTeam}
        title={`${historyTeam?.name || ""} · 执行历史`}
        width={700}
        onClose={() => setHistoryTeam(null)}
      >
        <Table<ResourceRecord>
          rowKey="id"
          loading={historyLoading}
          dataSource={history}
          pagination={{
            current: historyPage,
            total: historyTotal,
            pageSize: 10,
            showSizeChanger: false,
            onChange: (number) =>
              historyTeam && void loadHistory(historyTeam, number),
          }}
          columns={[
            {
              title: "执行时间",
              render: (_, row) => formatTime(row.created_at),
            },
            {
              title: "状态",
              render: (_, row) => <StatusTag status={row.status} />,
            },
          ]}
          expandable={{
            expandedRowKeys: history.map((row) => row.id),
            showExpandColumn: false,
            expandedRowRender: (row) => (
              <>
                <p className="secondary-text">
                  {String(row.spec?.prompt || "")}
                </p>
                <div className="response-content">
                  <MarkdownContent>
                    {String(
                      row.spec?.final_response ||
                        (row as any).error_message ||
                        "暂无输出",
                    )}
                  </MarkdownContent>
                </div>
              </>
            ),
          }}
          locale={{
            emptyText: (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description="暂无执行记录"
              />
            ),
          }}
        />
        {historyError && <Alert type="error" message={historyError} />}
      </Drawer>
      <Modal
        open={!!runTeamValue}
        title={`执行任务 · ${runTeamValue?.name}`}
        onCancel={() => !running && setRunTeamValue(null)}
        footer={null}
        width={760}
        maskClosable={!running}
        closable={!running}
      >
        <Input.TextArea
          rows={4}
          value={runPrompt}
          onChange={(event) => setRunPrompt(event.target.value)}
          placeholder="输入任务描述..."
          disabled={running}
        />
        <Button
          type="primary"
          loading={running}
          disabled={!runPrompt.trim()}
          className="mt-3"
          onClick={() => void run()}
        >
          执行
        </Button>
        {running && (
          <p className="secondary-text mt-3">
            团队正在执行，请等待成员返回结果。完成后可在历史记录中回看。
          </p>
        )}
        {runError && (
          <Alert
            className="mt-4"
            type="error"
            showIcon
            message="执行失败"
            description={runError}
          />
        )}
        {runResult && (
          <div className="operation-result mt-4">
            <Space>
              <strong>执行结果</strong>
              <StatusTag status={runResult.status} />
            </Space>
            {runResult.status === "waiting_approval" && (
              <Alert
                className="mt-3"
                type="warning"
                message="任务等待工具审批；当前团队执行暂不支持审批后自动续跑。"
              />
            )}
            <div className="response-content">
              <MarkdownContent>{runResult.response || ""}</MarkdownContent>
            </div>
            <Collapse
              items={[
                {
                  key: "steps",
                  label: "成员执行详情",
                  children: (
                    <pre className="json-view">
                      {JSON.stringify(runResult.messages || [], null, 2)}
                    </pre>
                  ),
                },
              ]}
            />
          </div>
        )}
      </Modal>
    </div>
  );
}
