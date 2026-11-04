import {
  ApartmentOutlined,
  ArrowRightOutlined,
  MessageOutlined,
  PlusOutlined,
  ReloadOutlined,
  RobotOutlined,
  TeamOutlined,
} from "@ant-design/icons";
import {
  Alert,
  Badge,
  Button,
  Card,
  Empty,
  List,
  Space,
  Spin,
  Statistic,
  Table,
  Typography,
} from "antd";
import React, { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAccess } from "../../access/context";
import type { ResourceRecord } from "../../models/types";
import { api } from "../../services/http";
import { formatTime, PageHeader, StatusTag } from "../../components/PageHeader";

interface Recent extends ResourceRecord {
  kind: string;
  path: string;
}
export function DashboardPage(): JSX.Element {
  const { user, can } = useAccess();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [recent, setRecent] = useState<Recent[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [errors, setErrors] = useState<string[]>([]);
  const [health, setHealth] = useState<"success" | "error" | "default">(
    "default",
  );
  const [refreshed, setRefreshed] = useState("");
  const load = async () => {
    setLoading(true);
    setErrors([]);
    const sources = [
      {
        key: "agents",
        api: "/agents",
        kind: "智能体",
        path: "/build/agents",
        permission: "agents:read",
      },
      {
        key: "workflows",
        api: "/workflows",
        kind: "工作流",
        path: "/build/workflows",
        permission: "workflows:read",
      },
      {
        key: "teams",
        api: "/agent-teams",
        kind: "团队",
        path: "/playground/multi",
        permission: "agents:read",
      },
      {
        key: "sessions",
        api: "/chat/sessions",
        kind: "对话",
        path: "/playground",
        permission: "chat:read",
      },
    ].filter((source) => can(source.permission));
    const items: Recent[] = [];
    const metrics: Record<string, number> = {};
    const failures: string[] = [];
    const results = await Promise.allSettled(
      sources.map((source) => api.list(source.api, 1, 5)),
    );
    results.forEach((result, index) => {
      const source = sources[index];
      if (result.status === "fulfilled") {
        metrics[source.key] = result.value.total;
        items.push(
          ...result.value.items.map((row) => ({
            ...row,
            kind: source.kind,
            path: source.path,
          })),
        );
      } else failures.push(`${source.kind}：${result.reason.message}`);
    });
    setRecent(
      items
        .sort((a, b) => b.created_at.localeCompare(a.created_at))
        .slice(0, 7),
    );
    setCounts(metrics);
    setErrors(failures);
    try {
      const response = await fetch("/health");
      const data = await response.json();
      setHealth(response.ok && data.status === "healthy" ? "success" : "error");
    } catch {
      setHealth("error");
    }
    setRefreshed(new Date().toLocaleTimeString("zh-CN", { hour12: false }));
    setLoading(false);
  };
  useEffect(() => {
    void load();
  }, []);
  const entries = [
    {
      title: "开始对话",
      description: "选择智能体，处理当前工作",
      icon: <MessageOutlined />,
      path: "/playground",
      permission: "chat:read",
    },
    {
      title: "构建智能体",
      description: "配置模型、提示词和工具",
      icon: <RobotOutlined />,
      path: "/build/agents/create",
      permission: "agents:write",
    },
    {
      title: "编排工作流",
      description: "将多个步骤连接成业务流程",
      icon: <ApartmentOutlined />,
      path: "/build/workflows/designer",
      permission: "workflows:write",
    },
    {
      title: "团队协作",
      description: "组织多个智能体共同执行任务",
      icon: <TeamOutlined />,
      path: "/playground/multi",
      permission: "agents:read",
    },
  ].filter((entry) => can(entry.permission));
  const metrics = [
    { key: "agents", label: "智能体", path: "/build/agents" },
    { key: "workflows", label: "工作流", path: "/build/workflows" },
    { key: "teams", label: "协作团队", path: "/playground/multi" },
    { key: "sessions", label: "我的对话", path: "/playground" },
  ].filter(
    (metric) =>
      ({
        agents: can("agents:read"),
        workflows: can("workflows:read"),
        teams: can("agents:read"),
        sessions: can("chat:read"),
      })[metric.key],
  );
  return (
    <div className="enterprise-page workspace-page">
      <PageHeader
        title="工作台"
        description={`欢迎回来，${user?.display_name || user?.username}。从最近的工作继续，或开始一个新任务。`}
        actions={
          <Button
            icon={<ReloadOutlined />}
            loading={loading}
            onClick={() => void load()}
          >
            刷新
          </Button>
        }
      />
      {errors.length > 0 && (
        <Alert
          className="mb-4"
          type="warning"
          showIcon
          message="部分数据未能加载"
          description={errors.join("；")}
          action={<Button onClick={() => void load()}>重试</Button>}
        />
      )}
      <div className="workspace-entries">
        {entries.map((entry) => (
          <button
            className="workspace-entry"
            key={entry.path}
            onClick={() => navigate(entry.path)}
          >
            <span className="workspace-entry-icon">{entry.icon}</span>
            <span>
              <strong>{entry.title}</strong>
              <small>{entry.description}</small>
            </span>
            <ArrowRightOutlined />
          </button>
        ))}
      </div>
      {!!metrics.length && (
        <div className="workspace-metrics">
          {metrics.map((metric) => (
            <Card key={metric.key} className="enterprise-card" size="small">
              <Statistic
                title={metric.label}
                value={counts[metric.key] ?? "—"}
                loading={loading}
              />
              <Link to={metric.path}>
                查看全部 <ArrowRightOutlined />
              </Link>
            </Card>
          ))}
        </div>
      )}
      <div className="workspace-columns">
        <Card
          className="enterprise-card"
          title="最近创建"
          extra={<span className="secondary-text">当前账号可访问的资源</span>}
        >
          <Table<Recent>
            rowKey={(row) => `${row.kind}-${row.id}`}
            size="middle"
            loading={loading}
            dataSource={recent}
            pagination={false}
            scroll={{ x: 480 }}
            columns={[
              {
                title: "名称",
                dataIndex: "name",
                render: (value, row) => (
                  <Button type="link" onClick={() => navigate(row.path)}>
                    {value || "未命名"}
                  </Button>
                ),
              },
              { title: "类型", dataIndex: "kind", width: 80 },
              {
                title: "状态",
                width: 110,
                render: (_, row) => <StatusTag status={row.status} />,
              },
              {
                title: "创建时间",
                width: 170,
                render: (_, row) => (
                  <span className="secondary-text">
                    {formatTime(row.created_at)}
                  </span>
                ),
              },
            ]}
            locale={{
              emptyText: (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="还没有工作记录"
                >
                  {can("agents:write") ? (
                    <Button
                      type="primary"
                      icon={<PlusOutlined />}
                      onClick={() => navigate("/build/agents/create")}
                    >
                      创建第一个智能体
                    </Button>
                  ) : (
                    <Typography.Text type="secondary">
                      联系管理员配置智能体与访问权限。
                    </Typography.Text>
                  )}
                </Empty>
              ),
            }}
          />
        </Card>
        <div className="workspace-side">
          <Card className="enterprise-card" title="开始使用">
            <List
              size="small"
              dataSource={[
                ...(can("models:write")
                  ? [
                      {
                        title: "接入与验证模型",
                        path: "/settings/models",
                        text: "配置真实服务，或使用本地演示模型",
                      },
                    ]
                  : []),
                ...(can("agents:write")
                  ? [
                      {
                        title: "配置智能体",
                        path: "/build/agents/create",
                        text: "选择模型，描述角色与行为",
                      },
                    ]
                  : []),
                ...(can("chat:read")
                  ? [
                      {
                        title: "开始一次对话",
                        path: "/playground",
                        text: "验证回答，并保留对话记录",
                      },
                    ]
                  : []),
              ]}
              renderItem={(item, index) => (
                <List.Item>
                  <Link className="getting-started-item" to={item.path}>
                    <span>{index + 1}</span>
                    <div>
                      <strong>{item.title}</strong>
                      <small>{item.text}</small>
                    </div>
                    <ArrowRightOutlined />
                  </Link>
                </List.Item>
              )}
              locale={{ emptyText: "暂无可用入口，请联系管理员分配权限。" }}
            />
          </Card>
          <Card className="enterprise-card" size="small">
            <div className="workspace-status">
              <Badge
                status={health}
                text={
                  health === "success"
                    ? "平台服务正常"
                    : health === "error"
                      ? "服务状态异常"
                      : "检测中"
                }
              />
              <span className="secondary-text">本地开发版</span>
            </div>
            <div className="secondary-text mt-3">
              {refreshed ? `最近刷新 ${refreshed}` : <Spin size="small" />}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
