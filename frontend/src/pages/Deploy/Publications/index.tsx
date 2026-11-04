import { PlusOutlined, ReloadOutlined } from "@ant-design/icons";
import {
  Alert,
  Button,
  Card,
  Form,
  Drawer,
  Input,
  message,
  Modal,
  Popconfirm,
  Select,
  Space,
  Table,
  Typography,
} from "antd";
import React, { useEffect, useState } from "react";
import { useAccess } from "../../../access/context";
import {
  formatTime,
  PageHeader,
  StatusTag,
} from "../../../components/PageHeader";
import type { ResourceRecord } from "../../../models/types";
import { api } from "../../../services/http";
export function PublicationsPage(): JSX.Element {
  const { can } = useAccess();
  const writable = can("agents:write");
  const [rows, setRows] = useState<ResourceRecord[]>([]);
  const [agents, setAgents] = useState<ResourceRecord[]>([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(false);
  const [credential, setCredential] = useState<{
    api_key: string;
    endpoint: string;
  } | null>(null);
  const [form] = Form.useForm();
  const selectedAgent = Form.useWatch("agent_id", form);
  const [versions, setVersions] = useState<
    Array<ResourceRecord & { version: string }>
  >([]);
  const [keyDetails, setKeyDetails] = useState<{
    name: string;
    keys: Array<Record<string, any>>;
  } | null>(null);
  useEffect(() => {
    let active = true;
    setVersions([]);
    if (open && selectedAgent)
      void api
        .list<ResourceRecord & { version: string }>(
          `/agents/${selectedAgent}/versions`,
          1,
          200,
        )
        .then((result) => {
          if (active) setVersions(result.items);
        })
        .catch((reason) => message.error(reason.message));
    return () => {
      active = false;
    };
  }, [open, selectedAgent]);
  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const result = await api.list("/agent-publications", page, 20);
      setRows(result.items);
      setTotal(result.total);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    void load();
  }, [page]);
  useEffect(() => {
    void api
      .list("/agents", 1, 200)
      .then((result) => setAgents(result.items))
      .catch((reason) => setError(reason.message));
  }, []);
  const create = async () => {
    try {
      const values = await form.validateFields();
      if (
        values.expires_at &&
        new Date(values.expires_at).getTime() <= Date.now()
      )
        throw new Error("有效期必须晚于当前时间");
      setSaving(true);
      const result = await api.post<{ api_key: string; endpoint: string }>(
        `/agents/${values.agent_id}/publish-api`,
        {
          name: values.name,
          permissions: values.permissions,
          version_id: values.version_id || "",
          expires_at: values.expires_at
            ? new Date(values.expires_at).toISOString()
            : "",
        },
      );
      setOpen(false);
      form.resetFields();
      setCredential(result);
      await load();
    } catch (reason) {
      if (reason instanceof Error) message.error(reason.message);
    } finally {
      setSaving(false);
    }
  };
  return (
    <div className="enterprise-page">
      <PageHeader
        title="API 发布与凭证"
        description="将智能体配置发布为调用快照，管理访问状态、调用权限与 API 密钥。"
        actions={
          <Space>
            <Button
              aria-label="刷新 API 发布"
              icon={<ReloadOutlined />}
              onClick={() => void load()}
            />
            {writable && (
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setOpen(true)}
              >
                发布 API
              </Button>
            )}
          </Space>
        }
      />
      {error && (
        <Alert className="mb-4" type="error" message={error} showIcon />
      )}
      <Card className="enterprise-card table-card">
        <Table<ResourceRecord>
          rowKey="id"
          loading={loading}
          dataSource={rows}
          scroll={{ x: 850 }}
          pagination={{
            current: page,
            total,
            pageSize: 20,
            showSizeChanger: false,
            onChange: setPage,
          }}
          columns={[
            { title: "发布名称", dataIndex: "name" },
            {
              title: "智能体",
              render: (_, row) =>
                agents.find((agent) => agent.id === row.agent_id)?.name ||
                row.agent_id?.slice(0, 8),
            },
            {
              title: "状态",
              render: (_, row) => (
                <StatusTag
                  status={
                    row.enabled && row.status === "active"
                      ? "active"
                      : "disabled"
                  }
                />
              ),
            },
            {
              title: "调用地址",
              render: (_, row) => (
                <Typography.Text code copyable>
                  {String(row.spec?.endpoint || "")}
                </Typography.Text>
              ),
            },
            {
              title: "发布时间",
              render: (_, row) => formatTime(row.created_at),
            },
            {
              title: "凭证",
              render: (_, row) => (
                <Button
                  type="link"
                  onClick={() =>
                    void api
                      .get<Array<Record<string, any>>>(
                        `/agent-publications/${row.id}/keys`,
                      )
                      .then((keys) => setKeyDetails({ name: row.name, keys }))
                      .catch((reason) => message.error(reason.message))
                  }
                >
                  查看凭证
                </Button>
              ),
            },
            ...(writable
              ? [
                  {
                    title: "操作",
                    render: (_: unknown, row: ResourceRecord) => (
                      <Space>
                        <Button
                          type="link"
                          onClick={async () => {
                            try {
                              await api.update("/agent-publications", row.id, {
                                enabled: !row.enabled,
                                status: row.enabled ? "disabled" : "active",
                              });
                              await load();
                            } catch (reason) {
                              message.error((reason as Error).message);
                            }
                          }}
                        >
                          {row.enabled ? "停用" : "启用"}
                        </Button>
                        <Popconfirm
                          title="轮换密钥？"
                          description="旧密钥将立即失效。"
                          onConfirm={async () => {
                            try {
                              const result = await api.post<{
                                api_key: string;
                              }>(
                                `/agent-publications/${row.id}/rotate-key`,
                                {},
                              );
                              setCredential({
                                api_key: result.api_key,
                                endpoint: String(row.spec?.endpoint || ""),
                              });
                              await load();
                            } catch (reason) {
                              message.error((reason as Error).message);
                            }
                          }}
                        >
                          <Button type="link">轮换密钥</Button>
                        </Popconfirm>
                      </Space>
                    ),
                  },
                ]
              : []),
          ]}
          locale={{ emptyText: "暂无 API 发布。发布后可获取访问凭证。" }}
        />
      </Card>
      <Drawer
        open={!!keyDetails}
        title={`${keyDetails?.name || ""} · 访问凭证`}
        width={680}
        onClose={() => setKeyDetails(null)}
      >
        <Table
          rowKey="id"
          dataSource={keyDetails?.keys || []}
          pagination={false}
          columns={[
            {
              title: "密钥前缀",
              render: (_, row) => row.spec?.key_prefix || "—",
            },
            {
              title: "状态",
              render: (_, row) => <StatusTag status={row.status} />,
            },
            {
              title: "权限",
              render: (_, row) => (row.spec?.permissions || []).join("、"),
            },
            {
              title: "有效期",
              render: (_, row) =>
                row.spec?.expires_at
                  ? formatTime(row.spec.expires_at)
                  : "不过期",
            },
          ]}
        />
      </Drawer>
      <Modal
        open={open}
        title="发布智能体 API"
        onCancel={() => !saving && setOpen(false)}
        onOk={() => void create()}
        confirmLoading={saving}
        okText="发布"
      >
        <Form
          form={form}
          layout="vertical"
          initialValues={{ permissions: ["chat"] }}
        >
          <Form.Item
            name="name"
            label="发布名称"
            rules={[{ required: true, whitespace: true }]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="agent_id"
            label="智能体"
            rules={[{ required: true }]}
          >
            <Select
              showSearch
              optionFilterProp="label"
              onChange={() => form.setFieldValue("version_id", undefined)}
              options={agents.map((agent) => ({
                value: agent.id,
                label: agent.name,
              }))}
            />
          </Form.Item>
          <Form.Item name="version_id" label="发布版本">
            <Select
              allowClear
              placeholder="当前草稿快照"
              options={versions.map((version) => ({
                value: version.id,
                label: version.version,
              }))}
            />
          </Form.Item>
          <Form.Item
            name="permissions"
            label="允许调用"
            rules={[{ required: true }]}
          >
            <Select
              mode="multiple"
              options={[
                { value: "chat", label: "对话" },
                { value: "stream", label: "流式对话" },
              ]}
            />
          </Form.Item>
          <Form.Item name="expires_at" label="有效期（留空不过期）">
            <Input type="datetime-local" />
          </Form.Item>
          <Alert
            type="info"
            message="发布保存当前智能体配置快照。修改智能体后，需要重新发布才会更新该快照。"
          />
        </Form>
      </Modal>
      <Modal
        open={!!credential}
        title="保存 API 密钥"
        onCancel={() => setCredential(null)}
        footer={
          <Button type="primary" onClick={() => setCredential(null)}>
            已保存，关闭
          </Button>
        }
      >
        <Alert
          type="info"
          showIcon
          message="完整密钥仅在本次显示，请保存到安全位置。"
        />
        <div className="mt-4">
          <Input.Password
            aria-label="API 密钥"
            value={credential?.api_key}
            readOnly
          />
        </div>
        <Button
          className="mt-3"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(credential?.api_key || "");
              message.success("密钥已复制");
            } catch {
              message.error("复制失败，请手动保存");
            }
          }}
        >
          复制密钥
        </Button>
        <p className="mt-4 secondary-text">
          请求时使用 X-API-Key 请求头。调用地址：
        </p>
        <Typography.Text code>{credential?.endpoint}</Typography.Text>
      </Modal>
    </div>
  );
}
