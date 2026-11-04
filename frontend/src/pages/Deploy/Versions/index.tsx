import { PlusOutlined, ReloadOutlined } from "@ant-design/icons";
import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  message,
  Modal,
  Popconfirm,
  Select,
  Space,
  Table,
  Tag,
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
interface Version extends ResourceRecord {
  version: string;
}
export function VersionsPage(): JSX.Element {
  const { can } = useAccess();
  const [agents, setAgents] = useState<ResourceRecord[]>([]);
  const [agentId, setAgentId] = useState("");
  const [rows, setRows] = useState<Version[]>([]);
  const [activeVersion, setActiveVersion] = useState("");
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form] = Form.useForm();
  useEffect(() => {
    void api
      .list("/agents", 1, 200)
      .then((result) => setAgents(result.items))
      .catch((reason) => setError(reason.message));
  }, []);
  const load = async () => {
    if (!agentId) return;
    setLoading(true);
    setError("");
    try {
      const result = await api.list<Version>(
        `/agents/${agentId}/versions`,
        page,
        20,
      );
      const agent = await api.get<ResourceRecord & { version?: string }>(
        `/agents/${agentId}`,
      );
      setActiveVersion(agent.version || "");
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
  }, [agentId, page]);
  return (
    <div className="enterprise-page">
      <PageHeader
        title="版本发布"
        description="保存版本快照与发布记录。对外调用时，请在 API 发布中选择对应版本。"
        actions={
          <Space>
            <Button
              aria-label="刷新版本"
              icon={<ReloadOutlined />}
              disabled={!agentId}
              onClick={() => void load()}
            />
            {can("agents:write") && (
              <Button
                type="primary"
                icon={<PlusOutlined />}
                disabled={!agentId}
                onClick={() => setOpen(true)}
              >
                创建版本
              </Button>
            )}
          </Space>
        }
      />
      {error && <Alert className="mb-4" type="error" message={error} />}
      <Card className="enterprise-card table-card">
        <div className="list-toolbar">
          <Select
            aria-label="选择版本所属智能体"
            placeholder="选择智能体"
            showSearch
            optionFilterProp="label"
            value={agentId || undefined}
            options={agents.map((agent) => ({
              value: agent.id,
              label: agent.name,
            }))}
            onChange={(id) => {
              setAgentId(id);
              setRows([]);
              setPage(1);
            }}
            style={{ width: 300 }}
          />
        </div>
        <Table<Version>
          rowKey="id"
          loading={loading}
          dataSource={rows}
          pagination={{
            current: page,
            total,
            pageSize: 20,
            showSizeChanger: false,
            onChange: setPage,
          }}
          scroll={{ x: 650 }}
          columns={[
            { title: "版本", dataIndex: "version" },
            { title: "名称", dataIndex: "name" },
            {
              title: "状态",
              render: (_, row) => (
                <>
                  {row.version === activeVersion ? (
                    <Tag color="success">当前发布</Tag>
                  ) : (
                    <StatusTag status={row.status} />
                  )}
                </>
              ),
            },
            {
              title: "创建时间",
              render: (_, row) => formatTime(row.created_at),
            },
            ...(can("agents:write")
              ? [
                  {
                    title: "操作",
                    render: (_: unknown, row: Version) => (
                      <Space>
                        <Popconfirm
                          title={`发布版本 ${row.version}？`}
                          onConfirm={async () => {
                            try {
                              await api.post(`/agents/${agentId}/release`, {
                                version_id: row.id,
                              });
                              message.success("版本已发布");
                              await load();
                            } catch (reason) {
                              message.error((reason as Error).message);
                            }
                          }}
                        >
                          <Button type="link">发布</Button>
                        </Popconfirm>
                        <Popconfirm
                          title={`回滚到版本 ${row.version}？`}
                          onConfirm={async () => {
                            try {
                              await api.post(`/agents/${agentId}/rollback`, {
                                target_version: row.version,
                              });
                              message.success("已回滚");
                              await load();
                            } catch (reason) {
                              message.error((reason as Error).message);
                            }
                          }}
                        >
                          <Button type="link">回滚</Button>
                        </Popconfirm>
                      </Space>
                    ),
                  },
                ]
              : []),
          ]}
          locale={{
            emptyText: agentId
              ? "还没有版本，请先创建版本快照。"
              : "请选择要管理版本的智能体。",
          }}
        />
      </Card>
      <Modal
        open={open}
        title="创建版本快照"
        confirmLoading={saving}
        onCancel={() => !saving && setOpen(false)}
        onOk={async () => {
          try {
            const values = await form.validateFields();
            setSaving(true);
            await api.post(`/agents/${agentId}/versions`, values);
            message.success("版本已创建");
            setOpen(false);
            form.resetFields();
            await load();
          } catch (reason) {
            if (reason instanceof Error) message.error(reason.message);
          } finally {
            setSaving(false);
          }
        }}
      >
        <Form form={form} layout="vertical">
          <Form.Item label="版本号" name="version" rules={[{ required: true }]}>
            <Input placeholder="例如：1.0.0" />
          </Form.Item>
          <Form.Item label="变更说明" name="change_summary">
            <Input.TextArea rows={3} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
