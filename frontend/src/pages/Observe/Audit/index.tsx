import { Card, Input, Space, Table, Tag, Typography, Button, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { ReloadOutlined, SafetyCertificateOutlined } from '@ant-design/icons';
import React, { useEffect, useState } from 'react';
import { api } from '../../../services/http';

interface AuditLog {
  id: string;
  name: string;
  code: string;
  status: string;
  user_id: string;
  agent_id: string;
  trace_id: string;
  description: string;
  metadata_json?: Record<string, unknown>;
  created_at: string;
}

export function AuditPage(): JSX.Element {
  const [rows, setRows] = useState<AuditLog[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [keyword, setKeyword] = useState('');
  const [integrity, setIntegrity] = useState<{ verified?: boolean; tip?: string } | null>(null);

  const load = async (page = 1): Promise<void> => {
    setLoading(true);
    try {
      const res = await api.list<AuditLog>('/audit-logs', page, 50);
      const list = keyword ? res.items.filter((r) => `${r.name}${r.code}${r.user_id}${r.trace_id}`.includes(keyword)) : res.items;
      setRows(list);
      setTotal(res.total);
    } catch (e) {
      message.error(e instanceof Error ? e.message : '加载失败');
    } finally {
      setLoading(false);
    }
  };

  const verify = async (): Promise<void> => {
    try {
      const res = await api.get<{ verified: boolean; tip?: string }>('/audit-integrity/verify');
      setIntegrity(res);
      message[res.verified ? 'success' : 'error'](res.verified ? '审计链完整' : '检测到链断裂');
    } catch (e) {
      message.error(e instanceof Error ? e.message : '完整性校验失败');
    }
  };

  useEffect(() => { void load(); }, []);

  const columns: ColumnsType<AuditLog> = [
    { title: '时间', dataIndex: 'created_at', width: 190, render: (v) => <Typography.Text type="secondary">{v || '-'}</Typography.Text> },
    { title: '动作', dataIndex: 'code', width: 220, render: (v) => <Tag color="processing">{v || '-'}</Tag> },
    { title: '主体', dataIndex: 'user_id', width: 160 },
    { title: '对象', dataIndex: 'agent_id', width: 220, render: (v, r) => v || r.name || '-' },
    { title: '描述', dataIndex: 'description', ellipsis: true },
    { title: '链路 ID', dataIndex: 'trace_id', width: 200, render: (v) => <Typography.Text code>{v || '-'}</Typography.Text> },
  ];

  return (
    <div className="enterprise-page">
      <section className="resource-hero">
        <div>
          <Space wrap><Tag color="processing">审计与合规</Tag><Tag>观测</Tag></Space>
          <Typography.Title level={1}>审计日志</Typography.Title>
          <Typography.Paragraph>追踪关键操作、敏感动作、发布变更和合规证据。审计链使用 hash chain 防篡改。</Typography.Paragraph>
        </div>
        <Space>
          <Button icon={<SafetyCertificateOutlined />} onClick={() => void verify()}>验证审计链</Button>
          <Button icon={<ReloadOutlined />} onClick={() => void load()}>刷新</Button>
        </Space>
      </section>

      {integrity ? (
        <Card bordered={false} className="enterprise-card" style={{ marginBottom: 16 }}>
          <Space>
            <Tag color={integrity.verified ? 'success' : 'error'}>{integrity.verified ? '链完整' : '链断裂'}</Tag>
            <Typography.Text type="secondary">{integrity.tip || '审计 hash chain 校验结果'}</Typography.Text>
          </Space>
        </Card>
      ) : null}

      <Card bordered={false} className="enterprise-card">
        <Space style={{ marginBottom: 12 }}>
          <Input.Search placeholder="按动作 / 主体 / trace_id 搜索" value={keyword} onChange={(e) => setKeyword(e.target.value)} onSearch={() => void load()} style={{ width: 360 }} />
          <Typography.Text type="secondary">共 {total} 条</Typography.Text>
        </Space>
        <Table<AuditLog>
          rowKey="id"
          loading={loading}
          dataSource={rows}
          columns={columns}
          pagination={{ total, pageSize: 50, onChange: (p) => void load(p) }}
        />
      </Card>
    </div>
  );
}
