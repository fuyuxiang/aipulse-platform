import { Graph } from '@antv/x6';
import { Button, Card, Drawer, Form, Input, Modal, Select, Space, Table, Tag, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import React, { useEffect, useRef, useState } from 'react';
import { api } from '../../../services/http';

interface Team {
  id: string;
  name: string;
  spec?: { topology?: string; coordinator_agent_id?: string; max_rounds?: number };
  members?: Array<{ id: string; agent_id: string; spec?: { role?: string; capabilities?: string[] } }>;
}

export function MultiAgentPage(): JSX.Element {
  const [teams, setTeams] = useState<Team[]>([]);
  const [loading, setLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
  const [runModalOpen, setRunModalOpen] = useState(false);
  const [runPrompt, setRunPrompt] = useState('');
  const [runResult, setRunResult] = useState<any>(null);
  const [form] = Form.useForm();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const graphRef = useRef<Graph | null>(null);

  const loadTeams = async (): Promise<void> => {
    setLoading(true);
    try {
      const res = await api.list('/agent-teams', 1, 50);
      setTeams(res.items as any[] || []);
    } catch { /* ignore */ }
    setLoading(false);
  };

  useEffect(() => { void loadTeams(); }, []);

  const createTeam = async (): Promise<void> => {
    try {
      const values = await form.validateFields();
      await api.create('/agent-teams', {
        name: values.name,
        topology: values.topology,
        coordinator_agent_id: values.coordinator_agent_id || '',
        max_rounds: values.max_rounds || 10,
        delegation_strategy: values.delegation_strategy || 'auto',
      } as any);
      message.success('团队创建成功');
      setDrawerOpen(false);
      form.resetFields();
      await loadTeams();
    } catch { message.error('创建失败'); }
  };

  const viewTeam = async (team: Team): Promise<void> => {
    try {
      const res = await fetch(`/api/v1/agent-teams/${team.id}`, {
        headers: { Authorization: `Bearer ${localStorage.getItem('aipulse_access_token') || ''}` },
      });
      if (res.ok) {
        const data = await res.json();
        setSelectedTeam(data);
        renderTopology(data);
      }
    } catch { /* ignore */ }
  };

  const renderTopology = (team: Team): void => {
    if (!containerRef.current) return;
    if (graphRef.current) graphRef.current.dispose();
    const graph = new Graph({
      container: containerRef.current,
      grid: true,
      panning: true,
      mousewheel: { enabled: true, modifiers: ['ctrl', 'meta'] },
    });

    const members = team.members || [];
    const topology = team.spec?.topology || 'star';
    const centerX = 300;
    const centerY = 200;

    if (topology === 'star') {
      graph.addNode({ id: 'coordinator', x: centerX - 60, y: centerY - 22, width: 140, height: 44, label: '协调者', attrs: { body: { fill: '#e6f7ff', stroke: '#1890ff' } } });
      members.forEach((m, i) => {
        const angle = (2 * Math.PI * i) / Math.max(members.length, 1);
        const x = centerX + 180 * Math.cos(angle) - 60;
        const y = centerY + 150 * Math.sin(angle) - 22;
        graph.addNode({ id: m.id, x, y, width: 120, height: 44, label: `${m.spec?.role || 'worker'}\n${m.agent_id?.slice(0, 8) || ''}` });
        graph.addEdge({ source: 'coordinator', target: m.id });
      });
    } else if (topology === 'pipeline') {
      members.forEach((m, i) => {
        graph.addNode({ id: m.id, x: 60 + i * 180, y: centerY - 22, width: 140, height: 44, label: `Step ${i + 1}\n${m.spec?.role || 'worker'}` });
        if (i > 0) graph.addEdge({ source: members[i - 1].id, target: m.id });
      });
    } else {
      members.forEach((m, i) => {
        const angle = (2 * Math.PI * i) / Math.max(members.length, 1);
        const x = centerX + 150 * Math.cos(angle) - 60;
        const y = centerY + 120 * Math.sin(angle) - 22;
        graph.addNode({ id: m.id, x, y, width: 120, height: 44, label: m.spec?.role || 'agent' });
      });
      for (let i = 0; i < members.length; i++) {
        for (let j = i + 1; j < members.length; j++) {
          graph.addEdge({ source: members[i].id, target: members[j].id });
        }
      }
    }
    graphRef.current = graph;
  };

  const runTeam = async (): Promise<void> => {
    if (!selectedTeam || !runPrompt.trim()) return;
    try {
      const res = await api.post<any>(`/agent-teams/${selectedTeam.id}/run`, { prompt: runPrompt });
      setRunResult(res);
      message.success('执行完成');
    } catch { message.error('执行失败'); }
  };

  const columns: ColumnsType<Team> = [
    { title: '名称', dataIndex: 'name' },
    { title: '拓扑', render: (_, r) => <Tag color="blue">{r.spec?.topology || 'star'}</Tag> },
    { title: '最大轮次', render: (_, r) => r.spec?.max_rounds || 10 },
    {
      title: '操作', render: (_, r) => (
        <Space>
          <Button size="small" onClick={() => void viewTeam(r)}>查看拓扑</Button>
          <Button size="small" type="primary" onClick={() => { setSelectedTeam(r); setRunModalOpen(true); }}>执行</Button>
        </Space>
      ),
    },
  ];

  return (
    <div className="p-5">
      <div className="mb-4 flex justify-between">
        <h3>多智能体协同</h3>
        <Button type="primary" onClick={() => setDrawerOpen(true)}>创建团队</Button>
      </div>
      <Table rowKey="id" columns={columns} dataSource={teams} loading={loading} pagination={{ pageSize: 20 }} />

      {selectedTeam && (
        <Card title={`拓扑视图: ${selectedTeam.name}`} className="mt-4">
          <div ref={containerRef} style={{ height: 400, border: '1px solid #f0f0f0' }} />
        </Card>
      )}

      <Drawer open={drawerOpen} title="创建 Agent 团队" width={480} onClose={() => setDrawerOpen(false)} extra={<Button type="primary" onClick={() => void createTeam()}>创建</Button>}>
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="团队名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="topology" label="通信拓扑" initialValue="star">
            <Select options={[{ value: 'star', label: 'Star (协调者模式)' }, { value: 'pipeline', label: 'Pipeline (流水线)' }, { value: 'mesh', label: 'Mesh (全连接)' }]} />
          </Form.Item>
          <Form.Item name="coordinator_agent_id" label="协调者 Agent ID"><Input /></Form.Item>
          <Form.Item name="max_rounds" label="最大轮次" initialValue={10}><Input type="number" /></Form.Item>
          <Form.Item name="delegation_strategy" label="委派策略" initialValue="auto">
            <Select options={[{ value: 'auto', label: '自动' }, { value: 'round_robin', label: '轮询' }, { value: 'capability', label: '能力匹配' }]} />
          </Form.Item>
        </Form>
      </Drawer>

      <Modal open={runModalOpen} title="执行多智能体任务" onCancel={() => { setRunModalOpen(false); setRunResult(null); }} footer={null} width={700}>
        <Input.TextArea rows={3} value={runPrompt} onChange={(e) => setRunPrompt(e.target.value)} placeholder="输入任务描述..." />
        <Button type="primary" className="mt-3" onClick={() => void runTeam()}>执行</Button>
        {runResult && (
          <Card className="mt-3" size="small" title="执行结果">
            <pre className="text-xs whitespace-pre-wrap max-h-60 overflow-y-auto">{JSON.stringify(runResult, null, 2)}</pre>
          </Card>
        )}
      </Modal>
    </div>
  );
}
