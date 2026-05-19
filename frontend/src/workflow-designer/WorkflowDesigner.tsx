import { Graph } from '@antv/x6';
import { Button, Input, Space, message } from 'antd';
import React, { useEffect, useRef, useState } from 'react';
import { api } from '../services/http';

interface WorkflowDesignerProps {
  onSaved?: () => void;
}

export function WorkflowDesigner({ onSaved }: WorkflowDesignerProps): JSX.Element {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const graphRef = useRef<Graph | null>(null);
  const [name, setName] = useState('新建工作流');
  const [workflowId, setWorkflowId] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!containerRef.current || graphRef.current) return;
    const graph = new Graph({
      container: containerRef.current,
      grid: true,
      panning: true,
      mousewheel: { enabled: true, modifiers: ['ctrl', 'meta'] },
      connecting: { allowBlank: false, allowLoop: false, highlight: true }
    });
    graph.addNode({ id: 'start', x: 80, y: 80, width: 120, height: 44, label: '开始', data: { type: 'start', config: {} } });
    graph.addNode({ id: 'agent', x: 290, y: 80, width: 160, height: 44, label: 'Agent 节点', data: { type: 'agent', config: { agent_id: '' } } });
    graph.addNode({ id: 'end', x: 540, y: 80, width: 120, height: 44, label: '结束', data: { type: 'end', config: {} } });
    graph.addEdge({ source: 'start', target: 'agent' });
    graph.addEdge({ source: 'agent', target: 'end' });
    graphRef.current = graph;
  }, []);

  const addNode = (type: string, label: string): void => {
    graphRef.current?.addNode({
      id: `${type}-${Date.now()}`,
      x: 120 + Math.random() * 360,
      y: 180 + Math.random() * 240,
      width: 150,
      height: 44,
      label,
      data: { type, config: defaultConfig(type) }
    });
  };

  const serialize = (): { nodes: Array<Record<string, unknown>>; edges: Array<Record<string, string>> } => {
    const cells = (graphRef.current?.toJSON().cells || []) as Array<Record<string, any>>;
    const nodes = cells.filter((cell) => cell.shape !== 'edge').map((cell) => ({
      id: String(cell.id),
      type: String(cell.data?.type || inferType(String(cell.label || ''))),
      label: String(cell.label || cell.id),
      config: cell.data?.config || {},
      position: cell.position || { x: 0, y: 0 },
    }));
    const edges = cells.filter((cell) => cell.shape === 'edge').map((cell) => ({
      source: String(cell.source?.cell || cell.source),
      target: String(cell.target?.cell || cell.target),
    })).filter((edge) => edge.source && edge.target);
    return { nodes, edges };
  };

  const save = async (): Promise<void> => {
    setSaving(true);
    try {
      const definition = serialize();
      const row = await api.create('/workflows', {
        name,
        status: 'draft',
        spec: definition,
        config: definition,
      });
      setWorkflowId(row.id);
      message.success('工作流已保存');
      onSaved?.();
    } catch (error) {
      message.error(error instanceof Error ? error.message : '工作流保存失败');
    } finally {
      setSaving(false);
    }
  };

  const validate = async (): Promise<void> => {
    if (!workflowId) {
      message.warning('请先保存工作流');
      return;
    }
    try {
      const result = await api.post<{ valid?: boolean; error?: string }>(`/workflows/${workflowId}/validate`, serialize());
      if (result.valid) {
        message.success('工作流校验通过');
      } else {
        message.error(result.error || '工作流校验失败');
      }
    } catch (error) {
      message.error(error instanceof Error ? error.message : '工作流校验失败');
    }
  };

  return (
    <div className="mb-5">
      <Space className="mb-3">
        <Input value={name} onChange={(event) => setName(event.target.value)} className="w-64" />
        <Button onClick={() => addNode('model', '模型节点')}>模型节点</Button>
        <Button onClick={() => addNode('rag', 'RAG 节点')}>RAG 节点</Button>
        <Button onClick={() => addNode('human_approval', '人工审批')}>审批节点</Button>
        <Button onClick={() => addNode('condition', '条件分支')}>条件分支</Button>
        <Button type="primary" loading={saving} onClick={() => void save()}>保存定义</Button>
        <Button disabled={!workflowId} onClick={() => void validate()}>校验</Button>
      </Space>
      <div ref={containerRef} className="workflow-canvas" />
    </div>
  );
}

function defaultConfig(type: string): Record<string, unknown> {
  if (type === 'model') return { model_id: '', model_type: 'chat_llm', payload: {} };
  if (type === 'rag') return { knowledge_base_id: '', payload: {} };
  if (type === 'human_approval') return { approvers: [] };
  if (type === 'condition') return { left: 'input', operator: 'equals', right_value: '' };
  return {};
}

function inferType(label: string): string {
  if (label.includes('开始')) return 'start';
  if (label.includes('结束')) return 'end';
  if (label.includes('RAG')) return 'rag';
  if (label.includes('审批')) return 'human_approval';
  if (label.includes('条件')) return 'condition';
  if (label.includes('模型')) return 'model';
  return 'agent';
}
