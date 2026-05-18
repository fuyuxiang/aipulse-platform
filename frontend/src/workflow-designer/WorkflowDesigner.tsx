import { Graph } from '@antv/x6';
import { Button, Space } from 'antd';
import React, { useEffect, useRef } from 'react';

export function WorkflowDesigner(): JSX.Element {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const graphRef = useRef<Graph | null>(null);

  useEffect(() => {
    if (!containerRef.current || graphRef.current) return;
    const graph = new Graph({
      container: containerRef.current,
      grid: true,
      panning: true,
      mousewheel: { enabled: true, modifiers: ['ctrl', 'meta'] },
      connecting: { allowBlank: false, allowLoop: false, highlight: true }
    });
    graph.addNode({ id: 'start', x: 80, y: 80, width: 120, height: 44, label: '开始' });
    graph.addNode({ id: 'agent', x: 290, y: 80, width: 160, height: 44, label: 'Agent 节点' });
    graph.addNode({ id: 'end', x: 540, y: 80, width: 120, height: 44, label: '结束' });
    graph.addEdge({ source: 'start', target: 'agent' });
    graph.addEdge({ source: 'agent', target: 'end' });
    graphRef.current = graph;
  }, []);

  const addNode = (label: string): void => {
    graphRef.current?.addNode({ x: 120 + Math.random() * 360, y: 180 + Math.random() * 240, width: 150, height: 44, label });
  };

  return (
    <div className="mb-5">
      <Space className="mb-3">
        <Button onClick={() => addNode('模型节点')}>模型节点</Button>
        <Button onClick={() => addNode('RAG 节点')}>RAG 节点</Button>
        <Button onClick={() => addNode('人工审批')}>审批节点</Button>
        <Button onClick={() => addNode('条件分支')}>条件分支</Button>
      </Space>
      <div ref={containerRef} className="workflow-canvas" />
    </div>
  );
}
