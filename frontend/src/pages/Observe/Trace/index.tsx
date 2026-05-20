import { Button, Card, Input, Space, Table, Tag, Tooltip } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import React, { useEffect, useState } from 'react';
import { getToken } from '../../../services/http';

interface TraceSpan {
  id: string;
  name: string;
  spec?: {
    trace_id?: string;
    span_id?: string;
    parent_span_id?: string;
    service?: string;
    operation?: string;
    status?: string;
    duration_ms?: number;
    start_time?: string;
    end_time?: string;
    attributes?: Record<string, any>;
    events?: Array<{ name: string; timestamp: string; attributes?: Record<string, any> }>;
  };
  trace_id?: string;
  latency_ms?: number;
  status?: string;
  created_at?: string;
}

export function TraceVisualizationPage(): JSX.Element {
  const [traces, setTraces] = useState<TraceSpan[]>([]);
  const [selectedTrace, setSelectedTrace] = useState<TraceSpan[] | null>(null);
  const [selectedTraceId, setSelectedTraceId] = useState('');
  const [loading, setLoading] = useState(false);
  const [searchId, setSearchId] = useState('');

  const headers = { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' };

  const loadTraces = async (): Promise<void> => {
    setLoading(true);
    try {
      const res = await fetch('/api/v1/observability/traces?page=1&page_size=50', { headers });
      if (res.ok) {
        const data = await res.json();
        setTraces(data.items || []);
      }
    } catch { /* ignore */ }
    setLoading(false);
  };

  useEffect(() => { void loadTraces(); }, []);

  const loadTraceDetail = async (traceId: string): Promise<void> => {
    try {
      const res = await fetch(`/api/v1/observability/traces/${traceId}`, { headers });
      if (res.ok) {
        const data = await res.json();
        setSelectedTrace(data.spans || []);
        setSelectedTraceId(traceId);
      }
    } catch { /* ignore */ }
  };

  const searchTrace = async (): Promise<void> => {
    if (searchId.trim()) {
      await loadTraceDetail(searchId.trim());
    }
  };

  const getSpanColor = (status: string): string => {
    if (status === 'error' || status === 'failed') return '#ff4d4f';
    if (status === 'warning') return '#faad14';
    return '#52c41a';
  };

  const getSpanTypeColor = (name: string): string => {
    if (name.includes('llm') || name.includes('model')) return 'purple';
    if (name.includes('tool')) return 'cyan';
    if (name.includes('rag') || name.includes('retriev')) return 'orange';
    if (name.includes('guardrail')) return 'red';
    return 'blue';
  };

  const renderWaterfall = (spans: TraceSpan[]): JSX.Element => {
    if (!spans.length) return <div className="text-gray-400 text-center py-8">无 Span 数据</div>;

    const sortedSpans = [...spans].sort((a, b) => {
      const aTime = a.spec?.start_time || a.created_at || '';
      const bTime = b.spec?.start_time || b.created_at || '';
      return aTime.localeCompare(bTime);
    });

    const minTime = sortedSpans[0]?.spec?.start_time || sortedSpans[0]?.created_at || '';
    const maxDuration = Math.max(...sortedSpans.map((s) => s.spec?.duration_ms || s.latency_ms || 50));
    const totalDuration = maxDuration * sortedSpans.length * 0.3 || 1000;

    return (
      <div className="border rounded p-4">
        <div className="flex justify-between text-xs text-gray-400 mb-2">
          <span>0ms</span>
          <span>{totalDuration.toFixed(0)}ms</span>
        </div>
        {sortedSpans.map((span, index) => {
          const duration = span.spec?.duration_ms || span.latency_ms || 50;
          const widthPercent = Math.max(3, (duration / totalDuration) * 100);
          const offsetPercent = (index * 5) % 60;
          const depth = span.spec?.parent_span_id ? 1 : 0;
          const status = span.spec?.status || span.status || 'ok';

          return (
            <Tooltip
              key={span.id}
              title={
                <div>
                  <div>{span.name}</div>
                  <div>Duration: {duration}ms</div>
                  <div>Status: {status}</div>
                  {span.spec?.attributes && Object.entries(span.spec.attributes).slice(0, 5).map(([k, v]) => (
                    <div key={k}>{k}: {String(v).slice(0, 50)}</div>
                  ))}
                </div>
              }
            >
              <div className="flex items-center mb-1 hover:bg-gray-50 rounded px-1" style={{ paddingLeft: `${depth * 24}px` }}>
                <div className="w-32 text-xs truncate flex-shrink-0">
                  <Tag color={getSpanTypeColor(span.name)} className="text-[10px]">{span.name}</Tag>
                </div>
                <div className="flex-1 h-6 relative">
                  <div
                    className="absolute h-5 rounded"
                    style={{
                      left: `${offsetPercent}%`,
                      width: `${widthPercent}%`,
                      backgroundColor: getSpanColor(status),
                      opacity: 0.8,
                      top: '2px',
                    }}
                  />
                </div>
                <div className="w-16 text-xs text-right flex-shrink-0 text-gray-500">{duration}ms</div>
              </div>
            </Tooltip>
          );
        })}
      </div>
    );
  };

  const traceColumns: ColumnsType<TraceSpan> = [
    { title: 'Trace ID', dataIndex: 'trace_id', render: (v) => <code className="text-xs">{v?.slice(0, 16) || '-'}</code> },
    { title: '名称', dataIndex: 'name' },
    { title: '延迟', dataIndex: 'latency_ms', render: (v) => `${v || 0}ms`, sorter: (a, b) => (a.latency_ms || 0) - (b.latency_ms || 0) },
    { title: '状态', dataIndex: 'status', render: (v: string) => <Tag color={v === 'error' ? 'red' : 'green'}>{v}</Tag> },
    { title: '时间', dataIndex: 'created_at', render: (v) => v?.slice(0, 19) || '' },
    {
      title: '操作', render: (_, r) => (
        <Button size="small" onClick={() => void loadTraceDetail(r.trace_id || r.id)}>查看调用链</Button>
      ),
    },
  ];

  return (
    <div className="p-5">
      <div className="mb-4 flex justify-between items-center">
        <h3>Trace 可视化</h3>
        <Space>
          <Input.Search
            placeholder="输入 Trace ID 搜索"
            value={searchId}
            onChange={(e) => setSearchId(e.target.value)}
            onSearch={() => void searchTrace()}
            style={{ width: 300 }}
          />
          <Button onClick={() => void loadTraces()}>刷新</Button>
        </Space>
      </div>

      {selectedTrace && (
        <Card
          title={`Trace 瀑布图: ${selectedTraceId.slice(0, 16)}...`}
          className="mb-4"
          extra={<Button size="small" onClick={() => { setSelectedTrace(null); setSelectedTraceId(''); }}>关闭</Button>}
        >
          <div className="mb-2 text-sm text-gray-500">
            共 {selectedTrace.length} 个 Span
          </div>
          {renderWaterfall(selectedTrace)}

          <div className="mt-4">
            <h4 className="text-sm font-medium mb-2">Span 详情</h4>
            <Table
              rowKey="id"
              size="small"
              dataSource={selectedTrace}
              pagination={false}
              columns={[
                { title: '名称', dataIndex: 'name', width: 150 },
                { title: 'Span ID', render: (_, r) => <code className="text-[10px]">{(r.spec?.span_id || r.id)?.slice(0, 12)}</code>, width: 120 },
                { title: '父 Span', render: (_, r) => <code className="text-[10px]">{r.spec?.parent_span_id?.slice(0, 12) || '-'}</code>, width: 120 },
                { title: '耗时', render: (_, r) => `${r.spec?.duration_ms || r.latency_ms || 0}ms`, width: 80 },
                { title: '状态', render: (_, r) => <Tag color={getSpanColor(r.spec?.status || r.status || 'ok')} className="text-xs">{r.spec?.status || r.status || 'ok'}</Tag>, width: 80 },
                { title: '属性', render: (_, r) => <span className="text-xs">{JSON.stringify(r.spec?.attributes || {}).slice(0, 80)}</span> },
              ]}
            />
          </div>
        </Card>
      )}

      <Table rowKey="id" columns={traceColumns} dataSource={traces} loading={loading} pagination={{ pageSize: 20 }} />
    </div>
  );
}
