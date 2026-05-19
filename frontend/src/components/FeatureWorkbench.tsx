import { Alert, Button, Card, Form, Input, Select, Space, Typography, message } from 'antd';
import React, { useMemo, useState } from 'react';
import type { PageConfig, ResourceRecord } from '../models/types';
import { api } from '../services/http';

interface Props {
  page: PageConfig;
  rows: ResourceRecord[];
  onChanged: () => Promise<void>;
}

interface Operation {
  key: string;
  label: string;
  needsResource: boolean;
  buildPath: (resourceId: string) => string;
}

function operationsFor(page: PageConfig): Operation[] {
  if (page.api === '/knowledge-bases') {
    return [
      { key: 'upload', label: '上传文档', needsResource: true, buildPath: (id) => `/knowledge-bases/${id}/documents` },
      { key: 'retrieve', label: '检索测试', needsResource: true, buildPath: (id) => `/knowledge-bases/${id}/retrieve` },
      { key: 'rebuild', label: '重建索引', needsResource: true, buildPath: (id) => `/knowledge-bases/${id}/rebuild-index` }
    ];
  }
  if (page.api === '/agents') {
    return [
      { key: 'version', label: '创建版本', needsResource: true, buildPath: (id) => `/agents/${id}/versions` },
      { key: 'release', label: '发布 Agent', needsResource: true, buildPath: (id) => `/agents/${id}/release` },
      { key: 'gray-release', label: '灰度发布', needsResource: true, buildPath: (id) => `/agents/${id}/gray-release` },
      { key: 'rollback', label: '回滚', needsResource: true, buildPath: (id) => `/agents/${id}/rollback` },
      { key: 'debug-run', label: '调试运行', needsResource: true, buildPath: (id) => `/agents/${id}/debug-run` }
    ];
  }
  if (page.api === '/workflows') {
    return [
      { key: 'validate', label: 'DAG 校验', needsResource: true, buildPath: (id) => `/workflows/${id}/validate` },
      { key: 'run', label: '执行 Workflow', needsResource: true, buildPath: (id) => `/workflows/${id}/run` },
      { key: 'publish', label: '发布版本', needsResource: true, buildPath: (id) => `/workflows/${id}/publish` }
    ];
  }
  if (page.api === '/tools') {
    return [{ key: 'invoke', label: '调用工具', needsResource: true, buildPath: (id) => `/tools/${id}/invoke` }];
  }
  if (page.api === '/memories') {
    return [
      { key: 'extract', label: '抽取记忆', needsResource: false, buildPath: () => '/memories/extract' },
      { key: 'search', label: '检索记忆', needsResource: false, buildPath: () => '/memories/search' }
    ];
  }
  if (page.api === '/models') {
    return [
      { key: 'version', label: '创建模型版本', needsResource: true, buildPath: (id) => `/models/${id}/versions` },
      { key: 'health', label: '健康检查', needsResource: true, buildPath: (id) => `/models/${id}/health-check` },
      { key: 'reset-circuit', label: '熔断重置', needsResource: true, buildPath: (id) => `/model-circuit-breakers/${id}/reset` },
      { key: 'test-chat', label: '对话测试', needsResource: true, buildPath: (id) => `/models/${id}/test-chat` },
      { key: 'test-embedding', label: 'Embedding 测试', needsResource: true, buildPath: (id) => `/models/${id}/test-embedding` },
      { key: 'test-rerank', label: 'Rerank 测试', needsResource: true, buildPath: (id) => `/models/${id}/test-rerank` },
      { key: 'test-moderation', label: '审核测试', needsResource: true, buildPath: (id) => `/models/${id}/test-moderation` }
    ];
  }
  if (page.api === '/model-credentials') {
    return [{ key: 'credential-test', label: '凭证测试', needsResource: true, buildPath: (id) => `/model-credentials/${id}/test` }];
  }
  if (page.api === '/model-routing-policies') {
    return [
      { key: 'route', label: '调度模拟', needsResource: false, buildPath: () => '/models/route' },
      { key: 'invoke', label: '路由调用', needsResource: false, buildPath: () => '/models/invoke' }
    ];
  }
  if (page.api.startsWith('/security/')) {
    return [{ key: 'security-check', label: '安全检查', needsResource: false, buildPath: () => '/security/check' }];
  }
  if (page.api === '/audit-logs') {
    return [{ key: 'audit-export', label: '审计导出', needsResource: false, buildPath: () => '/audit-logs/export' }];
  }
  if (page.api === '/evaluation/datasets' || page.api === '/evaluation/runs') {
    return [{ key: 'evaluation-run', label: '运行评测', needsResource: false, buildPath: () => '/evaluation/runs' }];
  }
  return [];
}

function parsePayload(text: string): Record<string, unknown> {
  if (!text.trim()) return {};
  const parsed = JSON.parse(text) as unknown;
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('JSON 必须是对象');
  }
  return parsed as Record<string, unknown>;
}

export function FeatureWorkbench({ page, rows, onChanged }: Props): JSX.Element | null {
  const operations = useMemo(() => operationsFor(page), [page]);
  const [operationKey, setOperationKey] = useState(operations[0]?.key || '');
  const [resourceId, setResourceId] = useState('');
  const [payloadText, setPayloadText] = useState('{}');
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const operation = operations.find((item) => item.key === operationKey);

  if (!operation) return null;

  const execute = async (): Promise<void> => {
    const targetId = resourceId || rows[0]?.id || '';
    if (operation.needsResource && !targetId) {
      message.warning('请先选择一条资源记录');
      return;
    }
    setRunning(true);
    try {
      const payload = parsePayload(payloadText);
      const response = await api.post(operation.buildPath(targetId), payload);
      setResult(response);
      await onChanged();
    } catch (error) {
      message.error(error instanceof Error ? error.message : '执行失败');
    } finally {
      setRunning(false);
    }
  };

  return (
    <Card className="mb-5" title="领域操作台" size="small">
      <Space direction="vertical" className="w-full" size="middle">
        <Form layout="vertical">
          <Space align="start" wrap>
            <Form.Item label="操作">
              <Select
                value={operationKey}
                style={{ width: 180 }}
                onChange={(value) => {
                  setOperationKey(value);
                  setResult(null);
                }}
                options={operations.map((item) => ({ label: item.label, value: item.key }))}
              />
            </Form.Item>
            {operation.needsResource ? (
              <Form.Item label="资源">
                <Select
                  value={resourceId || rows[0]?.id}
                  style={{ width: 260 }}
                  onChange={setResourceId}
                  options={rows.map((row) => ({ label: row.name || row.code || row.id, value: row.id }))}
                />
              </Form.Item>
            ) : null}
          </Space>
          <Form.Item label="请求 JSON">
            <Input.TextArea rows={5} value={payloadText} onChange={(event) => setPayloadText(event.target.value)} />
          </Form.Item>
          <Button type="primary" loading={running} onClick={() => void execute()}>
            执行
          </Button>
        </Form>
        {result ? (
          <Alert
            type="success"
            message="执行结果"
            description={<Typography.Text className="whitespace-pre-wrap">{JSON.stringify(result, null, 2)}</Typography.Text>}
          />
        ) : null}
      </Space>
    </Card>
  );
}

export const featureWorkbenchInternals = { operationsFor, parsePayload };
