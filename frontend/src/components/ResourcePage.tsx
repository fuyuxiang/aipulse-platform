import { createForm } from '@formily/core';
import { FormProvider, Field } from '@formily/react';
import { FormItem, Input } from '@formily/antd-v5';
import { Button, Drawer, Form, Input as AntInput, Space, Table, Tag, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import React, { useEffect, useMemo, useState } from 'react';
import type { PageConfig, ResourceRecord } from '../models/types';
import { api } from '../services/http';
import { WorkflowDesigner } from '../workflow-designer/WorkflowDesigner';
import { AgentWizard } from '../agent-designer/AgentWizard';
import { FeatureWorkbench } from './FeatureWorkbench';

interface Props {
  page: PageConfig;
}

export function ResourcePage({ page }: Props): JSX.Element {
  const [rows, setRows] = useState<ResourceRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [keyword, setKeyword] = useState('');
  const form = useMemo(() => createForm(), []);

  const load = async (pageNo = 1): Promise<void> => {
    setLoading(true);
    try {
      const result = await api.list(page.api, pageNo, 20);
      const filtered = keyword ? result.items.filter((item) => `${item.name}${item.code}${item.status}`.includes(keyword)) : result.items;
      setRows(filtered);
      setTotal(result.total);
    } catch (error) {
      message.error(error instanceof Error ? error.message : '加载失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, [page.api]);

  const columns: ColumnsType<ResourceRecord> = [
    { title: '名称', dataIndex: 'name', sorter: (a, b) => a.name.localeCompare(b.name) },
    { title: '编码', dataIndex: 'code' },
    { title: '状态', dataIndex: 'status', render: (value: string) => <Tag color={value === 'active' || value === 'success' ? 'green' : 'blue'}>{value}</Tag> },
    { title: '启用', dataIndex: 'enabled', render: (value: boolean) => <Tag color={value ? 'green' : 'red'}>{value ? '启用' : '停用'}</Tag> },
    { title: '更新时间', dataIndex: 'updated_at' },
    {
      title: '操作',
      render: (_, row) => (
        <Space>
          <Button size="small" onClick={() => void api.action(`${page.api}/${row.id}/enable`, {}).then(() => load())}>启用</Button>
          <Button size="small" onClick={() => void api.action(`${page.api}/${row.id}/disable`, {}).then(() => load())}>停用</Button>
        </Space>
      )
    }
  ];

  const submit = async (): Promise<void> => {
    const values = await form.submit<Record<string, unknown>>();
    await api.create(page.api, {
      name: String(values.name || ''),
      code: String(values.code || ''),
      description: String(values.description || ''),
      model_type: String(values.model_type || ''),
      provider_type: String(values.provider_type || ''),
      config: values.configText ? JSON.parse(String(values.configText)) : {},
      spec: values.specText ? JSON.parse(String(values.specText)) : {}
    });
    setOpen(false);
    form.reset();
    await load();
  };

  return (
    <div className="p-5">
      <div className="mb-4 flex items-center justify-between">
        <Space>
          <AntInput.Search placeholder="筛选名称、编码、状态" value={keyword} onChange={(event) => setKeyword(event.target.value)} onSearch={() => void load()} allowClear />
          <Button onClick={() => void load()}>刷新</Button>
        </Space>
        <Button type="primary" onClick={() => setOpen(true)}>新建</Button>
      </div>
      {page.designer === 'workflow' ? <WorkflowDesigner /> : null}
      {page.designer === 'agent' ? <AgentWizard /> : null}
      <FeatureWorkbench page={page} rows={rows} onChanged={() => load()} />
      <Table<ResourceRecord>
        rowKey="id"
        columns={columns}
        dataSource={rows}
        loading={loading}
        pagination={{ total, pageSize: 20, onChange: (pageNo) => void load(pageNo) }}
      />
      <Drawer open={open} title={`新建${page.title}`} width={520} onClose={() => setOpen(false)} extra={<Button type="primary" onClick={() => void submit()}>保存</Button>}>
        <FormProvider form={form}>
          <Form layout="vertical">
            <Field name="name" title="名称" required decorator={[FormItem]} component={[Input]} />
            <Field name="code" title="编码" decorator={[FormItem]} component={[Input]} />
            <Field name="description" title="描述" decorator={[FormItem]} component={[Input.TextArea, { rows: 3 }]} />
            <Field name="model_type" title="模型类型" decorator={[FormItem]} component={[Input]} />
            <Field name="provider_type" title="Provider 类型" decorator={[FormItem]} component={[Input]} />
            <Field name="configText" title="配置 JSON" decorator={[FormItem]} component={[Input.TextArea, { rows: 5 }]} />
            <Field name="specText" title="规格 JSON" decorator={[FormItem]} component={[Input.TextArea, { rows: 5 }]} />
          </Form>
        </FormProvider>
      </Drawer>
    </div>
  );
}
