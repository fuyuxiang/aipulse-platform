import { AppstoreOutlined, DownloadOutlined, StarOutlined } from '@ant-design/icons';
import { Button, Card, Col, Drawer, Form, Input, Rate, Row, Select, Space, Table, Tabs, Tag, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import React, { useEffect, useState } from 'react';
import { getToken } from '../../services/http';

export function MarketplacePage(): JSX.Element {
  const [listings, setListings] = useState<any[]>([]);
  const [installs, setInstalls] = useState<any[]>([]);
  const [categories, setCategories] = useState<any[]>([]);
  const [stats, setStats] = useState<any>({});
  const [loading, setLoading] = useState(false);
  const [publishOpen, setPublishOpen] = useState(false);
  const [detailOpen, setDetailOpen] = useState(false);
  const [selectedListing, setSelectedListing] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [form] = Form.useForm();

  const headers = { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' };

  const load = async (): Promise<void> => {
    setLoading(true);
    try {
      const [lRes, iRes, cRes, sRes] = await Promise.all([
        fetch('/api/v1/marketplace/listings?page=1&page_size=50', { headers }),
        fetch('/api/v1/marketplace/installs?page=1&page_size=50', { headers }),
        fetch('/api/v1/marketplace/categories?page=1&page_size=50', { headers }),
        fetch('/api/v1/marketplace/stats', { headers }),
      ]);
      if (lRes.ok) setListings((await lRes.json()).items || []);
      if (iRes.ok) setInstalls((await iRes.json()).items || []);
      if (cRes.ok) setCategories((await cRes.json()).items || []);
      if (sRes.ok) setStats(await sRes.json());
    } catch { /* ignore */ }
    setLoading(false);
  };

  useEffect(() => { void load(); }, []);

  const search = async (): Promise<void> => {
    try {
      const res = await fetch(`/api/v1/marketplace/search?q=${encodeURIComponent(searchQuery)}&page=1&page_size=50`, { headers });
      if (res.ok) setListings((await res.json()).items || []);
    } catch { /* ignore */ }
  };

  const publishListing = async (): Promise<void> => {
    try {
      const values = await form.validateFields();
      await fetch('/api/v1/marketplace/listings', {
        method: 'POST', headers,
        body: JSON.stringify({
          name: values.name,
          type: values.type,
          resource_id: values.resource_id,
          description: values.description,
          category_id: values.category_id,
          tags: values.tags ? values.tags.split(',').map((t: string) => t.trim()) : [],
          version: values.version || '1.0.0',
        }),
      });
      message.success('发布成功，等待审核');
      setPublishOpen(false);
      form.resetFields();
      await load();
    } catch { message.error('发布失败'); }
  };

  const installListing = async (listingId: string): Promise<void> => {
    try {
      await fetch(`/api/v1/marketplace/listings/${listingId}/install`, { method: 'POST', headers });
      message.success('安装成功');
      await load();
    } catch { message.error('安装失败'); }
  };

  const viewDetail = async (listing: any): Promise<void> => {
    try {
      const res = await fetch(`/api/v1/marketplace/listings/${listing.id}`, { headers });
      if (res.ok) {
        setSelectedListing(await res.json());
        setDetailOpen(true);
      }
    } catch { /* ignore */ }
  };

  const listingColumns: ColumnsType<any> = [
    { title: '名称', dataIndex: 'name', render: (v, r) => <a onClick={() => void viewDetail(r)}>{v}</a> },
    { title: '类型', render: (_, r) => <Tag color="blue">{r.spec?.type || 'agent'}</Tag> },
    { title: '版本', render: (_, r) => r.spec?.version || '1.0.0' },
    { title: '评分', render: (_, r) => <span><StarOutlined className="text-yellow-400" /> {r.spec?.rating_avg?.toFixed(1) || '-'}</span> },
    { title: '安装数', render: (_, r) => <span><DownloadOutlined /> {r.spec?.install_count || 0}</span> },
    { title: '状态', dataIndex: 'status', render: (v: string) => <Tag color={v === 'published' ? 'green' : 'orange'}>{v}</Tag> },
    {
      title: '操作', render: (_, r) => (
        <Space>
          <Button size="small" type="primary" onClick={() => void installListing(r.id)}>安装</Button>
          <Button size="small" onClick={() => void viewDetail(r)}>详情</Button>
        </Space>
      ),
    },
  ];

  const installColumns: ColumnsType<any> = [
    { title: '名称', dataIndex: 'name' },
    { title: '类型', render: (_, r) => <Tag>{r.spec?.listing_type || ''}</Tag> },
    { title: '版本', render: (_, r) => r.spec?.version || '' },
    { title: '安装时间', render: (_, r) => r.spec?.installed_at?.slice(0, 10) || '' },
    { title: '状态', dataIndex: 'status', render: (v: string) => <Tag color="green">{v}</Tag> },
  ];

  return (
    <div className="p-5">
      <Row gutter={16} className="mb-4">
        <Col span={8}><Card><div className="text-center"><AppstoreOutlined className="text-2xl text-blue-500" /><div className="mt-1">上架数 {stats.total_listings || 0}</div></div></Card></Col>
        <Col span={8}><Card><div className="text-center"><DownloadOutlined className="text-2xl text-green-500" /><div className="mt-1">安装数 {stats.total_installs || 0}</div></div></Card></Col>
        <Col span={8}><Card><div className="text-center"><StarOutlined className="text-2xl text-yellow-500" /><div className="mt-1">评价数 {stats.total_reviews || 0}</div></div></Card></Col>
      </Row>

      <Tabs items={[
        {
          key: 'browse', label: '浏览市场',
          children: (
            <>
              <div className="mb-4 flex justify-between">
                <Space>
                  <Input.Search placeholder="搜索 Agent、工具、Workflow..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} onSearch={() => void search()} style={{ width: 300 }} />
                </Space>
                <Button type="primary" onClick={() => setPublishOpen(true)}>发布到市场</Button>
              </div>
              <Table rowKey="id" columns={listingColumns} dataSource={listings} loading={loading} pagination={{ pageSize: 20 }} />
            </>
          ),
        },
        {
          key: 'installed', label: '已安装',
          children: <Table rowKey="id" columns={installColumns} dataSource={installs} loading={loading} pagination={{ pageSize: 20 }} />,
        },
      ]} />

      <Drawer open={publishOpen} title="发布到市场" width={520} onClose={() => setPublishOpen(false)} extra={<Button type="primary" onClick={() => void publishListing()}>发布</Button>}>
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="type" label="类型" initialValue="agent">
            <Select options={[{ value: 'agent', label: 'Agent' }, { value: 'tool', label: '工具' }, { value: 'workflow', label: 'Workflow' }, { value: 'prompt', label: 'Prompt 模板' }]} />
          </Form.Item>
          <Form.Item name="resource_id" label="资源 ID" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="description" label="描述"><Input.TextArea rows={3} /></Form.Item>
          <Form.Item name="category_id" label="分类">
            <Select allowClear options={categories.map((c) => ({ value: c.id, label: c.name }))} />
          </Form.Item>
          <Form.Item name="tags" label="标签 (逗号分隔)"><Input placeholder="AI, 客服, 翻译" /></Form.Item>
          <Form.Item name="version" label="版本" initialValue="1.0.0"><Input /></Form.Item>
        </Form>
      </Drawer>

      <Drawer open={detailOpen} title={selectedListing?.name || '详情'} width={600} onClose={() => setDetailOpen(false)}>
        {selectedListing && (
          <Space direction="vertical" className="w-full" size="middle">
            <Card size="small">
              <p><strong>类型:</strong> <Tag>{selectedListing.spec?.type}</Tag></p>
              <p><strong>版本:</strong> {selectedListing.spec?.version}</p>
              <p><strong>描述:</strong> {selectedListing.spec?.description}</p>
              <p><strong>作者:</strong> {selectedListing.spec?.author || '匿名'}</p>
              <p><strong>评分:</strong> <Rate disabled value={selectedListing.spec?.rating_avg || 0} /></p>
              <p><strong>安装数:</strong> {selectedListing.spec?.install_count || 0}</p>
            </Card>
            {selectedListing.reviews?.length > 0 && (
              <Card size="small" title="评价">
                {selectedListing.reviews.map((r: any, i: number) => (
                  <div key={i} className="mb-2 border-b pb-2">
                    <Rate disabled value={r.spec?.rating || 5} className="text-sm" />
                    <p className="text-sm mt-1">{r.spec?.comment || ''}</p>
                  </div>
                ))}
              </Card>
            )}
            <Button type="primary" block onClick={() => void installListing(selectedListing.id)}>安装到我的空间</Button>
          </Space>
        )}
      </Drawer>
    </div>
  );
}
