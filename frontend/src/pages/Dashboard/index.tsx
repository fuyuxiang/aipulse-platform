import { Card, Col, Row, Statistic } from 'antd';
import React, { useEffect, useState } from 'react';
import { api } from '../../services/http';

export function DashboardPage(): JSX.Element {
  const [summary, setSummary] = useState<Record<string, number>>({});
  useEffect(() => {
    void api.list('/observability/metrics').catch(() => ({ items: [], total: 0, page: 1, page_size: 20 }));
    fetch('/api/v1/observability/dashboard', { headers: { Authorization: `Bearer ${localStorage.getItem('aipulse_access_token') || ''}` } })
      .then((res) => res.json())
      .then((data) => setSummary((data.summary || {}) as Record<string, number>))
      .catch(() => setSummary({}));
  }, []);
  return (
    <div className="p-5">
      <Row gutter={[16, 16]}>
        {Object.entries(summary).map(([key, value]) => (
          <Col span={6} key={key}>
            <Card><Statistic title={key} value={value} /></Card>
          </Col>
        ))}
      </Row>
    </div>
  );
}

