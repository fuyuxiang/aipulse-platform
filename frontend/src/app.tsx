import { ProLayout } from '@ant-design/pro-components';
import { BellOutlined, CloudServerOutlined, LogoutOutlined, SafetyCertificateOutlined, UserOutlined } from '@ant-design/icons';
import { Avatar, Badge, Button, ConfigProvider, Dropdown, Space, Tag, Tooltip, Typography } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import React from 'react';
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { buildMenus } from './access/menu';
import { ResourcePage } from './components/ResourcePage';
import { ChatPage } from './pages/Chat';
import { CostAnalyticsPage } from './pages/CostAnalytics';
import { DashboardPage } from './pages/Dashboard';
import { GuardrailsPage } from './pages/Guardrails';
import { LoginPage } from './pages/Login';
import { MarketplacePage } from './pages/Marketplace';
import { MultiAgentPage } from './pages/MultiAgent';
import { PromptStudioPage } from './pages/PromptStudio';
import { SchedulerPage } from './pages/Scheduler';
import { TraceVisualizationPage } from './pages/TraceVisualization';
import { pageConfigs } from './routes/pageConfig';
import { getToken } from './services/http';

export function App(): JSX.Element {
  const location = useLocation();
  const navigate = useNavigate();
  const token = getToken();
  const tenantName = localStorage.getItem('aipulse_tenant') || '当前租户';
  const username = localStorage.getItem('aipulse_username') || '用户';
  if (location.pathname === '/login') return <LoginPage />;
  if (!token) return <Navigate to="/login" replace />;
  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: '#1d5fd7',
          colorSuccess: '#16a36f',
          colorWarning: '#c87900',
          colorError: '#cf3030',
          colorInfo: '#1d5fd7',
          borderRadius: 8,
          fontFamily:
            'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        },
        components: {
          Card: { borderRadiusLG: 8 },
          Button: { borderRadius: 6 },
          Table: { borderColor: '#e5eaf2', headerBg: '#f6f8fb', headerColor: '#344054' },
          Layout: { bodyBg: '#f5f7fb', headerBg: '#ffffff', siderBg: '#ffffff' },
        },
      }}
    >
      <ProLayout
        title="AIPulse AgentOS"
        logo={<div className="app-logo-mark">AP</div>}
        layout="mix"
        navTheme="light"
        fixedHeader
        fixSiderbar
        className="enterprise-shell"
        route={{ path: '/', routes: buildMenus() }}
        location={{ pathname: location.pathname }}
        menuItemRender={(item, dom) => <Link to={item.path || '/dashboard'}>{dom}</Link>}
        menuFooterRender={() => (
          <div className="menu-runtime-card">
            <div className="menu-runtime-title">Agent Runtime</div>
            <div className="menu-runtime-meta">
              <span>生产运行控制面</span>
              <span>v0.1</span>
            </div>
          </div>
        )}
        rightContentRender={() => (
          <div className="app-header-actions">
            <Tag icon={<CloudServerOutlined />} color="processing">{tenantName}</Tag>
            <Tag icon={<SafetyCertificateOutlined />} color="success">RBAC</Tag>
            <Tooltip title="告警中心">
              <Badge dot={false}>
                <Button type="text" shape="circle" icon={<BellOutlined />} />
              </Badge>
            </Tooltip>
            <Dropdown
              menu={{
                items: [
                  { key: 'tenant', label: `${tenantName} / ${username}`, disabled: true },
                  { key: 'logout', label: '退出登录', icon: <LogoutOutlined /> },
                ],
                onClick: ({ key }) => {
                  if (key === 'logout') {
                    localStorage.removeItem('aipulse_access_token');
                    localStorage.removeItem('aipulse_refresh_token');
                    localStorage.removeItem('aipulse_tenant');
                    localStorage.removeItem('aipulse_username');
                    navigate('/login');
                  }
                },
              }}
            >
              <Space className="user-entry">
                <Avatar size={28} icon={<UserOutlined />} />
                <Typography.Text strong>{username}</Typography.Text>
              </Space>
            </Dropdown>
          </div>
        )}
        onMenuHeaderClick={() => navigate('/dashboard')}
      >
        <div className="app-content-surface">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/multi-agent" element={<MultiAgentPage />} />
            <Route path="/guardrails" element={<GuardrailsPage />} />
            <Route path="/prompt-studio" element={<PromptStudioPage />} />
            <Route path="/marketplace" element={<MarketplacePage />} />
            <Route path="/scheduler" element={<SchedulerPage />} />
            <Route path="/cost-analytics" element={<CostAnalyticsPage />} />
            <Route path="/trace" element={<TraceVisualizationPage />} />
            {pageConfigs.map((page) => (
              <Route key={page.path} path={page.path} element={<ResourcePage page={page} />} />
            ))}
          </Routes>
        </div>
      </ProLayout>
    </ConfigProvider>
  );
}
