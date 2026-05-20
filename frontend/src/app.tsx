import { ProLayout } from '@ant-design/pro-components';
import { BellOutlined, CloudServerOutlined, LogoutOutlined, SafetyCertificateOutlined, UserOutlined } from '@ant-design/icons';
import { Avatar, Badge, Button, ConfigProvider, Dropdown, Space, Tag, Tooltip, Typography } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import React from 'react';
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { buildMenus } from './access/menu';
import { ResourcePage } from './components/ResourcePage';
import { ChatPage } from './pages/Playground';
import { CostAnalyticsPage } from './pages/Observe/Cost';
import { DashboardPage } from './pages/Home';
import { GuardrailsPage } from './pages/Settings/Guardrails';
import { LoginPage } from './pages/Login';
import { MarketplacePage } from './pages/Deploy/Channels';
import { MultiAgentPage } from './pages/Playground/Multi';
import { PromptStudioPage } from './pages/Build/Prompts';
import { SchedulerPage } from './pages/Settings/Scheduler';
import { TraceVisualizationPage } from './pages/Observe/Trace';
import { AuditPage } from './pages/Observe/Audit';
import { AlertsPage } from './pages/Observe/Alerts';
import { UsersPage } from './pages/Settings/Users';
import { RolesPage } from './pages/Settings/Roles';
import { OrganizationsPage } from './pages/Settings/Organizations';
import { SecurityPage } from './pages/Settings/Security';
import { SystemPage } from './pages/Settings/System';
import { legacyRedirects, pageConfigs } from './routes/pageConfig';
import { getToken } from './services/http';

const HOME_PATH = '/home';

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
        menuItemRender={(item, dom) => <Link to={item.path || HOME_PATH}>{dom}</Link>}
        menuFooterRender={() => (
          <div className="menu-runtime-card">
            <div className="menu-runtime-title">AIPulse AgentOS</div>
            <div className="menu-runtime-meta">
              <span>智能体基础设施平台</span>
              <span>v1.0</span>
            </div>
          </div>
        )}
        rightContentRender={() => (
          <div className="app-header-actions">
            <Tag icon={<CloudServerOutlined />} color="processing">{tenantName}</Tag>
            <Tag icon={<SafetyCertificateOutlined />} color="success">RBAC</Tag>
            <Tooltip title="告警中心">
              <Badge dot={false}>
                <Button type="text" shape="circle" icon={<BellOutlined />} onClick={() => navigate('/observe/alerts')} />
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
        onMenuHeaderClick={() => navigate(HOME_PATH)}
      >
        <div className="app-content-surface">
          <Routes>
            <Route path="/" element={<Navigate to={HOME_PATH} replace />} />
            <Route path="/home" element={<DashboardPage />} />
            <Route path="/playground" element={<ChatPage />} />
            <Route path="/playground/multi" element={<MultiAgentPage />} />
            <Route path="/settings/guardrails" element={<GuardrailsPage />} />
            <Route path="/build/prompts" element={<PromptStudioPage />} />
            <Route path="/deploy/channels" element={<MarketplacePage />} />
            <Route path="/settings/scheduler" element={<SchedulerPage />} />
            <Route path="/observe/cost" element={<CostAnalyticsPage />} />
            <Route path="/observe/trace" element={<TraceVisualizationPage />} />
            <Route path="/observe/audit" element={<AuditPage />} />
            <Route path="/observe/alerts" element={<AlertsPage />} />
            <Route path="/settings/users" element={<UsersPage />} />
            <Route path="/settings/roles" element={<RolesPage />} />
            <Route path="/settings/organizations" element={<OrganizationsPage />} />
            <Route path="/settings/security" element={<SecurityPage />} />
            <Route path="/settings/system" element={<SystemPage />} />
            {Object.entries(legacyRedirects).map(([from, to]) => (
              <Route key={from} path={from} element={<Navigate to={to} replace />} />
            ))}
            {pageConfigs.map((page) => (
              <Route key={page.path} path={page.path} element={<ResourcePage page={page} />} />
            ))}
          </Routes>
        </div>
      </ProLayout>
    </ConfigProvider>
  );
}
