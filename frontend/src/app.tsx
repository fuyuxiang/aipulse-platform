import { ProLayout } from '@ant-design/pro-components';
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

export function App(): JSX.Element {
  const location = useLocation();
  const navigate = useNavigate();
  if (location.pathname === '/login') return <LoginPage />;
  return (
    <ProLayout
      title="AIPulse"
      layout="mix"
      route={{ path: '/', routes: buildMenus() }}
      location={{ pathname: location.pathname }}
      menuItemRender={(item, dom) => <Link to={item.path || '/dashboard'}>{dom}</Link>}
      onMenuHeaderClick={() => navigate('/dashboard')}
    >
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
    </ProLayout>
  );
}
