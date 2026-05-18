import { ProLayout } from '@ant-design/pro-components';
import React from 'react';
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { buildMenus } from './access/menu';
import { ResourcePage } from './components/ResourcePage';
import { DashboardPage } from './pages/Dashboard';
import { LoginPage } from './pages/Login';
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
        {pageConfigs.map((page) => (
          <Route key={page.path} path={page.path} element={<ResourcePage page={page} />} />
        ))}
      </Routes>
    </ProLayout>
  );
}

