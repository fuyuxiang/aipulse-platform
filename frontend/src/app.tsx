import { Button, ConfigProvider, Result, Spin } from "antd";
import zhCN from "antd/locale/zh_CN";
import React, { lazy, Suspense, useEffect, useState } from "react";
import {
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router-dom";
import { AccessContext, canVisit, type CurrentUser } from "./access/context";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { PlatformShell } from "./components/PlatformShell";
import { DashboardPage } from "./pages/Home";
import { LoginPage } from "./pages/Login";
import { legacyRedirects, pageConfigs } from "./routes/pageConfig";
import { api, getToken } from "./services/http";

const ResourcePage = lazy(() =>
  import("./components/ResourcePage").then((module) => ({
    default: module.ResourcePage,
  })),
);
const ChatPage = lazy(() =>
  import("./pages/Playground").then((module) => ({ default: module.ChatPage })),
);
const CostAnalyticsPage = lazy(() =>
  import("./pages/Observe/Cost").then((module) => ({
    default: module.CostAnalyticsPage,
  })),
);
const GuardrailsPage = lazy(() =>
  import("./pages/Settings/Guardrails").then((module) => ({
    default: module.GuardrailsPage,
  })),
);
const PublicationsPage = lazy(() =>
  import("./pages/Deploy/Publications").then((module) => ({
    default: module.PublicationsPage,
  })),
);
const VersionsPage = lazy(() =>
  import("./pages/Deploy/Versions").then((module) => ({
    default: module.VersionsPage,
  })),
);
const MarketplacePage = lazy(() =>
  import("./pages/Deploy/Channels").then((module) => ({
    default: module.MarketplacePage,
  })),
);
const MultiAgentPage = lazy(() =>
  import("./pages/Playground/Multi").then((module) => ({
    default: module.MultiAgentPage,
  })),
);
const PromptStudioPage = lazy(() =>
  import("./pages/Build/Prompts").then((module) => ({
    default: module.PromptStudioPage,
  })),
);
const SchedulerPage = lazy(() =>
  import("./pages/Settings/Scheduler").then((module) => ({
    default: module.SchedulerPage,
  })),
);
const TraceVisualizationPage = lazy(() =>
  import("./pages/Observe/Trace").then((module) => ({
    default: module.TraceVisualizationPage,
  })),
);
const AuditPage = lazy(() =>
  import("./pages/Observe/Audit").then((module) => ({
    default: module.AuditPage,
  })),
);
const AlertsPage = lazy(() =>
  import("./pages/Observe/Alerts").then((module) => ({
    default: module.AlertsPage,
  })),
);
const UsersPage = lazy(() =>
  import("./pages/Settings/Users").then((module) => ({
    default: module.UsersPage,
  })),
);
const RolesPage = lazy(() =>
  import("./pages/Settings/Roles").then((module) => ({
    default: module.RolesPage,
  })),
);
const OrganizationsPage = lazy(() =>
  import("./pages/Settings/Organizations").then((module) => ({
    default: module.OrganizationsPage,
  })),
);
const SecurityPage = lazy(() =>
  import("./pages/Settings/Security").then((module) => ({
    default: module.SecurityPage,
  })),
);
const SystemPage = lazy(() =>
  import("./pages/Settings/System").then((module) => ({
    default: module.SystemPage,
  })),
);

export function App(): JSX.Element {
  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: "#2563eb",
          colorSuccess: "#169b70",
          colorWarning: "#bd790a",
          colorError: "#d14343",
          borderRadius: 8,
          fontFamily:
            'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
        },
        components: {
          Table: {
            headerBg: "#f8fafc",
            headerColor: "#475467",
            borderColor: "#edf0f3",
          },
          Card: { boxShadowTertiary: "none" },
          Button: { controlHeight: 34 },
        },
      }}
    >
      <ErrorBoundary>
        <AuthenticatedApp />
      </ErrorBoundary>
    </ConfigProvider>
  );
}

function AuthenticatedApp(): JSX.Element {
  const location = useLocation();
  const navigate = useNavigate();
  const token = getToken();
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const onUnauthorized = () => {
      setUser(null);
      navigate("/login", { replace: true });
    };
    window.addEventListener("aipulse:unauthorized", onUnauthorized);
    return () =>
      window.removeEventListener("aipulse:unauthorized", onUnauthorized);
  }, [navigate]);
  useEffect(() => {
    let active = true;
    setUser(null);
    setError("");
    if (token)
      void api
        .me()
        .then((value) => {
          if (active) setUser(value as unknown as CurrentUser);
        })
        .catch((reason) => {
          if (active) setError(reason.message);
        });
    return () => {
      active = false;
    };
  }, [token, attempt]);
  if (location.pathname === "/login") return <LoginPage />;
  if (!token) return <Navigate to="/login" replace />;
  if (error)
    return (
      <Result
        status="warning"
        title="账号信息加载失败"
        subTitle={error}
        extra={
          <Button onClick={() => setAttempt((value) => value + 1)}>
            重新加载
          </Button>
        }
      />
    );
  if (!user)
    return (
      <div className="app-loading">
        <Spin size="large" />
        <span>正在加载工作空间…</span>
      </div>
    );
  const page = pageConfigs.find((item) => item.path === location.pathname);
  return (
    <AccessContext.Provider value={user}>
      <PlatformShell user={user}>
        {page && !canVisit(user, page) ? (
          <Result
            status="403"
            title="暂无访问权限"
            subTitle="请联系管理员分配所需权限。"
            extra={
              <Button onClick={() => navigate("/home")}>返回工作台</Button>
            }
          />
        ) : (
          <Suspense
            fallback={
              <div className="page-loading">
                <Spin />
                <span>正在加载页面…</span>
              </div>
            }
          >
            <Routes>
              <Route path="/" element={<Navigate to="/home" replace />} />
              <Route path="/home" element={<DashboardPage />} />
              <Route path="/playground" element={<ChatPage />} />
              <Route path="/playground/multi" element={<MultiAgentPage />} />
              <Route path="/settings/guardrails" element={<GuardrailsPage />} />
              <Route path="/build/prompts" element={<PromptStudioPage />} />
              <Route path="/deploy/api-keys" element={<PublicationsPage />} />
              <Route path="/deploy/versions" element={<VersionsPage />} />
              <Route path="/deploy/channels" element={<MarketplacePage />} />
              <Route path="/settings/scheduler" element={<SchedulerPage />} />
              <Route path="/observe/cost" element={<CostAnalyticsPage />} />
              <Route
                path="/observe/trace"
                element={<TraceVisualizationPage />}
              />
              <Route path="/observe/audit" element={<AuditPage />} />
              <Route path="/observe/alerts" element={<AlertsPage />} />
              <Route path="/settings/users" element={<UsersPage />} />
              <Route path="/settings/roles" element={<RolesPage />} />
              <Route
                path="/settings/organizations"
                element={<OrganizationsPage />}
              />
              <Route path="/settings/security" element={<SecurityPage />} />
              <Route path="/settings/system" element={<SystemPage />} />
              {Object.entries(legacyRedirects).map(([from, to]) => (
                <Route
                  key={from}
                  path={from}
                  element={<Navigate to={to} replace />}
                />
              ))}
              {pageConfigs.map((page) => (
                <Route
                  key={page.path}
                  path={page.path}
                  element={<ResourcePage key={page.path} page={page} />}
                />
              ))}
              <Route
                path="*"
                element={
                  <Result
                    status="404"
                    title="页面不存在"
                    extra={
                      <Button onClick={() => navigate("/home")}>
                        返回工作台
                      </Button>
                    }
                  />
                }
              />
            </Routes>
          </Suspense>
        )}
      </PlatformShell>
    </AccessContext.Provider>
  );
}
