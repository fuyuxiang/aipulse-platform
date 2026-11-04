import {
  BellOutlined,
  CloudServerOutlined,
  LogoutOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  PlusOutlined,
  SearchOutlined,
  UserOutlined,
} from "@ant-design/icons";
import {
  Avatar,
  Breadcrumb,
  Button,
  Drawer,
  Dropdown,
  Grid,
  Menu,
  Select,
  Space,
  Tooltip,
} from "antd";
import React, {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { buildMenus } from "../access/menu";
import { canVisit, hasPermission, type CurrentUser } from "../access/context";
import { pageConfigs } from "../routes/pageConfig";

export function PlatformShell({
  user,
  children,
}: {
  user: CurrentUser;
  children: React.ReactNode;
}): JSX.Element {
  const location = useLocation();
  const navigate = useNavigate();
  const screens = Grid.useBreakpoint();
  const mobile = screens.lg === false;
  const [collapsed, setCollapsed] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const [openKeys, setOpenKeys] = useState<string[]>([]);
  const searchRef = useRef<any>(null);
  const pages = pageConfigs.filter((page) => canVisit(user, page));
  const page = pageConfigs.find((item) => item.path === location.pathname);
  const menus = useMemo(() => buildMenus(user), [user]);
  useLayoutEffect(() => {
    const parent = menus.find((item) =>
      item.children?.some(
        (child) =>
          child.path === location.pathname ||
          (page?.designer &&
            child.path ===
              (page.designer === "agent"
                ? "/build/agents"
                : "/build/workflows")),
      ),
    );
    if (parent?.path)
      setOpenKeys((keys) => [...new Set([...keys, parent.path!])]);
    setDrawer(false);
    document.title = `${page?.title || "页面"} · AIPulse`;
  }, [location.pathname, menus, page]);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "k") {
        event.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
  const items = menus.map((item) => ({
    key: item.path!,
    icon: item.icon,
    label: item.children?.length ? (
      item.name
    ) : (
      <Link to={item.path!}>{item.name}</Link>
    ),
    children: item.children?.map((child) => ({
      key: child.path!,
      label: <Link to={child.path!}>{child.name}</Link>,
    })),
  }));
  const navigation = (
    <Menu
      mode="inline"
      selectedKeys={[
        page?.designer
          ? page.designer === "agent"
            ? "/build/agents"
            : "/build/workflows"
          : location.pathname,
      ]}
      openKeys={collapsed && !mobile ? undefined : openKeys}
      onOpenChange={setOpenKeys}
      inlineCollapsed={collapsed && !mobile}
      items={items}
    />
  );
  const brand = (
    <Link to="/home" className="brand-entry" aria-label="AIPulse 工作台">
      <span className="app-logo-mark">AP</span>
      {(!collapsed || mobile) && (
        <span>
          <strong>AIPulse</strong>
          <small>智能体协作平台</small>
        </span>
      )}
    </Link>
  );
  const quickActions = [
    {
      key: "/build/agents/create",
      label: "创建智能体",
      permission: "agents:write",
    },
    {
      key: "/build/workflows/designer",
      label: "创建工作流",
      permission: "workflows:write",
    },
    { key: "/playground", label: "开始对话", permission: "chat:write" },
  ].filter((item) => hasPermission(user, item.permission));
  const logout = () => {
    [
      "aipulse_access_token",
      "aipulse_refresh_token",
      "aipulse_tenant",
      "aipulse_username",
    ].forEach((key) => localStorage.removeItem(key));
    navigate("/login", { replace: true });
  };
  return (
    <div
      className={`platform-shell ${collapsed ? "sidebar-collapsed" : ""} ${mobile ? "platform-mobile" : ""}`}
    >
      {!mobile && (
        <aside className="platform-sidebar">
          {brand}
          <nav aria-label="主导航">{navigation}</nav>
          <div className="sidebar-footer">
            {!collapsed && (
              <>
                <span className="environment-dot" />
                本地开发版<span className="version-label">v0.1.0</span>
              </>
            )}
          </div>
        </aside>
      )}
      <Drawer
        placement="left"
        width={240}
        open={drawer && mobile}
        onClose={() => setDrawer(false)}
        title={brand}
        styles={{ body: { padding: 0 } }}
      >
        {navigation}
      </Drawer>
      <div className="platform-body">
        <header className="platform-header">
          <Button
            type="text"
            aria-label={
              mobile ? "打开导航" : collapsed ? "展开导航" : "收起导航"
            }
            icon={
              collapsed || mobile ? (
                <MenuUnfoldOutlined />
              ) : (
                <MenuFoldOutlined />
              )
            }
            onClick={() =>
              mobile ? setDrawer(true) : setCollapsed((value) => !value)
            }
          />
          <Breadcrumb
            className="header-breadcrumb"
            items={[
              { title: <Link to="/home">AIPulse</Link> },
              ...(page && !mobile && page.path !== "/home"
                ? [{ title: page.group }]
                : []),
              { title: page?.title || "页面" },
            ]}
          />
          <div className="header-spacer" />
          <Select
            ref={searchRef}
            aria-label="搜索页面"
            className="global-page-search"
            showSearch
            value={null}
            suffixIcon={<SearchOutlined />}
            placeholder="搜索页面  ⌘ K"
            optionFilterProp="label"
            options={pages.map((item) => ({
              value: item.path,
              label: `${item.title} · ${item.group}`,
            }))}
            onChange={navigate}
            notFoundContent="没有匹配的页面"
          />
          {!!quickActions.length && (
            <Dropdown
              menu={{
                items: quickActions,
                onClick: ({ key }) => navigate(key),
              }}
              trigger={["click"]}
            >
              <Button
                icon={<PlusOutlined />}
                aria-label="快捷新建"
                className="quick-create"
              >
                新建
              </Button>
            </Dropdown>
          )}
          <Tooltip title="当前租户">
            <span className="tenant-indicator">
              <CloudServerOutlined />
              {localStorage.getItem("aipulse_tenant") || "当前租户"}
            </span>
          </Tooltip>
          {hasPermission(user, "alerts:read") && (
            <Tooltip title="告警中心">
              <Button
                type="text"
                aria-label="告警中心"
                icon={<BellOutlined />}
                onClick={() => navigate("/observe/alerts")}
              />
            </Tooltip>
          )}
          <Dropdown
            trigger={["click"]}
            menu={{
              items: [
                {
                  key: "identity",
                  label: user.display_name || user.username,
                  disabled: true,
                },
                { key: "logout", label: "退出登录", icon: <LogoutOutlined /> },
              ],
              onClick: ({ key }) => key === "logout" && logout(),
            }}
          >
            <Button type="text" className="user-entry" aria-label="账号菜单">
              <Space>
                <Avatar size={26} icon={<UserOutlined />} />
                <span className="username-label">
                  {user.display_name || user.username}
                </span>
              </Space>
            </Button>
          </Dropdown>
        </header>
        <main className="app-content-surface" id="main-content">
          {children}
        </main>
      </div>
    </div>
  );
}
