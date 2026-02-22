import { IconFontFill } from '@/components/icon-font';
import { RAGFlowAvatar } from '@/components/ragflow-avatar';
import { useTheme } from '@/components/theme-provider';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { LanguageList, LanguageMap, ThemeEnum } from '@/constants/common';
import { useChangeLanguage } from '@/hooks/logic-hooks';
import { useNavigatePage } from '@/hooks/logic-hooks/navigate-hooks';
import { useNavigateWithFromState } from '@/hooks/route-hook';
import { useFetchUserInfo } from '@/hooks/use-user-setting-request';
import { Routes } from '@/routes';
import { camelCase } from 'lodash';
import {
  ChevronDown,
  Cpu,
  File,
  House,
  Library,
  MessageSquareText,
  Moon,
  Search,
  Sun,
} from 'lucide-react';
import React, { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useLocation } from 'umi';
import { NavLink } from 'umi';
import { BellButton } from './bell-button';
import './sider-nav.css';

const PathMap = {
  [Routes.Datasets]: [Routes.Datasets],
  [Routes.Chats]: [Routes.Chats],
  [Routes.Searches]: [Routes.Searches],
  [Routes.Agents]: [Routes.Agents],
  [Routes.Memories]: [Routes.Memories, Routes.Memory, Routes.MemoryMessage],
  [Routes.Files]: [Routes.Files],
} as const;

interface NavItem {
  path: string;
  name: string;
  icon: React.ComponentType<{ className?: string }>;
}

export function SiderNav() {
  const { t } = useTranslation();
  const { pathname } = useLocation();
  const navigate = useNavigateWithFromState();
  const { navigateToOldProfile } = useNavigatePage();

  const changeLanguage = useChangeLanguage();
  const { setTheme, theme } = useTheme();

  const {
    data: { language = 'English', avatar, nickname },
  } = useFetchUserInfo();

  const handleItemClick = (key: string) => () => {
    changeLanguage(key);
  };

  const items = LanguageList.map((x) => ({
    key: x,
    label: <span>{LanguageMap[x as keyof typeof LanguageMap]}</span>,
  }));

  const onThemeClick = React.useCallback(() => {
    setTheme(theme === ThemeEnum.Dark ? ThemeEnum.Light : ThemeEnum.Dark);
  }, [setTheme, theme]);

  const tagsData: NavItem[] = useMemo(
    () => [
      { path: Routes.Root, name: t('header.Root'), icon: House },
      { path: Routes.Datasets, name: t('header.dataset'), icon: Library },
      { path: Routes.Chats, name: t('header.chat'), icon: MessageSquareText },
      { path: Routes.Searches, name: t('header.search'), icon: Search },
      { path: Routes.Agents, name: t('header.flow'), icon: Cpu },
      { path: Routes.Memories, name: t('header.memories'), icon: Cpu },
      { path: Routes.Files, name: t('header.fileManager'), icon: File },
    ],
    [t],
  );

  const handleLogoClick = useCallback(() => {
    navigate(Routes.Root);
  }, [navigate]);

  const activePathName = useMemo(() => {
    const name = Object.keys(PathMap).find((x: string) => {
      const pathList = PathMap[x as keyof typeof PathMap];
      return pathList.some((y: string) => pathname.indexOf(y) > -1);
    });
    if (name) {
      return name;
    } else {
      return pathname;
    }
  }, [pathname]);

  const isActive = (itemPath: string) => {
    if (itemPath === Routes.Root) {
      return pathname === Routes.Root;
    }
    return pathname.startsWith(itemPath);
  };

  return (
    <aside className="sider-nav">
      {/* Logo 区域 */}
      <div className="sider-logo" onClick={handleLogoClick}>
        <img src="/logoPic.png" alt="logo" className="sider-logo-img" />
        <span className="sider-logo-text">RAGFlow</span>
      </div>

      {/* 导航菜单 */}
      <nav className="sider-nav-menu">
        {tagsData.map((item) => {
          const Icon = item.icon;
          const active = isActive(item.path);
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={`sider-nav-item ${active ? 'sider-nav-item-active' : ''}`}
            >
              <Icon className="sider-nav-icon" />
              <span className="sider-nav-text">{item.name}</span>
            </NavLink>
          );
        })}
      </nav>

      {/* 底部用户区域 */}
      <div className="sider-footer">
        <div className="sider-footer-actions">
          <DropdownMenu>
            <DropdownMenuTrigger>
              <div className="sider-action-btn">
                <ChevronDown className="sider-action-icon" />
              </div>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              {items.map((x) => (
                <DropdownMenuItem key={x.key} onClick={handleItemClick(x.key)}>
                  {x.label}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <BellButton />
        </div>

        <div className="sider-user" onClick={navigateToOldProfile}>
          <RAGFlowAvatar
            name={nickname}
            avatar={avatar}
            isPerson
            className="sider-user-avatar"
          />
          <div className="sider-user-info">
            <span className="sider-user-name">{nickname}</span>
            <span className="sider-user-email">{language}</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
