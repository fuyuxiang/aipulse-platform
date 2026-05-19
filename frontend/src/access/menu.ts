import type { MenuDataItem } from '@ant-design/pro-components';
import { pageConfigs } from '../routes/pageConfig';

const GROUP_ORDER = ['首页', '开发', '运行', '治理', '设置'];

const GROUP_ICONS: Record<string, string> = {
  首页: 'DashboardOutlined',
  开发: 'CodeOutlined',
  运行: 'RocketOutlined',
  治理: 'SafetyCertificateOutlined',
  设置: 'SettingOutlined',
};

export function buildMenus(): MenuDataItem[] {
  const grouped = new Map<string, MenuDataItem[]>();

  for (const page of pageConfigs) {
    const children = grouped.get(page.group) || [];
    children.push({ path: page.path, name: page.title });
    grouped.set(page.group, children);
  }

  return GROUP_ORDER.map((group) => {
    const children = grouped.get(group) || [];
    if (group === '首页') {
      return { path: '/dashboard', name: '首页', icon: GROUP_ICONS[group] };
    }
    return {
      path: `/${encodeURIComponent(group)}`,
      name: group,
      icon: GROUP_ICONS[group],
      children,
    };
  });
}
