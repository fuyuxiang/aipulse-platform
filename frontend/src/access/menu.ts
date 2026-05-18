import type { MenuDataItem } from '@ant-design/pro-components';
import { pageConfigs } from '../routes/pageConfig';

export function buildMenus(): MenuDataItem[] {
  const grouped = new Map<string, MenuDataItem>();
  for (const page of pageConfigs) {
    const group = grouped.get(page.group) || { path: `/${encodeURIComponent(page.group)}`, name: page.group, children: [] };
    group.children = [...(group.children || []), { path: page.path, name: page.title }];
    grouped.set(page.group, group);
  }
  return Array.from(grouped.values());
}

