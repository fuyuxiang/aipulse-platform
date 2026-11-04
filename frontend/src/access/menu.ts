import {
  AppstoreOutlined,
  BuildOutlined,
  CloudUploadOutlined,
  ExperimentOutlined,
  EyeOutlined,
  HomeOutlined,
  PlayCircleOutlined,
  SettingOutlined,
} from "@ant-design/icons";
interface MenuDataItem {
  path?: string;
  name?: string;
  icon?: React.ReactNode;
  children?: MenuDataItem[];
}
import React from "react";
import { pageConfigs } from "../routes/pageConfig";
import { canVisit, type CurrentUser } from "./context";

const GROUP_ORDER = ["工作台", "使用", "构建", "评测", "观测", "发布", "设置"];

const GROUP_ICONS: Record<string, React.ComponentType> = {
  工作台: HomeOutlined,
  构建: BuildOutlined,
  使用: PlayCircleOutlined,
  评测: ExperimentOutlined,
  观测: EyeOutlined,
  发布: CloudUploadOutlined,
  设置: SettingOutlined,
};

const FLAT_GROUPS = new Set(["工作台", "评测"]);

const HIDDEN_PATHS = new Set<string>([
  "/build/agents/create",
  "/build/workflows/designer",
]);

export function buildMenus(user?: CurrentUser): MenuDataItem[] {
  const grouped = new Map<string, MenuDataItem[]>();

  for (const page of pageConfigs) {
    if (HIDDEN_PATHS.has(page.path) || (user && !canVisit(user, page)))
      continue;
    const children = grouped.get(page.group) || [];
    children.push({ path: page.path, name: page.title });
    grouped.set(page.group, children);
  }

  return GROUP_ORDER.filter((group) => grouped.has(group)).map((group) => {
    const children = grouped.get(group) || [];
    const icon = React.createElement(GROUP_ICONS[group] || AppstoreOutlined);
    if (FLAT_GROUPS.has(group) && children.length === 1) {
      const only = children[0];
      return { path: only.path, name: group, icon };
    }
    return {
      path: `/__group_${encodeURIComponent(group)}`,
      name: group,
      icon,
      children,
    };
  });
}
