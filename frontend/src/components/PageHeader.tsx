import { Tag, Typography } from "antd";
import React from "react";

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: string;
  actions?: React.ReactNode;
}): JSX.Element {
  return (
    <header className="page-heading">
      <div>
        <Typography.Title level={2}>{title}</Typography.Title>
        {description && (
          <Typography.Paragraph>{description}</Typography.Paragraph>
        )}
      </div>
      <div className="page-heading-actions">{actions}</div>
    </header>
  );
}

const labels: Record<string, string> = {
  active: "正常",
  enabled: "已启用",
  disabled: "已停用",
  draft: "草稿",
  published: "已发布",
  released: "已发布",
  partial_success: "部分完成",
  completed: "已完成",
  success: "成功",
  running: "运行中",
  pending: "待处理",
  failed: "失败",
  error: "异常",
  cancelled: "已取消",
  waiting_approval: "等待审批",
  blocked: "已阻断",
  archived: "已归档",
  approved: "已批准",
  rejected: "已拒绝",
};
export function StatusTag({ status }: { status?: string }): JSX.Element {
  const color = [
    "active",
    "enabled",
    "completed",
    "success",
    "published",
    "approved",
  ].includes(status || "")
    ? "success"
    : ["failed", "error", "blocked", "rejected"].includes(status || "")
      ? "error"
      : ["waiting_approval", "pending"].includes(status || "")
        ? "warning"
        : status === "running"
          ? "processing"
          : "default";
  return <Tag color={color}>{labels[status || ""] || status || "未设置"}</Tag>;
}
export function formatTime(value?: string): string {
  if (!value) return "—";
  const date = new Date(
    /^\d{4}-\d\d-\d\dT/.test(value) && !/(Z|[+-]\d\d:\d\d)$/.test(value)
      ? `${value}Z`
      : value,
  );
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString("zh-CN", { hour12: false });
}
export function parseObject(value?: string): Record<string, unknown> {
  const parsed: unknown = JSON.parse(value || "{}");
  if (!parsed || Array.isArray(parsed) || typeof parsed !== "object")
    throw new Error("配置必须是 JSON 对象");
  return parsed as Record<string, unknown>;
}
