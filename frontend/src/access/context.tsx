import React, { createContext, useContext } from "react";
import type { PageConfig } from "../models/types";

export interface CurrentUser {
  id: string;
  username: string;
  display_name?: string;
  is_superuser: boolean;
  permissions: string[];
}

export function hasPermission(
  user: CurrentUser | null,
  permission: string,
): boolean {
  const domain = permission.split(":")[0];
  return (
    !!user &&
    (user.is_superuser ||
      user.permissions.includes("*") ||
      user.permissions.includes(permission) ||
      user.permissions.includes(`${domain}:*`))
  );
}

export function pagePermission(page: PageConfig): string {
  if (page.path === "/home") return "";
  let domain = page.api.split("/")[1];
  domain =
    (
      {
        "agent-publications": "agents",
        "agent-teams": "agents",
        marketplace: "agents",
        "prompt-templates": "agents",
        "knowledge-bases": "knowledge",
        "model-routing-policies": "models",
        cost: "observability",
        "audit-logs": "audit",
        guardrails: "security",
        scheduler: "runtime",
        "api-keys": "system",
      } as Record<string, string>
    )[domain] || domain;
  return `${domain}:${page.designer ? "write" : "read"}`;
}

export function canVisit(user: CurrentUser | null, page: PageConfig): boolean {
  if (page.path === "/settings/tenants") return !!user?.is_superuser;
  const permission = pagePermission(page);
  return !permission || hasPermission(user, permission);
}

export const AccessContext = createContext<CurrentUser | null>(null);
export function useAccess() {
  const user = useContext(AccessContext);
  return { user, can: (permission: string) => hasPermission(user, permission) };
}
