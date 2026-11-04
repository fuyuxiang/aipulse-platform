import { describe, expect, it } from "vitest";
import {
  canVisit,
  hasPermission,
  type CurrentUser,
} from "../../src/access/context";
import { buildMenus } from "../../src/access/menu";
import { pageConfigs } from "../../src/routes/pageConfig";
const reader: CurrentUser = {
  id: "reader",
  username: "reader",
  is_superuser: false,
  permissions: ["agents:read", "chat:read"],
};
describe("navigation access", () => {
  it("keeps resource views but excludes creation and administrator pages for read-only users", () => {
    expect(hasPermission(reader, "agents:write")).toBe(false);
    expect(
      canVisit(
        reader,
        pageConfigs.find((page) => page.path === "/build/agents")!,
      ),
    ).toBe(true);
    expect(
      canVisit(
        reader,
        pageConfigs.find((page) => page.path === "/build/agents/create")!,
      ),
    ).toBe(false);
    const paths = buildMenus(reader).flatMap(
      (group) => group.children?.map((child) => child.path) || [group.path],
    );
    expect(paths).toContain("/playground");
    expect(paths).not.toContain("/settings/users");
    expect(paths).not.toContain("/settings/tenants");
  });
  it("honors domain wildcards without giving tenant administrators platform tenant management", () => {
    const admin = { ...reader, permissions: ["agents:*", "tenants:*"] };
    expect(hasPermission(admin, "agents:write")).toBe(true);
    expect(
      canVisit(
        admin,
        pageConfigs.find((page) => page.path === "/settings/tenants")!,
      ),
    ).toBe(false);
  });
});
