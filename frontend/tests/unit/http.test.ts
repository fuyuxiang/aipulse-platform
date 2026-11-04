import { afterEach, describe, expect, it, vi } from "vitest";
import { api, getToken } from "../../src/services/http";
afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});
describe("HTTP authentication and errors", () => {
  it("returns an empty token when storage has no token", () => {
    expect(getToken()).toBe("");
  });
  it("expires the local session on a rejected token", async () => {
    localStorage.setItem("aipulse_access_token", "synthetic-token");
    const listener = vi.fn();
    window.addEventListener("aipulse:unauthorized", listener, { once: true });
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(JSON.stringify({ detail: "expired" }), { status: 401 }),
      ),
    );
    await expect(api.get("/agents")).rejects.toMatchObject({
      status: 401,
      message: "expired",
    });
    expect(getToken()).toBe("");
    expect(listener).toHaveBeenCalledOnce();
  });
  it("reports forbidden writes without treating them as successful or logging out", async () => {
    localStorage.setItem("aipulse_access_token", "synthetic-token");
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(JSON.stringify({ message: "forbidden" }), {
            status: 403,
          }),
      ),
    );
    await expect(api.create("/agents", { name: "test" })).rejects.toMatchObject(
      { status: 403, message: "当前账号没有执行此操作的权限" },
    );
    expect(getToken()).toBe("synthetic-token");
  });
});
