import { expect, test, type Page } from "@playwright/test";
import { pageConfigs } from "../../src/routes/pageConfig";
async function login(page: Page) {
  await page.goto("/login");
  await page.getByLabel("用户名").fill("admin");
  await page.getByLabel("密码", { exact: true }).fill("admin123456");
  await page.getByRole("button", { name: "登录平台" }).click();
  await expect(
    page.getByRole("heading", { name: "工作台", exact: true }),
  ).toBeVisible();
}

test("resource list is visible immediately and search, edit, disable and delete persist", async ({
  page,
  request,
}, testInfo) => {
  await login(page);
  const headers = {
    Authorization: `Bearer ${await page.evaluate(() => localStorage.getItem("aipulse_access_token"))}`,
  };
  const name = `UX 工具 ${Date.now()}`;
  const created = await request.post("/api/v1/tools", {
    headers,
    data: { name, config: { type: "calculator" } },
  });
  expect(created.ok()).toBeTruthy();
  for (let index = 0; index < 21; index++)
    expect(
      (
        await request.post("/api/v1/tools", {
          headers,
          data: { name: `UX newer ${index}-${Date.now()}` },
        })
      ).ok(),
    ).toBeTruthy();
  await page.goto("/build/tools");
  await expect(
    page.getByRole("heading", { name: "工具", exact: true }),
  ).toBeVisible();
  expect(
    (await page.locator(".ant-table-thead").boundingBox())!.y,
  ).toBeLessThan(350);
  await expect(page.getByText("请求 JSON", { exact: true })).toBeHidden();
  await page.getByRole("searchbox", { name: "按名称搜索" }).fill(name);
  await page.getByRole("searchbox", { name: "按名称搜索" }).press("Enter");
  const row = page.getByRole("row").filter({ hasText: name });
  await expect(row).toBeVisible();
  await row.getByRole("button", { name: /编辑/ }).click();
  const drawer = page.getByRole("dialog");
  await drawer.getByLabel("描述", { exact: true }).fill("可持久化的产品配置");
  await drawer.getByRole("button", { name: /保\s*存/ }).click();
  await expect(row).toContainText("可持久化的产品配置");
  await row.getByRole("button", { name: `更多操作 ${name}` }).click();
  await page.getByRole("menuitem", { name: "停用", exact: true }).click();
  await expect(row).toContainText("已停用");
  await page.reload();
  const record = await request.get(
    `/api/v1/tools/${(await created.json()).id}`,
    { headers },
  );
  expect((await record.json()).enabled).toBe(false);
  await page.getByRole("searchbox", { name: "按名称搜索" }).fill(name);
  await page.getByRole("searchbox", { name: "按名称搜索" }).press("Enter");
  await page.screenshot({ path: testInfo.outputPath("resource-list.png") });
  await row.getByRole("button", { name: `删除 ${name}`, exact: true }).click();
  await page.getByRole("button", { name: /确\s*定/, exact: true }).click();
  await expect(row).toHaveCount(0);
});

test("workspace navigation works on desktop and mobile without document overflow", async ({
  page,
}, testInfo) => {
  await login(page);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await expect(page.locator(".brand-entry")).toContainText("AIPulse");
  await expect(page.locator(".workspace-status")).toContainText("平台服务正常");
  await expect(page.locator(".workspace-page .ant-spin-spinning")).toHaveCount(
    0,
  );
  await page.screenshot({ path: testInfo.outputPath("workspace.png") });
  const search = page.getByRole("combobox", { name: "搜索页面" });
  await search.fill("知识库");
  await page.getByTitle("知识库 · 构建", { exact: true }).click();
  await expect(page).toHaveURL(/\/build\/knowledge$/);
  await expect(
    page.getByRole("heading", { name: "知识库", exact: true }),
  ).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("button", { name: "打开导航", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "打开导航", exact: true }).click();
  await page
    .getByRole("dialog")
    .getByRole("link", { name: "工作台", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "工作台", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect
    .poll(async () => page.evaluate(() => document.documentElement.scrollWidth))
    .toBeLessThanOrEqual(390);
  await page.screenshot({
    path: testInfo.outputPath("workspace-mobile.png"),
    fullPage: true,
  });
});

test("published API can be created and disabled from its dedicated page", async ({
  page,
  request,
}) => {
  await login(page);
  const headers = {
    Authorization: `Bearer ${await page.evaluate(() => localStorage.getItem("aipulse_access_token"))}`,
  };
  const name = `UX API agent ${Date.now()}`;
  expect(
    (await request.post("/api/v1/agents", { headers, data: { name } })).ok(),
  ).toBeTruthy();
  await page.goto("/deploy/api-keys");
  await page.getByRole("button", { name: "发布 API" }).click();
  const modal = page.getByRole("dialog");
  await modal.getByLabel("发布名称").fill(`测试发布 ${name}`);
  await modal.getByLabel("智能体", { exact: true }).click();
  await page.getByText(name, { exact: true }).last().click();
  await modal.getByRole("button", { name: /发\s*布/, exact: true }).click();
  await expect(page.getByRole("dialog").getByLabel("API 密钥")).toHaveValue(
    /^ak-/,
  );
  await page.getByRole("button", { name: "已保存，关闭" }).click();
  const row = page.getByRole("row").filter({ hasText: `测试发布 ${name}` });
  await expect(row).toContainText("/api/v1/published/");
  await row.getByRole("button", { name: /停\s*用/, exact: true }).click();
  await expect(row).toContainText("已停用");
});

test("every navigation destination renders without script or API errors", async ({
  page,
}) => {
  test.setTimeout(120000);
  await login(page);
  const scriptErrors: string[] = [];
  const apiErrors: string[] = [];
  page.on("pageerror", (error) => scriptErrors.push(error.message));
  page.on("response", (response) => {
    if (response.url().includes("/api/v1/") && response.status() >= 400)
      apiErrors.push(`${response.status()} ${response.url()}`);
  });
  for (const path of [
    "/home",
    "/build/agents",
    "/build/workflows",
    "/build/tools",
    "/build/knowledge",
    "/build/prompts",
    "/playground",
    "/playground/multi",
    "/eval",
    "/observe/trace",
    "/observe/cost",
    "/observe/alerts",
    "/observe/audit",
    "/deploy/channels",
    "/deploy/api-keys",
    "/deploy/versions",
    "/settings/models",
    "/settings/model-routing",
    "/settings/guardrails",
    "/settings/security",
    "/settings/scheduler",
    "/settings/tenants",
    "/settings/organizations",
    "/settings/users",
    "/settings/roles",
    "/settings/system",
  ]) {
    const destination = pageConfigs.find((page) => page.path === path)!;
    await page
      .getByRole("combobox", { name: "搜索页面" })
      .fill(destination.title);
    await page
      .getByTitle(`${destination.title} · ${destination.group}`, {
        exact: true,
      })
      .click();
    await expect(page).toHaveURL(new RegExp(`${path}$`));
    await page.waitForLoadState("networkidle");
    await expect(page.locator(".page-loading")).toHaveCount(0);
    await expect(page.locator("main")).toBeVisible();
    await expect(page.getByText("页面加载失败", { exact: true })).toHaveCount(
      0,
    );
  }
  expect(scriptErrors).toEqual([]);
  expect(apiErrors).toEqual([]);
});

test("workflow warns before discarding changes and can run its saved agent node", async ({
  page,
  request,
}, testInfo) => {
  await login(page);
  const headers = {
    Authorization: `Bearer ${await page.evaluate(() => localStorage.getItem("aipulse_access_token"))}`,
  };
  const localModel = (
    await (
      await request.get("/api/v1/models?name=Local%20chat_llm", { headers })
    ).json()
  ).items.find(
    (model: { provider_type: string }) =>
      model.provider_type === "echo_agent_native",
  );
  const created = await request.post("/api/v1/agents", {
    headers,
    data: {
      name: `UX workflow runner ${Date.now()}`,
      config: { model_id: localModel.id, memory_policy: { enabled: false } },
    },
  });
  expect(created.ok()).toBeTruthy();
  const agent = await created.json();
  const workflow = await request.post("/api/v1/workflows", {
    headers,
    data: {
      name: `UX executable ${Date.now()}`,
      spec: {
        nodes: [
          { id: "start", type: "start", label: "开始" },
          { id: "approval", type: "human_approval", label: "人工确认" },
          {
            id: "agent",
            type: "agent",
            label: "已配置 Agent",
            config: { agent_id: agent.id, prompt: "UX saved prompt 42" },
          },
          { id: "end", type: "end", label: "结束" },
        ],
        edges: [
          { source: "start", target: "approval" },
          { source: "approval", target: "agent" },
          { source: "agent", target: "end" },
        ],
      },
    },
  });
  expect(workflow.ok()).toBeTruthy();
  const saved = await workflow.json();
  await page.goto(`/build/workflows/designer?workflow=${saved.id}`);
  await expect(page.getByRole("textbox", { name: "工作流名称" })).toHaveValue(
    saved.name,
  );
  await page
    .getByRole("textbox", { name: "工作流名称" })
    .fill("UX unsaved title");
  await expect(page.getByText("未保存", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "返回列表" }).click();
  await expect(page.getByRole("dialog")).toContainText("当前修改尚未保存");
  await page.getByRole("button", { name: "继续编辑" }).click();
  await expect(page).toHaveURL(/designer/);
  await page.getByRole("button", { name: "保存定义" }).click();
  await expect(page.getByText("工作流已保存", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "校验", exact: false }).click();
  await expect(page.getByText("工作流校验通过", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "试运行", exact: false }).click();
  const modal = page.getByRole("dialog");
  await modal.getByLabel("工作流运行输入").fill("synthetic context");
  await modal.getByRole("button", { name: "开始运行" }).click();
  await expect(modal.getByText("等待审批", { exact: true })).toBeVisible();
  await modal.getByRole("button", { name: "批准并继续", exact: true }).click();
  await expect(modal.getByText("成功", { exact: true })).toBeVisible();
  await expect(
    modal
      .locator(".response-content")
      .filter({ hasText: "UX saved prompt 42" }),
  ).toBeVisible();
  await expect(modal.getByText("已配置 Agent", { exact: true })).toBeVisible();
  await expect(page.locator(".ant-message-notice")).toHaveCount(0);
  await page.screenshot({
    path: testInfo.outputPath("workflow-approval-result.png"),
    fullPage: false,
  });
});

test("model connection can be configured without writing JSON", async ({
  page,
  request,
}) => {
  await login(page);
  await page.goto("/settings/models");
  await page.getByRole("button", { name: "创建模型" }).click();
  const drawer = page.getByRole("dialog");
  const name = `UX model ${Date.now()}`;
  await drawer.getByLabel("名称", { exact: true }).fill(name);
  await drawer.getByLabel("模型标识").fill("synthetic-model");
  await drawer.getByLabel("服务地址").fill("http://127.0.0.1:9999/v1");
  await drawer.getByLabel("认证方式").focus();
  await drawer.getByLabel("认证方式").press("ArrowDown");
  await page.getByText("无需认证", { exact: true }).last().click();
  await drawer.getByRole("button", { name: /保\s*存/, exact: true }).click();
  await expect(page.getByText("模型已保存", { exact: true })).toBeVisible();
  const headers = {
    Authorization: `Bearer ${await page.evaluate(() => localStorage.getItem("aipulse_access_token"))}`,
  };
  const result = await request.get(
    `/api/v1/models?name=${encodeURIComponent(name)}`,
    { headers },
  );
  const model = (await result.json()).items[0];
  expect(model.config.base_url).toBe("http://127.0.0.1:9999/v1");
  expect(model.config.auth_scheme).toBe("none");
  expect(model.model_id).toBe("synthetic-model");
});

test("knowledge base can be configured, populated and retrieved through forms", async ({
  page,
}) => {
  await login(page);
  await page.goto("/build/knowledge");
  await page
    .getByRole("button", { name: "创建知识库", exact: false })
    .first()
    .click();
  const drawer = page.getByRole("dialog");
  const name = `UX knowledge ${Date.now()}`;
  await drawer.getByLabel("名称", { exact: true }).fill(name);
  await drawer.getByLabel("向量模型", { exact: true }).click();
  await page.getByText("Local embedding", { exact: true }).last().click();
  await drawer.getByRole("button", { name: /保\s*存/, exact: true }).click();
  await expect(page.getByText("知识库已保存", { exact: true })).toBeVisible();
  await page.getByRole("searchbox", { name: "按名称搜索" }).fill(name);
  await page.getByRole("searchbox", { name: "按名称搜索" }).press("Enter");
  const row = page.getByRole("row").filter({ hasText: name });
  await row.getByRole("button", { name: /详\s*情/, exact: true }).click();
  const detail = page.getByRole("dialog");
  await detail
    .getByLabel("文件名", { exact: true })
    .fill("synthetic-ledger.txt");
  await detail
    .getByLabel("文档内容", { exact: true })
    .fill("Source S1: Bluebridge has equipment 17 and training 25.");
  await detail.getByRole("button", { name: "上传文档", exact: false }).click();
  await expect(detail.locator(".operation-result")).toBeVisible();
  await detail.getByRole("combobox", { name: "选择操作", exact: true }).focus();
  await detail
    .getByRole("combobox", { name: "选择操作", exact: true })
    .press("ArrowDown");
  await page.getByText("检索测试", { exact: true }).last().click();
  await detail
    .getByLabel("检索问题", { exact: true })
    .fill("Bluebridge equipment training");
  await detail.getByRole("button", { name: "检索测试", exact: false }).click();
  await expect(detail.locator(".operation-result")).toContainText("Bluebridge");
  await expect(detail.locator(".operation-result")).toContainText("17");
});
