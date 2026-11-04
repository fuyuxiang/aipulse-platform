import { expect, test, type Page } from "@playwright/test";

async function login(page: Page, username = "admin", password = "admin123456") {
  await page.goto("/login");
  await page.getByLabel("用户名").fill(username);
  await page.getByLabel("密码", { exact: true }).fill(password);
  await page.getByRole("button", { name: "登录平台" }).click();
  await expect(page).toHaveURL(/\/home$/);
}

test("ordinary user created in UI can chat, reload history, then loses access after revocation", async ({
  page,
  request,
}, testInfo) => {
  const admin = await request.post("/api/v1/auth/login", {
    data: { username: "admin", password: "admin123456" },
  });
  expect(admin.ok()).toBeTruthy();
  const headers = {
    Authorization: `Bearer ${(await admin.json()).access_token}`,
  };
  const localModel = (
    await (
      await request.get("/api/v1/models?name=Local%20chat_llm", { headers })
    ).json()
  ).items.find(
    (model: { provider_type: string }) =>
      model.provider_type === "echo_agent_native",
  );
  const role = await request.post("/api/v1/roles", {
    headers,
    data: {
      name: "E2E 业务使用者",
      code: `e2e-${Date.now()}`,
      config: { permissions: ["agents:read", "chat:read", "chat:write"] },
    },
  });
  expect(role.ok()).toBeTruthy();
  const agent = await request.post("/api/v1/agents", {
    headers,
    data: {
      name: "E2E 测试助手",
      config: {
        model_id: localModel.id,
        system_prompt: "Synthetic browser test.",
        memory_policy: { enabled: false },
      },
    },
  });
  expect(agent.ok()).toBeTruthy();
  await login(page);
  await page
    .getByRole("menuitem", { name: "setting 设置", exact: true })
    .click();
  await page.getByRole("link", { name: "用户", exact: true }).click();
  await page.getByRole("button", { name: "新建用户" }).click();
  const drawer = page.getByRole("dialog");
  await drawer.getByLabel("登录账号").fill("e2e-caller");
  await drawer.getByLabel("初始密码").fill("e2e-password-123");
  await drawer.getByLabel("显示名称").fill("E2E 普通用户");
  await drawer.getByLabel("角色", { exact: true }).click();
  await page.getByText("E2E 业务使用者", { exact: true }).last().click();
  await page.keyboard.press("Escape");
  await drawer.getByRole("button", { name: /保\s*存/ }).click();
  await expect(page.getByText("用户与角色已保存")).toBeVisible();
  await page.evaluate(() => localStorage.clear());
  await login(page, "e2e-caller", "e2e-password-123");
  await page
    .getByRole("menuitem", { name: "play-circle 使用", exact: true })
    .click();
  await page.getByRole("link", { name: "对话", exact: true }).click();
  await page.getByRole("combobox", { name: "选择 Agent" }).click();
  await page.getByText("E2E 测试助手", { exact: true }).last().click();
  await page.getByRole("button", { name: "新建对话" }).click();
  await page.getByPlaceholder("输入消息...").fill("browser-evidence-42");
  await page.getByRole("button", { name: /发\s*送/ }).click();
  await expect(page.getByText(/local chat_llm response:/)).toBeVisible();
  await page.reload();
  await page.getByText("新对话", { exact: true }).first().click();
  await expect(page.getByText(/local chat_llm response:/)).toBeVisible();
  await page.setViewportSize({ width: 1100, height: 760 });
  const selector = page.getByRole("combobox", { name: "选择 Agent" });
  await expect(selector).toBeInViewport();
  expect((await selector.boundingBox())!.y).toBeGreaterThanOrEqual(64);
  await expect(page.getByPlaceholder("输入消息...")).toBeInViewport();
  await page.screenshot({
    path: testInfo.outputPath("ordinary-user-chat.png"),
    fullPage: true,
  });
  const users = await request.get("/api/v1/users", { headers });
  const user = (await users.json()).items.find(
    (u: { username: string }) => u.username === "e2e-caller",
  );
  expect(
    (
      await request.put(`/api/v1/users/${user.id}/roles`, { headers, data: [] })
    ).ok(),
  ).toBeTruthy();
  expect(
    await page.evaluate(
      async () =>
        (
          await fetch("/api/v1/chat/sessions", {
            headers: {
              Authorization: `Bearer ${localStorage.getItem("aipulse_access_token")}`,
            },
          })
        ).status,
    ),
  ).toBe(403);
});

test("wizard preserves cross-step config and repeated workflow save updates one resource", async ({
  page,
  request,
}, testInfo) => {
  await login(page);
  const headers = {
    Authorization: `Bearer ${await page.evaluate(() => localStorage.getItem("aipulse_access_token"))}`,
  };
  await page.getByRole("menuitem", { name: "build 构建", exact: true }).click();
  await page.getByRole("link", { name: "智能体", exact: true }).click();
  await page.getByRole("button", { name: "创建智能体" }).click();
  await page.getByLabel("智能体名称").fill("E2E 跨步骤助手");
  await page.getByRole("button", { name: "下一步" }).click();
  await page.getByLabel("模型", { exact: true }).click();
  await page
    .getByText("Local chat_llm（本地演示）", { exact: true })
    .last()
    .click();
  await page.getByRole("button", { name: "下一步" }).click();
  await page.getByLabel("系统提示词").fill("preserved-system-prompt");
  for (let i = 0; i < 4; i++)
    await page.getByRole("button", { name: "下一步" }).click();
  await page.getByRole("button", { name: "创建智能体" }).click();
  await expect(page.getByText("智能体创建成功")).toBeVisible();
  const saved = (
    await (await request.get("/api/v1/agents", { headers })).json()
  ).items.find((a: { name: string }) => a.name === "E2E 跨步骤助手");
  expect(saved.config.system_prompt).toBe("preserved-system-prompt");
  expect(saved.config.model_id).toBeTruthy();
  await page.getByRole("link", { name: "工作流", exact: true }).click();
  await page.getByRole("button", { name: "创建工作流" }).click();
  await page
    .locator(".workflow-canvas")
    .getByText("Agent 节点", { exact: true })
    .click();
  const configDrawer = page.getByRole("dialog");
  await configDrawer.getByLabel("节点名称", { exact: true }).fill("验收节点");
  await configDrawer.getByLabel("选择资源", { exact: true }).click();
  await page.getByText("E2E 跨步骤助手", { exact: true }).last().click();
  await configDrawer
    .getByLabel("高级配置 JSON")
    .fill('{"prompt":"workflow-preserved-prompt"}');
  await configDrawer.getByRole("button", { name: "应用配置" }).click();
  await expect(
    page.locator(".workflow-canvas").getByText("验收节点", { exact: true }),
  ).toBeVisible();
  const count = (
    await (await request.get("/api/v1/workflows", { headers })).json()
  ).total;
  await page.getByRole("button", { name: "保存定义" }).click();
  await expect(page.getByText("工作流已保存")).toBeVisible();
  await page.getByRole("button", { name: "保存定义" }).click();
  await expect
    .poll(
      async () =>
        (await (await request.get("/api/v1/workflows", { headers })).json())
          .total,
    )
    .toBe(count + 1);
  const definitions = (
    await (await request.get("/api/v1/workflows", { headers })).json()
  ).items;
  const definition = definitions.find(
    (w: { name: string }) => w.name === "新建工作流",
  );
  expect(
    definition.spec.nodes.find((n: { id: string }) => n.id === "agent").config,
  ).toEqual({ prompt: "workflow-preserved-prompt", agent_id: saved.id });
  await page.reload();
  await page
    .locator(".workflow-canvas")
    .getByText("验收节点", { exact: true })
    .click();
  await expect(
    page.getByRole("dialog").getByLabel("高级配置 JSON"),
  ).toHaveValue(/workflow-preserved-prompt/);
  await page.getByRole("dialog").getByRole("button", { name: "关闭" }).click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect(page.locator(".ant-drawer-content-wrapper")).toBeHidden();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: testInfo.outputPath("workflow-designer.png"),
    fullPage: true,
  });
});

test("team can be created with selected members, executed and viewed in history", async ({
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
  for (const name of ["E2E 整理成员", "E2E 汇总成员"]) {
    expect(
      (
        await request.post("/api/v1/agents", {
          headers,
          data: {
            name,
            config: {
              model_id: localModel.id,
              memory_policy: { enabled: false },
            },
          },
        })
      ).ok(),
    ).toBeTruthy();
  }
  await page
    .getByRole("menuitem", { name: "play-circle 使用", exact: true })
    .click();
  await page.getByRole("link", { name: "团队协作", exact: true }).click();
  await page.getByRole("button", { name: "创建团队" }).click();
  const drawer = page.getByRole("dialog");
  await drawer.getByLabel("团队名称").fill("E2E 流水团队");
  await drawer.getByLabel("通信拓扑").focus();
  await drawer.getByLabel("通信拓扑").press("ArrowDown");
  await page.getByText("流水线", { exact: true }).last().click();
  for (const [index, name] of ["E2E 整理成员", "E2E 汇总成员"].entries()) {
    await drawer.getByRole("button", { name: "添加成员" }).click();
    await drawer.getByLabel("Agent", { exact: true }).nth(index).click();
    await page.getByText(name, { exact: true }).last().click();
    await drawer
      .getByLabel("职责", { exact: true })
      .nth(index)
      .fill(index === 0 ? "整理测试资料" : "汇总整理结果");
  }
  await drawer.getByRole("button", { name: /创\s*建/, exact: true }).click();
  await expect(page.getByText("团队创建成功")).toBeVisible();
  const row = page.getByRole("row").filter({ hasText: "E2E 流水团队" });
  await row.getByRole("button", { name: /编\s*辑/, exact: true }).click();
  const editDrawer = page.getByRole("dialog");
  await editDrawer
    .getByLabel("职责", { exact: true })
    .first()
    .fill("编辑后的职责");
  await editDrawer
    .getByRole("button", { name: /保\s*存/, exact: true })
    .click();
  await expect(page.getByText("团队已保存", { exact: true })).toBeVisible();
  await row.getByRole("button", { name: /执\s*行/, exact: true }).click();
  const modal = page.getByRole("dialog");
  await modal
    .getByPlaceholder("输入任务描述...")
    .fill("Synthetic source S1: samplefact17");
  await modal.getByRole("button", { name: /执\s*行/, exact: true }).click();
  await expect(modal.getByText("已完成", { exact: true })).toBeVisible({
    timeout: 15000,
  });
  await expect(modal.locator(".response-content")).toContainText(
    "samplefact17",
  );
  await modal.getByRole("button", { name: /close|关闭/i }).click();
  await row.getByRole("button", { name: "历史记录" }).click();
  await expect(
    page
      .getByRole("dialog")
      .getByText(/samplefact17/)
      .last(),
  ).toBeVisible();
  await expect
    .poll(async () =>
      Math.round((await page.getByRole("dialog").boundingBox())!.x),
    )
    .toBe(page.viewportSize()!.width - 700);
  await expect(page.getByText("执行完成", { exact: true })).toHaveCount(0);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: testInfo.outputPath("team-history.png"),
    fullPage: false,
  });
});
