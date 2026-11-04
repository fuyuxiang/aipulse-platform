# AIPulse Frontend

React 18、TypeScript、Ant Design、React Router 7 与 X6。首页、资源管理、智能体向导、工作流编辑器、对话和团队协作使用独立的页面结构；重型页面按需加载。

需要 Node.js 20.19+（20.x）或 22.12+。

```bash
cd frontend
npm ci
npm run dev
```

开发入口为 `http://127.0.0.1:3000`，`/api`、`/health` 与 WebSocket 代理至 `http://127.0.0.1:8000`。后端需另行启动。初始化的 Local 模型用于演示；真实模型在「设置 → 模型」配置服务地址、模型标识和认证方式。

```bash
npm run lint
npm run test
npm run build
npm run e2e
```

`npm run e2e` 自动启动使用临时 SQLite 与独立数据目录的前后端，使用 3100/8107 端口；请先安装 Playwright 对应的 Chromium，或设置 `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH`。测试不会使用现有业务数据库。截图保存在日志打印的临时证据目录。

生产构建输出到 `dist/`，每次构建清理旧资源。静态服务器需支持 SPA 路由回退，并配置 API 反向代理。应用中的导航根据 `/auth/me` 权限展示，创建与变更入口按权限启用，服务端仍执行实际鉴权。

核心入口：

- 工作台 `/home`：快捷操作、最近资源、可访问资源数量。
- 使用：对话 `/playground`、团队协作 `/playground/multi`。
- 构建：智能体、工作流、工具、知识库、提示词。
- 发布：版本发布 `/deploy/versions`、API 发布与凭证 `/deploy/api-keys`。
- 评测、观测与设置保留原有业务入口和旧路径重定向。

具体整改和验收见 `docs/reviews/aipulse-frontend-acceptance.md`。
