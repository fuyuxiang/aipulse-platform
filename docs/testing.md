# 测试

## GitHub Actions

平台工作流为 [tests.yml](../.github/workflows/tests.yml)（AIPulse CI）和 [release.yml](../.github/workflows/release.yml)（AIPulse Release）。保留文件路径以延续 GitHub 已有工作流身份与历史记录；原有 RAGFlow 测试、定时发布、自建 `ragflow-test` Runner、Docker Hub / PyPI 发布配置已移除。

### 触发与检查

CI 在 `main` 推送、目标为 `main` 的 PR，以及 Actions 页手动运行时触发。PR 不需要额外标签，新建和重新打开 PR 也会运行；文档提交同样获得完整检查结果。新提交会取消同一分支或 PR 的旧 CI。

| 作业 | 检查内容 |
| --- | --- |
| `Python 3.11` / `Python 3.12` | 安装平台与 Echo Agent 基础依赖；检查 `backend/`、`runtime/`、`scripts/` 的 Ruff，执行后端 mypy、两套 pytest，以及新建 SQLite 的 Alembic 迁移和默认数据初始化。 |
| `Frontend checks and build` | 使用 Node.js 22 和 `npm ci`，执行高危级别依赖审计、类型检查、单元测试与生产构建。 |
| `Browser business tests` | 安装 Chromium 与 Linux 系统依赖，通过 `npm run e2e` 启动临时数据环境并运行完整浏览器回归。 |
| `CI passed` | 检查上述作业均为成功；失败、取消或跳过不能获得通过结果。 |

Python 工具链固定在 [requirements-ci.txt](../requirements-ci.txt)；应用依赖仍由后端与引擎的 `pyproject.toml` 声明。官方 Actions 固定到提交 SHA，运行于 GitHub 托管的 Ubuntu 24.04，无需旧项目 Runner、Docker、真实模型密钥或第三方发布凭证。

在 Actions 的对应运行中可下载以下产物，保留 7 天：

- `python-3.11-reports`、`python-3.12-reports`：后端和 Runtime 的 JUnit XML。
- `aipulse-frontend`：构建后的前端静态资源。
- `browser-test-reports`：Playwright HTML 报告、测试截图及失败 Trace。浏览器测试仅使用隔离的合成数据，不上传业务数据库。

`AIPULSE_E2E_OUTPUT_DIR` 可指定浏览器测试输出目录；默认仍为脚本打印的临时目录。CI 禁止意外提交的 `test.only`。测试临时目录使用操作系统标准目录，可在 macOS 和 Linux 中运行。

分支保护如需阻止未通过检查的合并，可将 `CI passed` 设为 required check；分支保护设置独立于工作流文件，不由脚本自动修改。

### 发布验证

发布标签采用 `vMAJOR.MINOR.PATCH` 或 `vMAJOR.MINOR.PATCH-rc.1` 等格式。标签触发 `release.yml`，先调用同一份 CI；全部通过后打包源码、下载已验证的前端构建并生成 `SHA256SUMS`，上传至该标签的 GitHub Release。预发布标签生成 prerelease，工作流不再定时移动 `nightly` 标签，也不发布 RAGFlow 镜像或 SDK。

默认作业只有 `contents: read`；仅最后的 Release 作业具有 `contents: write`。推送 `main` 不会创建 Release，也不会部署业务环境。

首次迁移时，若 Actions 侧边栏仍以旧名称 `tests`、`release` 显示且标注 Disabled，可分别打开并点击 **Enable workflow**。它们对应上述保留的文件路径，启用后执行 AIPulse 配置。随后在测试工作流中点击 **Run workflow**，选择 `main` 手动验收；发布工作流仍仅由版本标签触发。

工作流启停可在 Actions 页管理，具体方式见 [GitHub 官方说明](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/disable-and-enable-workflows)。工作流通过、分支保护启用和生产部署是不同状态，不能相互替代。

## 本地检查

在仓库根目录安装并使用相同的工具链：

```bash
python -m pip install -r requirements-ci.txt -e ./echo-agent -e ./backend
npm --prefix frontend ci
python scripts/check_all.py
```

`check_all.py` 包含 Python 静态检查、后端和 Runtime 测试、前端类型检查、单元测试及构建。浏览器测试另行执行，命令如下。

后端：

```bash
cd backend
ruff check .
mypy app
pytest
```

Runtime：

```bash
cd runtime
pytest
```

前端：

```bash
cd frontend
npm run lint
npm run test
npm run build
npm run e2e
```

测试覆盖：

- 登录、JWT、RBAC 拒绝、审计 hash chain。
- Agent、模型、知识库、记忆、安全、评测 API。
- OpenAPI 契约关键路径。
- Runtime adapter 生命周期和 echo-agent debug run。
- Workflow DAG 校验、环检测和节点执行。
- LocalVectorStore 向量、关键词、混合检索和删除。
- 前端页面配置、服务层和登录页 E2E。


## 2026-11-04 评审回归

- `backend/tests/integration/test_review_regressions.py`：普通用户授撤权、租户与会话隔离、审批防伪/绑定/到期/单次消费、模型故障、真实下游工具效果、进程重启检查点恢复、取消传播、发布快照及 Key、记忆范围、导入答案评分。
- `runtime/tests/test_policy_bridges.py`：工具 allow/deny、禁用记忆、未知 Provider 禁止静默演示。
- `frontend/tests/e2e/business.spec.ts`：隔离后台和浏览器，从正常菜单操作用户/聊天/Agent 向导/工作流/团队；使用本地演示 Provider，不作为真实模型质量证据。
- `npm run e2e` 自动创建临时 SQLite/文件/向量目录，在 127.0.0.1:3100 和 :8107 启动测试服务；端口应空闲。需要安装与 Playwright 对应的 Chromium，或用 `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` 指定可用本机浏览器。证据位置会打印在日志中。
- `scripts/run_tests.py` 是 pytest/unit 汇总，不含 lint、build、E2E 和真实模型验收，不能据其成功宣称完成全部验收。

真实模型验收为显式 opt-in，使用独立临时数据库，仅发送合成数据：

```bash
PYTHONPATH=backend python3 scripts/review_real_model_smoke.py
# 使用 LLM_BASE_URL / LLM_MODEL / LLM_API_KEY；不打印密钥。
# 已授权且不需鉴权的测试网关：
PYTHONPATH=backend python3 scripts/review_real_model_smoke.py --base-url "$REVIEW_BASE_URL" --model 'Echo Cedar' --tool-model 'Echo Reed' --no-auth --extended
```

extended 模式假设同一测试网关提供 `embed-pro`、`rerank-pro` 和支持函数调用的 tool-model，验证真实文档/查询向量、重排、工具副作用日志和三成员流水线。若模型不支持函数调用或未返回所需调用，验收应失败，不能用回答中“已调用工具”的文字替代工具日志。

## 前端产品化回归（2026-11-04）

- `frontend/tests/e2e/product.spec.ts`：列表首屏层级、跨页服务端搜索、编辑/启停/删除持久化、桌面搜索与手机导航、26 个导航页面的脚本及 API 错误、API 发布与停用、工作流未保存保护及审批后继续、模型连接表单、知识库配置/文档上传/检索。
- `frontend/tests/unit/access.test.ts`：只读角色、领域通配权限、租户管理边界。
- `frontend/tests/unit/markdown.test.tsx`：表格与代码展示，不执行 HTML、不自动加载模型回答中的外部图片。
- 后端新增回归：编辑遮罩配置不覆盖服务端密钥、团队编辑在引用无效时不部分保存、版本列表过滤及跨智能体版本发布拒绝。
- Vitest 使用 `vitest.config.mts`；Node.js 要求见 `frontend/package.json`。依赖升级后必须一起运行类型检查、单元测试、构建和浏览器回归。
