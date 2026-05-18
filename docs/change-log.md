# Change Log

## 2026-05-18

新增本地调试版企业级智能体平台目录：

- `backend/`
- `runtime/`
- `frontend/`
- `scripts/`
- `data/`
- `docs/`

`./echo-agent` 未做源码修改。

### echo-agent 修改说明

- 修改文件：无。
- 修改原因：不需要修改，当前通过 `runtime/adapter` 和 editable install/PYTHONPATH 引用。
- 影响范围：无原引擎侵入。
- 回滚方式：删除新增平台目录即可恢复原 `echo-agent` 状态。
- 测试覆盖：runtime adapter 生命周期测试、debug-run 测试、后端 Runtime API 集成测试。

