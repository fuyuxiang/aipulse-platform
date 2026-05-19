# AIPulse Platform

<p align="center">
  <strong>Enterprise Agent Platform — Control Plane + Runtime Plane</strong>
</p>

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white&style=flat-square">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white&style=flat-square">
  <img alt="React 18" src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black&style=flat-square">
  <img alt="TypeScript" src="https://img.shields.io/badge/TypeScript-5.5-3178C6?logo=typescript&logoColor=white&style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-proprietary-333?style=flat-square">
</p>

---

AIPulse 是企业级智能体管理平台，采用 Control Plane + Runtime Plane 双层架构。Control Plane 负责租户隔离、RBAC、模型治理、工具审批、知识库管理、Workflow 编排和审计合规；Runtime Plane 基于 [Echo Agent](./echo-agent/) 引擎执行 Agent 实例，提供 Agent Loop、多模型路由、MCP/A2A 协议、认知记忆和安全审批能力。平台不侵入引擎代码，通过 Runtime Adapter 桥接两层。

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│              React + TypeScript + Ant Design Pro             │
└────────────────────────────┬────────────────────────────────┘
                             │ REST / WebSocket
┌────────────────────────────▼────────────────────────────────┐
│                    Control Plane (backend/)                   │
│                                                              │
│  Tenancy · RBAC · Model Management · Model Routing           │
│  Tools Center · Memory Center · Knowledge · Workflows        │
│  Evaluation · Alerts · Audit (hash-chain) · Security         │
└────────────────────────────┬────────────────────────────────┘
                             │ Runtime API
┌────────────────────────────▼────────────────────────────────┐
│                   Runtime Plane (runtime/)                    │
│                                                              │
│  EchoAgentAdapter · Instance Isolation · Task Executors      │
│  Workflow Executors · Local Storage · Telemetry Bridge        │
└────────────────────────────┬────────────────────────────────┘
                             │ Python API (editable install)
┌────────────────────────────▼────────────────────────────────┐
│                  Echo Agent Engine (echo-agent/)              │
│                                                              │
│  Agent Loop · Multi-Provider Models · 30+ Tools · MCP/A2A   │
│  4-Layer Memory · Security Approval · Observability          │
└─────────────────────────────────────────────────────────────┘
```

**设计原则：** Control Plane 只管元数据、权限、策略和配置；Runtime Plane 通过 Adapter 注入租户上下文、模型配置、工具权限和资源限制，驱动 Echo Agent 实例执行。两层通过接口契约解耦，引擎代码零修改。

---

## Features

| 领域 | 能力 |
|------|------|
| 多租户与权限 | 租户隔离、RBAC、API Key 管理、操作审计（hash chain 防篡改） |
| 模型治理 | 多 Provider 管理、智能路由、Fallback 策略、用量配额、凭证池 |
| Agent 管理 | 版本化配置、多 Agent 编排、会话管理、运行时实例生命周期 |
| 工具中心 | 内置工具 + MCP 动态扩展、工具权限审批、调用审计 |
| 记忆中心 | 四层认知记忆（Working → Episodic → Semantic → Archival）、混合检索 |
| 知识库 | 文档解析、向量索引、BM25 + 向量混合检索、RAG Pipeline |
| Workflow | 可视化编排（AntV X6）、条件分支、循环、人工审批节点 |
| 评测 | 数据集管理、自动评测、指标对比、回归检测 |
| 安全 | LLM 驱动的风险预审、路径策略、能力声明、管理员审批 |
| 可观测性 | OpenTelemetry Trace、结构化日志、健康检查、告警规则 |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ / npm 9+
- SQLite（开发环境默认，生产可替换为 PostgreSQL/MySQL）

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ../echo-agent
pip install -e .
alembic upgrade head
python ../scripts/init_db.py
python ../scripts/init_admin.py
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://127.0.0.1:3000
```

### Default Credentials

| Field | Value |
|-------|-------|
| Tenant | `default` |
| Username | `admin` |
| Password | `admin123456` |

> 首次登录后请立即修改默认密码。

---

## Project Structure

```
aipulse-platform/
├── backend/            # Control Plane — FastAPI
│   ├── app/
│   │   ├── api/              # REST endpoints
│   │   ├── services/         # Business logic
│   │   ├── repositories/     # Data access
│   │   ├── models/           # SQLAlchemy models
│   │   ├── schemas/          # Pydantic schemas
│   │   ├── auth/             # Authentication & RBAC
│   │   ├── tenancy/          # Multi-tenant isolation
│   │   ├── model_management/ # Provider & model CRUD
│   │   ├── model_routing/    # Smart routing & fallback
│   │   ├── tools_center/     # Tool registry & approval
│   │   ├── memory_center/    # Memory management
│   │   ├── knowledge/        # RAG pipeline
│   │   ├── workflows/        # Workflow engine
│   │   ├── evaluation/       # Eval datasets & runs
│   │   ├── audit/            # Hash-chain audit log
│   │   └── security_center/  # Policy enforcement
│   ├── alembic/              # DB migrations
│   └── tests/
├── runtime/            # Runtime Plane — Echo Agent bridge
│   ├── adapter/              # EchoAgentRuntimeAdapter
│   ├── executors/            # Task & workflow executors
│   └── storage/              # Local storage backends
├── echo-agent/         # Agent Engine (submodule, unmodified)
├── frontend/           # React + TypeScript + Ant Design Pro
│   └── src/
├── scripts/            # Init, start, test, cleanup
├── data/               # SQLite, vectors, logs, traces
└── docs/               # Architecture, API, security, ops
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI, SQLAlchemy 2.0, Alembic, Pydantic v2 |
| Frontend | React 18, TypeScript 5, Ant Design Pro, AntV X6, Formily |
| Agent Engine | Echo Agent (Agent Loop, MCP, A2A, OpenTelemetry) |
| Storage | SQLite (dev) → PostgreSQL/MySQL (prod) |
| Vector | Local FAISS (dev) → Milvus (prod) |
| Queue | LocalTaskExecutor (dev) → Celery/MQ (prod) |
| Observability | JSONL + Trace files (dev) → ELK + Jaeger + Prometheus (prod) |

---

## Development

```bash
# Lint & type check
cd backend && ruff check . && mypy app/
cd frontend && npm run lint

# Tests
cd backend && pytest
cd frontend && npm run test

# E2E
cd frontend && npm run e2e
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | 双层架构设计与模块边界 |
| [API Reference](docs/api.md) | REST API 接口规范 |
| [Local Development](docs/local-development.md) | 本地开发环境搭建 |
| [Security](docs/security.md) | 安全模型与策略 |
| [Testing](docs/testing.md) | 测试策略与覆盖率 |
| [Operations](docs/operations-local.md) | 本地运维手册 |
| [Compliance Matrix](docs/compliance-matrix.md) | 企业合规对照表 |
| [Production Deployment](docs/future-production-deployment.md) | 生产部署规划 |

---

## License

Proprietary. All rights reserved.
