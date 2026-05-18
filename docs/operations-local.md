# 本地运维

- 数据库：`data/sqlite/aipulse.db`
- 文件存储：`data/files`
- 向量索引：`data/vector`
- 后端结构化日志：`data/logs/backend.jsonl`
- Trace：`data/traces`
- 导出文件：`data/exports`

常用命令：

```bash
python scripts/init_db.py
python scripts/init_admin.py
python scripts/run_backend.py
python scripts/run_frontend.py
python scripts/run_tests.py
python scripts/reset_local_data.py
```

当前阶段不包含 Docker、Kubernetes、Helm、Ingress、HPA、NetworkPolicy、Prometheus/Grafana/ELK/Jaeger 部署配置。

