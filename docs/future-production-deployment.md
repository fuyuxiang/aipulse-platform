# 未来生产部署扩展

当前阶段只交付本地源码调试能力，不交付云原生部署配置。

生产阶段建议：

- 数据库替换为 PostgreSQL，并启用连接池、迁移审查和备份。
- Runtime Plane 拆分为执行集群，通过 MQ/Celery/Kafka 接收任务。
- 对象存储替换为 MinIO/S3。
- 向量存储替换为 Milvus。
- 日志接入 ELK/OpenSearch。
- Trace 接入 Jaeger/Tempo。
- Metrics 接入 Prometheus/Grafana。
- 配置接入配置中心和 KMS/Secret Manager。
- 引入 Kubernetes/Helm、Ingress、HPA、NetworkPolicy、ServiceAccount、RBAC 和灰度发布。

