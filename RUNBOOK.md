# Customer Churn Prediction API - Operations Runbook

**Version:** 1.0  
**Last Updated:** 2026-10-30  
**On-Call:** mlops-team@example.com  

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Incident Response](#incident-response)
4. [Rollback Procedures](#rollback-procedures)
5. [Common Troubleshooting](#common-troubleshooting)
6. [Backup & Recovery](#backup--recovery)
7. [Database Migrations](#database-migrations)
8. [Scaling Procedures](#scaling-procedures)
9. [Monitoring & Alerts](#monitoring--alerts)
10. [Security](#security)

---

## 1. Overview

The Customer Churn Prediction API serves predictions using a scikit-onnx model deployed via Triton Inference Server. The stack runs on Kubernetes with GPU support.

### Services
| Service | Port | Purpose |
|---------|------|---------|
| Backend API | 8000 | FastAPI application |
| Frontend | 80 | Nginx React/Vue frontend |
| Triton | 8000/8001/8002 | Model inference (HTTP/gRPC/metrics) |
| MLflow | 5000 | Experiment tracking |
| PostgreSQL | 5432 | Metadata storage |
| Redis | 6379 | Request caching |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Visualization |
| Loki | 3100 | Log aggregation |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Users                            │
└──────────────────────┬──────────────────────────────┘
                       │
                   ┌───▼───┐
                   │ Nginx │ (Load Balancer/Ingress)
                   └───┬───┘
        ┌──────────────┼──────────────┐
        │              │              │
     ┌──▼──┐       ┌───▼───┐      ┌───▼───┐
     │Frontend│     │Backend│     │Triton│
     │(nginx) │     │(FastAPI)│    │(GPU)  │
     └──┬────┘     └───┬───┘      └───┬───┘
         │              │              │
         │        ┌────▼────┐          │
         │        │Gunicorn │          │
         │        │Workers  │          │
         │        └────┬────┘          │
         │             │               │
         │        ┌────▼────┐          │
         │        │Artifacts│          │
         │        │(model.joblib)     │
         │        └────┬────┘          │
         │             │               │
         └────────────┼───────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
     ┌──▼──┐      ┌───▼───┐     ┌───▼───┐
     │Redis│      │Postgres│     │ MLflow│
     └─────┘      └────────┘     └───────┘

     ┌──────────────────────────────────────┐
     │  Monitoring Stack                    │
     │  Prometheus + Grafana + Loki + Tempo │
     └──────────────────────────────────────┘
```

---

## 3. Incident Response

### 3.1 Incident Triage Checklist

1. **Identify the issue**
   - Check Grafana dashboards (Overview, GPU Utilization, API Health)
   - Check alert manager for active alerts
   - Check if there are open incidents in PagerDuty

2. **Assess impact**
   - Is the API responding? `curl http://<api-url>/health`
   - Are Triton inference requests timing out?
   - Are error rates elevated? (Check Prometheus: `rate(http_requests_total{status=~"5.."}[5m])`)
   - Are GPU utilization metrics abnormal?

3. **Gather diagnostics**
   ```bash
   # Check backend pods
   kubectl get pods -n churn
   kubectl describe pod <pod-name> -n churn
   kubectl logs <pod-name> -n churn --since=1h

   # Check Triton status
   kubectl exec -n churn <triton-pod> -- tritonserver --version
   kubectl logs <triton-pod> -n churn --since=1h

   # Check GPU status
   kubectl exec -n churn <triton-pod> -- nvidia-smi

   # Check Prometheus
   kubectl port-forward svc/churn-prometheus -n churn 9090:9000 &
   curl http://localhost:9090/-/healthy
   ```

4. **Communicate**
   - Create a Slack thread in #mlops-incidents
   - Post updates every 15 minutes
   - Escalate to engineering lead if unresolved in 1 hour

### 3.2 Escalation Path

| Severity | Response Time | Escalation |
|----------|--------------|------------|
| P0 - Critical (API down) | 5 min | Engineering Lead → CTO |
| P1 - High (Degraded performance) | 15 min | Engineering Lead |
| P2 - Medium (Partial failure) | 1 hour | On-call Engineer |
| P3 - Low (Minor issue) | 4 hours | Backlog |

### 3.3 Incident Template

```markdown
**Incident #XXXX - [Brief Description]**

- **Started:** [timestamp]
- **Detected by:** [alert/manual/other]
- **Affected Services:** [list services]
- **Impact:** [user impact description]

**Timeline:**
- [timestamp] - Issue detected
- [timestamp] - Investigation started
- [timestamp] - Root cause identified
- [timestamp] - Fix applied
- [timestamp] - Service restored

**Root Cause:** [description]

**Resolution:** [what fixed it]
```

---

## 4. Rollback Procedures

### 4.1 Rollback Backend

1. Identify the last known good deployment:
   ```bash
   kubectl rollout history deployment/churn-backend -n churn
   ```

2. Rollback to the last revision:
   ```bash
   kubectl rollout undo deployment/churn-backend -n churn
   ```

3. Or rollback to a specific revision:
   ```bash
   kubectl rollout undo deployment/churn-backend -n churn --to-revision=5
   ```

4. Verify rollback:
   ```bash
   kubectl rollout status deployment/churn-backend -n churn
   ```

### 4.2 Rollback Model (Triton)

1. List available model versions:
   ```bash
   kubectl exec -n churn <triton-pod> -- ls /models
   ```

2. If using model versions:
   ```bash
   # Unload the problematic version
   curl -X POST http://<triton-service>:8000/v2/repository/models/churn_v1/unload
   # Load the previous version
   curl -X POST http://<triton-service>:8000/v2/repository/models/churn_v2/load
   ```

3. Or update the model repository:
   ```bash
   kubectl patch deployment churn-triton-standalone -n churn -p '{
     "spec": {
       "template": {
         "spec": {
           "containers": [{
             "name": "triton-server",
             "volumeMounts": [{
               "name": "model-repo",
               "mountPath": "/models"
             }]
           }]
         }
       }
     }
   }'
   ```

### 4.3 Rollback Database Migration

```bash
# Navigate to migration directory
cd k8s/migrations

# Rollback the last migration
python manage.py migrate <app_name> <previous_migration_name>

# Or if using alembic:
alembic downgrade -1
```

### 4.4 Automated Rollback

The CI/CD pipeline includes automated rollback on failed health checks:
```yaml
# In the deployment pipeline
- name: Health check
  run: |
    if ! curl -f http://api.churn.example.com/health; then
      kubectl rollout undo deployment/churn-backend -n churn
      exit 1
    fi
```

---

## 5. Common Troubleshooting

### 5.1 High API Latency

**Symptoms:** P99 latency > 1s, timeout errors in logs

**Checklist:**
1. Check Triton queue depth:
   ```promql
   nv_gpu_utilization > 85
   ```
2. Check GPU utilization:
   ```bash
   kubectl exec -n churn <pod> -- nvidia-smi
   ```
3. Check if auto-scaling is working:
   ```bash
   kubectl get hpa -n churn
   ```
4. Consider scaling the backend pods:
   ```bash
   kubectl scale deployment/churn-backend -n churn --replicas=5
   ```

### 5.2 Model Loading Errors

**Symptoms:** 503 errors, Triton reports model not found

**Checklist:**
1. Verify model exists in repository:
   ```bash
   kubectl exec -n churn <triton-pod> -- ls /models/
   ```
2. Check Triton logs:
   ```bash
   kubectl logs <triton-pod> -n churn
   ```
3. Verify model config.pbtxt:
   - Correct platform (`onnxruntime_onnx`)
   - Input/output names match model
   - Tensor names are correct

### 5.3 GPU Not Available

**Symptoms:** Pods stuck in Pending, GPU metrics unavailable

**Checklist:**
1. Verify GPU nodes exist:
   ```bash
   kubectl get nodes -l kubernetes.io/accelerator=nvidia.com/gpu
   ```
2. Check node capacity:
   ```bash
   kubectl describe node <gpu-node>
   ```
3. Verify GPU driver is installed:
   ```bash
   kubectl exec -n churn <pod> -- nvidia-smi
   ```

### 5.4 Database Connection Failures

**Symptoms:** 500 errors, database timeout errors

**Checklist:**
1. Check PostgreSQL status:
   ```bash
   kubectl exec -n churn <postgres-pod> -- pg_isready
   ```
2. Check connection pool:
   ```bash
   kubectl exec -n churn <postgres-pod> -- psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"
   ```
3. Check credentials:
   ```bash
   kubectl get secret churn-secrets -n churn -o yaml
   ```

### 5.5 Redis Cache Failures

**Symptoms:** Slow response times, cache miss errors

**Checklist:**
1. Check Redis connectivity:
   ```bash
   kubectl exec -n churn <redis-pod> -- redis-cli ping
   ```
2. Check memory:
   ```bash
   kubectl exec -n churn <redis-pod> -- redis-cli info memory
   ```

---

## 6. Backup & Recovery

### 6.1 Model Artifact Backup

**Scheduled:** Daily at 02:00 UTC

```bash
# Backup model artifacts to S3/MinIO
dvc push -r origin

# Or copy artifacts directly:
kubectl cp churn/model-bucket /tmp/model-backup
aws s3 sync /tmp/model-backup s3://churn-models/backup/$(date +%Y%m%d)/
```

**Recovery:**
```bash
# Restore from S3/MinIO
aws s3 sync s3://churn-models/backup/<date> /tmp/model-restore/
kubectl cp /tmp/model-restore churn/model-bucket
dvc pull -r origin
```

### 6.2 Database Backup

**Scheduled:** Daily at 01:00 UTC

```bash
# Backup PostgreSQL
kubectl exec -n churn <postgres-pod> -- \
  pg_dump -U postgres churn_db | \
  gzip > /backups/churn_db_$(date +%Y%m%d).sql.gz

# Upload to S3
aws s3 cp /backups/ s3://churn-backups/ --recursive
```

**Recovery:**
```bash
# Download from S3
aws s3 cp s3://churn-backups/churn_db_<date>.sql.gz /tmp/

# Restore
kubectl cp /tmp/churn_db_<date>.sql.gz churn/postgres:/tmp/
kubectl exec -n churn <postgres-pod> -- bash -c "
  gunzip /tmp/churn_db_<date>.sql.gz && \
  pg_restore -U postgres -d churn_db /tmp/churn_db_<date>.sql
"
```

### 6.3 MLflow Backup

**Scheduled:** Daily at 03:00 UTC

```bash
# Backup MLflow artifacts and database
kubectl exec -n churn <mlflow-pod> -- \
  pg_dump -U mlflow_user mlflow_db | \
  gzip > /backups/mlflow_$(date +%Y%m%d).sql.gz

kubectl cp churn/mlflow-artifacts /tmp/mlflow-backup
aws s3 sync /tmp/mlflow-backup s3://churn-mlflow/backup/$(date +%Y%m%d)/
```

---

## 7. Database Migrations

### 7.1 Running Migrations

```bash
# Apply all pending migrations
kubectl exec -n churn <backend-pod> -- alembic upgrade head

# Check current migration
kubectl exec -n churn <backend-pod> -- alembic current
```

### 7.2 Migration Rollback

```bash
# Rollback one migration
kubectl exec -n churn <backend-pod> -- alembic downgrade -1

# Rollback to specific migration
kubectl exec -n churn <backend-pod> -- alembic downgrade <revision>
```

### 7.3 Migration Best Practices

- Always backup database before running migrations
- Run migrations during low-traffic periods
- Use `--sql` flag first to review SQL: `alembic upgrade head --sql`
- Test migrations on a copy of production data
- Have a rollback plan for each migration

---

## 8. Scaling Procedures

### 8.1 Manual Scaling

```bash
# Scale backend pods
kubectl scale deployment/churn-backend -n churn --replicas=N

# Scale Triton pods
kubectl scale deployment/churn-triton-standalone -n churn --replicas=M
```

### 8.2 Resource Adjustments

```bash
# Update resource limits for backend
kubectl patch deployment churn-backend -n churn -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "backend",
          "resources": {
            "requests": {"cpu": "2", "memory": "4Gi", "nvidia.com/gpu": 1},
            "limits": {"cpu": "4", "memory": "8Gi", "nvidia.com/gpu": 1}
          }
        }]
      }
    }
  }
}'
```

### 8.3 HPA Configuration

The HPA is configured to scale based on:
- CPU utilization > 70%
- Memory utilization > 80%
- GPU utilization > 70%
- Request latency > 500ms

View HPA status:
```bash
kubectl get hpa -n churn
kubectl describe hpa churn-backend-hpa -n churn
```

---

## 9. Monitoring & Alerts

### 9.1 Key Metrics to Watch

| Metric | Threshold | Alert Level |
|--------|-----------|-------------|
| API P99 Latency | > 1s | Critical |
| API Error Rate | > 5% | Critical |
| GPU Utilization | > 85% | Critical |
| GPU Memory | > 90% | Critical |
| Pod Restarts | > 3/hour | Warning |
| Queue Depth | > 100 | Warning |
| DB Connections | > 80% capacity | Warning |

### 9.2 Grafana Dashboards

- **Churn Prediction API - Overview**: Main dashboard with latency, error rate, throughput
- **GPU Monitoring**: GPU utilization, memory, temperature
- **Triton Inference**: Model inference latency, request rate, queue depth
- **Infrastructure**: Node metrics, pod status, resource usage

### 9.3 Alert Acknowledgment

```bash
# Silence alerts during maintenance
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: maintenance-silence
  namespace: churn
spec:
  groups:
  - name: maintenance
    rules:
    - alert: ScheduledMaintenance
      expr: ON() vector(1)
      annotations:
        summary: "Scheduled maintenance"
EOF
```

---

## 10. Security

### 10.1 Secret Rotation

```bash
# Rotate database password
kubectl create secret generic churn-secrets \
  --namespace=churn \
  --from-literal=database-url=postgresql://postgres:newpass@postgres:5432/churn_db \
  --from-literal=redis-url=redis://:newpass@redis:6379/0 \
  --from-literal=api-key=newapikey123 \
  --dry-run=client -o yaml > /tmp/new-secrets.yaml

# Seal secret with SealedSecrets
kubectl create secret generic churn-secrets \
  --namespace=churn \
  --from-literal=database-url=postgresql://postgres:newpass@postgres:5432/churn_db \
  | kubeseal --format yaml > kubernetes/sealed-secrets.yaml

kubectl apply -f kubernetes/sealed-secrets.yaml
kubectl rollout restart deployment/churn-backend -n churn
```

### 10.2 Vulnerability Scanning

- Trivy scans images on every CI run
- TruffleHog scans for secrets in every PR
- Results reported to GitHub Security tab

### 10.3 Network Policies

```yaml
# Default deny all traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny
  namespace: churn
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

---

## Appendix

### A. Useful Commands

```bash
# View all pods
kubectl get pods -n churn -o wide

# Port forward for local access
kubectl port-forward svc/churn-backend-svc -n churn 8000:8000
kubectl port-forward svc/churn-prometheus -n churn 9090:9090
kubectl port-forward svc/churn-grafana -n churn 3000:3000

# Restart a deployment
kubectl rollout restart deployment/churn-backend -n churn

# Check deployment status
kubectl rollout status deployment/churn-backend -n churn
```

### B. Contact Information

| Role | Name | Slack | Phone |
|------|------|-------|-------|
| Engineering Lead | [Name] | @ml-engineer-lead | [phone] |
| On-call (Primary) | Rotating | @oncall-ml | [pager] |
| On-call (Secondary) | Rotating | @oncall-ml-backup | [pager] |
| Security Team | Security | @security-team | [phone] |

### C. Documentation References

- [Triton Inference Server Docs](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Kubernetes GPU Docs](https://kubernetes.io/docs/tasks/manage-gpus/)
- [Prometheus Alerting](https://prometheus.io/docs/alerting/latest/overview/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)