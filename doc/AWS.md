# AWS Deployment Strategy with Kubernetes (EKS)

This guide outlines the minimal setup required to deploy the stock prediction MLOps pipeline to AWS using Kubernetes (EKS), based on your existing `docker-compose` configuration.

## 🏗 Architecture Overview

We will translate the Docker Compose services into Kubernetes Pods/Deployments.

| Docker Compose Service | AWS / K8s Equivalent | Description |
|-----------------------|----------------------|-------------|
| **FastAPI** | Deployment + Service (LoadBalancer) | The main application logic. |
| **Frontend (Streamlit)** | Deployment + Service (LoadBalancer) | The UI accessible to users. |
| **Redis** | StatefulSet (or AWS ElastiCache) | Caching and Feature Store online store. |
| **Qdrant** | StatefulSet + PVC | Vector database for memory. |
| **Prometheus/Grafana** | Deployments + Services | Monitoring stack. |
| **Volumes (`outputs/`)** | Persistent Volume (EBS/EFS) | Shared storage for model artifacts. |

---

## ✅ Minimal AWS Services Checklist

To meet the requirement of "Minimal Must Services" while using Kubernetes:

1.  **Amazon EKS (Elastic Kubernetes Service)**: The managed Kubernetes control plane.
2.  **Amazon ECR (Elastic Container Registry)**: To store your custom Docker images (`fastapi` and `frontend`).
3.  **Amazon EC2 (Node Groups)**: The actual virtual machines where your K8s pods will run.
    *   *Tip*: Use a Node Group with mixed instances (e.g., `t3.medium` or `t3.large`) to balance cost and performance.
4.  **Amazon EBS (Elastic Block Store)** : To provide extensive persistent storage for your `outputs/`, `qdrant_data`, and `redis_data`.

---

## 🚀 Step-by-Step Setup Guide

### 1. Prerequisites
Install the following CLI tools:
*   `aws` CLI
*   `kubectl`
*   `eksctl` (The easiest way to set up EKS)

### 2. Create the EKS Cluster
Run the following command to create a simple cluster with 2 worker nodes:

```bash
eksctl create cluster \
  --name mlops-stock-cluster \
  --version 1.29 \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.large \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 3 \
  --managed
```

### 3. Container Registry (ECR)
Create repositories for your custom images:

```bash
aws ecr create-repository --repository-name mlops-fastapi
aws ecr create-repository --repository-name mlops-frontend
```

**Build and Push Images:**
Authenticate Docker to ECR:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
```

Build and push:
```bash
# FastAPI
docker build -t mlops-fastapi .
docker tag mlops-fastapi:latest <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/mlops-fastapi:latest
docker push <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/mlops-fastapi:latest

# Frontend
cd frontend
docker build -t mlops-frontend .
docker tag mlops-frontend:latest <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/mlops-frontend:latest
docker push <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/mlops-frontend:latest
```

### 4. Kubernetes Manifests (The K8s "Docker Compose")

Create a file named `k8s-deployment.yaml` with the following content. This translates your `docker-compose` to K8s resources.

**Crucial Note on `outputs/`**: We use a `PersistentVolumeClaim` to ensure your model artifacts persist even if pods restart.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: outputs-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
# REDIS
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis/redis-stack:latest
        ports:
        - containerPort: 6379
---
apiVersion: v1
kind: Service
metadata:
  name: redis
spec:
  ports:
  - port: 6379
  selector:
    app: redis
---
# QDRANT
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
spec:
  serviceName: "qdrant"
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports:
        - containerPort: 6333
        volumeMounts:
        - name: qdrant-storage
          mountPath: /qdrant/storage
  volumeClaimTemplates:
  - metadata:
      name: qdrant-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 5Gi
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant
spec:
  ports:
  - port: 6333
  selector:
    app: qdrant
---
# FASTAPI
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fastapi
  template:
    metadata:
      labels:
        app: fastapi
    spec:
      containers:
      - name: fastapi
        image: <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/mlops-fastapi:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "redis"
        - name: REDIS_PORT
          value: "6379"
        # Mount the secrets/env vars here or use ConfigMap
        volumeMounts:
        - name: outputs-storage
          mountPath: /app/outputs
      volumes:
      - name: outputs-storage
        persistentVolumeClaim:
          claimName: outputs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: fastapi-service
spec:
  type: LoadBalancer
  ports:
  - port: 8000
    targetPort: 8000
  selector:
    app: fastapi
---
# FRONTEND
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/mlops-frontend:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8501
        env:
        - name: API_URL
          value: "http://fastapi-service:8000" 
---
apiVersion: v1
kind: Service
metadata:
  name: frontend-service
spec:
  type: LoadBalancer
  ports:
  - port: 8501
    targetPort: 8501
  selector:
    app: frontend
```

### 5. Deploy
Run:
```bash
kubectl apply -f k8s-deployment.yaml
```

### 6. Access
After a few minutes, AWS will provision Load Balancers for your services. Get the URLs:

```bash
kubectl get services
```

Look for `EXTERNAL-IP` for `frontend-service` and `fastapi-service`.

## 📦 Persistence Note (`outputs/`)
In `docker-compose`, you mapped `./outputs` to host directory. In K8s, we used a **PersistentVolumeClaim (PVC)** named `outputs-pvc`.
- This ensures that when the FastAPI pod writes models/plots to `/app/outputs`, they are saved to an actual EBS disk on AWS.
- If the pod crashes or restarts, the new pod will re-attach to this disk, and your data remains safe.

## 💰 Cost Optimization Tips
- **Shut down when not in use**: EKS clusters cost ~$0.10/hour just for the control plane.
- **Spot Instances**: Use Spot Nodes for training workloads to save up to 90%.
- **Single Node**: For development, you can set `--nodes 1` in `eksctl`.
