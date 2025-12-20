# 🚀 AWS MLOps Deployment: Complete "Zero-to-Hero" Guide

This guide is designed for someone who has **zero AWS experience**. We will go from having just an AWS account to a full Kubernetes Cluster with CI/CD, staying under a **$10 budget**.

---

## 1. 🛠 The "Pre-Flight" Setup (Do this first)

If you've never used AWS, follow these exact steps:

1.  **Create an AWS Account:** Go to [aws.amazon.com](https://aws.amazon.com) and sign up. You will need a Credit/Debit card for verification.
2.  **Create an IAM User (Security):** 
    *   Search for **IAM** in the AWS Console.
    *   Create a user named `mlops-admin`.
    *   Attach the policy: `AdministratorAccess`.
    *   Go to **Security Credentials** and create an **Access Key**. Save the `Access Key ID` and `Secret Access Key` safely!
3.  **Install Tools on your Mac:**
    ```bash
    brew install awscli terraform helm kubectl
    aws configure
    # Enter your Access Key, Secret Key, and region (e.g., us-east-1)
    ```

---

## 2. 🏗 Phase 1: Infrastructure (Terraform)

Terraform is a script that "orders" your servers. It ensures you don't miss any steps.

### Why this is cheap ($):
*   **Spot Instances:** We use `m5.large` Spot instances. They are 70% cheaper than normal servers.
*   **Deploy & Destroy:** We stand up the cluster, record the video, and destroy it immediately.

---

## 3. 🤖 Phase 2: Automation (GitHub Actions)

You don't want to manually upload code. We use GitHub Actions to do it for you.

1.  **GitHub Secrets:** Go to your GitHub Repo > Settings > Secrets.
2.  Add: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.
3.  Now, every time you `git push`, GitHub will:
    *   Build your Docker image.
    *   Push it to **AWS ECR** (Elastic Container Registry).
    *   Tell **AWS EKS** (Kubernetes) to update your app.

---

## 4. 📦 Phase 3: The "S3 Trick" for Models

In MLOps, models are large. Keeping them inside Kubernetes is bad practice.
*   We use the **AWS S3 CSI Driver**.
*   It mounts an S3 bucket to your pod at `/app/outputs`.
*   **The Result:** When your Python code runs `torch.save(model, "outputs/model.pt")`, the file is **instantly uploaded to AWS S3**. This is how real-world AI companies store models.

---

## 🎬 The "Showcase" Script (Follow this for your Video)

### 1. Preparation (Start of Video)
"Today we are deploying our Stock Pipeline to the cloud."
```bash
cd terraform/
terraform apply --auto-approve
```
*(Wait 10-15 mins for EKS to boot. Great time to explain the architecture!)*

### 2. Deployment
"I'm pushing a code change now."
```bash
git add .
git commit -m "Optimize LSTM layers"
git push origin main
```
*(Show the GitHub Actions tab in the browser — it looks very professional for a video.)*

### 3. Verification & S3
"The app is live on AWS. Let's trigger training."
```bash
# Get the AWS LoadBalancer URL
kubectl get svc fastapi-service
# Trigger training
curl -X POST http://<AWS-URL>/train-parent
```
**The "WOW" Moment:** Open your S3 bucket in the browser and refresh. Show the audience the model files appearing there. This proves the pipeline works from end-to-end.

### 4. Cleanup (End of Video)
"Always destroy your infra to save money."
```bash
terraform destroy --auto-approve
```
**Total Cost:** ~$0.50. Perfect.

---

## ⚠️ Beginner Pitfalls to Avoid
*   **Regions:** Always stay in one region (e.g., `us-east-1`).
*   **Cleanup:** If you forget to run `terraform destroy`, AWS will keep billing you!
*   **NVIDIA:** Only use `g4dn` instances if you really need a GPU for the video. Regular CPUs are 10x cheaper for simple LSTM models.
