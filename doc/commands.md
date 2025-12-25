# 📋 Essential API Commands

Quick reference for the most important API endpoints.

---

## 🏠 Basic Commands

### Health Check
```bash
curl -X GET http://localhost:8000/health
```

### Project Info
```bash
curl -X GET http://localhost:8000/
```

### API Documentation
```bash
open http://localhost:8000/docs
```

---

## 🎯 Training

### Train Parent Model
```bash
curl -X POST http://localhost:8000/train-parent
```
**Description**: Train the S&P 500 base model (required first step)

### Train Child Model
```bash
curl -X POST http://localhost:8000/train-child \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```
**Description**: Train a specific stock model using transfer learning

---

## 🔮 Predictions

### Predict Parent Model
```bash
curl -X POST http://localhost:8000/predict-parent
```

### Predict Child Model
```bash
curl -X POST http://localhost:8000/predict-child \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```
**Note**: Auto-trains if model doesn't exist

---

## 📊 Status & Monitoring

### Check Training Status
```bash
# Parent model status
curl -X GET http://localhost:8000/status/parent

# Child model status
curl -X GET http://localhost:8000/status/aapl
```

### Monitor Parent Model
```bash
curl -X POST http://localhost:8000/monitor/parent
```
**Description**: Run drift detection and agent evaluation

### Monitor Specific Ticker
```bash
curl -X POST http://localhost:8000/monitor/AAPL
```

---

## 🗂️ Outputs & System

### View Outputs Directory
```bash
# List all outputs (shows all tickers)
curl -X GET http://localhost:8000/outputs

# List specific ticker outputs
curl -X GET http://localhost:8000/outputs/AAPL
```
**Description**: View all generated files (models, plots, evaluations)  
**Returns**: Empty list if no outputs exist yet

### Inspect Cache
```bash
# List all cached predictions
curl -X GET http://localhost:8000/system/cache

# Get specific ticker cache
curl -X GET http://localhost:8000/system/cache?ticker=AAPL
```

### View Logs
```bash
curl -X GET http://localhost:8000/system/logs?lines=100
```

### System Reset
```bash
# ⚠️ WARNING: Deletes ALL data
curl -X DELETE http://localhost:8000/system/reset
```

---

## 📈 Monitoring

### Prometheus Metrics
```bash
curl -X GET http://localhost:8000/metrics
```

### Grafana Dashboard
```bash
open http://localhost:3000
```
**Credentials**: admin/admin

---

## 🤖 AI Agent

### Analyze Stock
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "thread_id": "user123"}'
```

---

## 🎨 Frontend

### Main UI
```bash
open http://localhost:8501
```

### Monitoring Dashboard
```bash
open http://localhost:8502
```

---

## 🚀 Quick Start

```bash
# 1. Health check
curl -X GET http://localhost:8000/health

# 2. Train parent model
curl -X POST http://localhost:8000/train-parent

# 3. Check status
curl -X GET http://localhost:8000/status/parent

# 4. Predict (auto-trains child if needed)
curl -X POST http://localhost:8000/predict-child \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'

# 5. View outputs
curl -X GET http://localhost:8000/outputs/AAPL

# 6. Monitor performance
curl -X POST http://localhost:8000/monitor/parent
```

---

## 📝 Important Notes

- **Rate Limits**: Training (5/hour), Predictions (40/hour)
- **Caching**: Predictions cached for 30 minutes
- **Auto-Healing**: Missing models trigger automatic training
- **Async Training**: Use `/status/{task_id}` to check progress
- **Empty Outputs**: `/outputs` returns empty list if no models trained yet

