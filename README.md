## Commands

- Train parent model
```bash
curl -X POST http://localhost:8000/train-parent
```

- Train child model
```bash
curl -X POST http://localhost:8000/train-child -H "Content-Type: application/json" -d '{"ticker": "AAPL"}'
```

- Training Status parent model
```bash
curl -X GET http://localhost:8000/status/parent
```

- Training Status child model
```bash 
curl -X GET http://localhost:8000/status/aapl
```

- Predict child model
```bash
curl -X POST http://localhost:8000/predict-child -H "Content-Type: application/json" -d '{"ticker": "AAPL"}'
```

- Metrics
```bash
curl -X GET http://localhost:8000/metrics
```

- Health Check
```bash
curl -X GET http://localhost:8000/health
```
