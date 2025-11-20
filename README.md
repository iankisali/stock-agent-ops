## Commands

- Train parent model
```bash
curl -X POST http://localhost:8000/train-parent
```

- Train child model
```bash
curl -X POST http://localhost:8000/train-child -H "Content-Type: application/json" -d '{"ticker": "AAPL"}'
```

- Predict parent model
```bash
curl -X POST http://localhost:8000/predict-parent -H "Content-Type: application/json" -d '{"ticker": "AAPL"}'
```

- Predict child model
```bash
curl -X POST http://localhost:8000/predict-child -H "Content-Type: application/json" -d '{"ticker": "AAPL"}'
```

- Metric    
```bash
curl -X GET http://localhost:8000/metrics
```
