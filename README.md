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

- Predict parent model
```bash
curl -X POST http://localhost:8000/predict-parent -H "Content-Type: application/json" -d '{"ticker": "parent"}'
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

- Reset
```bash
curl -X DELETE http://localhost:8000/system/reset
```

- Cache
```bash
curl "http://localhost:8000/system/cache"

curl "http://localhost:8000/system/cache?ticker=AAPL"
```