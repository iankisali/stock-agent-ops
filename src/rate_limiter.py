import time 
from fastapi import HTTPException 

def simple_rate_limit(redis_client, key: str, limit: int, window_sec: int):
    """
    Baby-level rate limiter:
    - key: a string to identify client (IP or endpoint)
    - limit: max requests per window
    - window_sec: time window in seconds
    """
    now = int(time.time())
    window_key = f"rate_limit:{key}:{now // window_sec}"

    count = redis_client.incr(window_key)
    redis_client.expire(window_key, window_sec)
    
    if count > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")