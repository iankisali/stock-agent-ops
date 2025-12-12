
import json
import redis
import hashlib
from typing import Optional, Any
from functools import wraps

class RedisSemanticCache:
    """
    Cache for Tool/LLM outputs. 
    Notes: 
    - For true 'Semantic' cache, we would embedding the query and search in Redis Vector Store.
    - For this version, we implement "Exact Match" caching for tool calls to save time.
    """
    def __init__(self, host='redis', port=6379, ttl=3600):
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl

    def get_cache(self, key: str) -> Optional[str]:
        return self.r.get(key)
    
    def set_cache(self, key: str, value: str):
        self.r.setex(key, self.ttl, value)

    @staticmethod
    def hash_input(data: Any) -> str:
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

# Global instance
# We will initialize this in main.py
cache_instance = None
