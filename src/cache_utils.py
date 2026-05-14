import json
from functools import wraps
from flask import request
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
import time


@dataclass
class CacheEntry:
    value: object
    expires_at: float | None = None


class LRUCache:
    def __init__(self, capacity=50, ttl_seconds=None):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self.lock = RLock()

    def get(self, key):
        with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                return None

            if entry.expires_at is not None and entry.expires_at < time.time():
                self.cache.pop(key, None)
                return None

            self.cache.move_to_end(key)
            return entry.value

    def put(self, key, value):
        with self.lock:
            expires_at = (
                time.time() + self.ttl_seconds if self.ttl_seconds is not None else None
            )
            self.cache[key] = CacheEntry(value=value, expires_at=expires_at)
            self.cache.move_to_end(key)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)


# Global instances
api_cache = LRUCache(capacity=100, ttl_seconds=300)
model_registry = LRUCache(capacity=10, ttl_seconds=3600)


def cached_api(f):
    """Cache Flask API responses based on request payload and URL"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        key = None
        payload = request.get_json(silent=True)

        if request.method == "POST" and request.is_json and payload:
            try:
                serialized = json.dumps(payload, sort_keys=True)
                key = f"{f.__name__}_{request.path}_{serialized}"
            except TypeError:
                pass
        elif request.method == "GET":
            key = f"{f.__name__}_{request.url}"

        if key:
            cached_resp = api_cache.get(key)
            if cached_resp:
                return cached_resp

        response = f(*args, **kwargs)

        # Support both standard response objects and tuples (response, status_code)
        if key:
            status = getattr(response, "status_code", None)
            if status is None and isinstance(response, tuple):
                status = response[1]

            if status == 200:
                api_cache.put(key, response)

        return response

    return decorated_function
