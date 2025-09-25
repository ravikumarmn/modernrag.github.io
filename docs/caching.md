# Caching

The Caching module provides functionality for storing and retrieving query results to improve performance and reduce redundant operations.

## Configuration

The `CacheConfig` class manages configuration settings for the caching system:

```python
class CacheConfig(BaseSettings):
    """Configuration settings for the caching system."""
    
    # Memory cache settings
    memory_cache_enabled: bool = True
    memory_cache_ttl: int = 3600  # Time-to-live in seconds
    memory_cache_max_size: int = 1000  # Maximum number of items
    
    # Disk cache settings
    disk_cache_enabled: bool = True
    disk_cache_ttl: int = 86400  # Time-to-live in seconds (1 day)
    disk_cache_directory: str = ".cache"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

## Cache Manager

The `CacheManager` class implements the singleton pattern for efficient cache management:

```python
class CacheManager:
    """Manages caching operations with singleton pattern."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._config = get_cache_config()
        self._memory_cache = LRUCache(maxsize=self._config.memory_cache_max_size)
        self._initialized = True
    
    async def get_cached_result(self, query: str):
        """Get cached result for a query."""
        # Implementation details...
    
    async def cache_result(self, query: str, result: str):
        """Cache result for a query."""
        # Implementation details...
```

## Public API Functions

```python
async def get_cached_result(query: str):
    """Get cached result for a query."""
    manager = CacheManager()
    return await manager.get_cached_result(query)

async def cache_result(query: str, result: str):
    """Cache result for a query."""
    manager = CacheManager()
    return await manager.cache_result(query, result)

async def clear_cache():
    """Clear all caches."""
    manager = CacheManager()
    return await manager.clear_cache()

async def clear_expired_cache():
    """Clear expired cache entries."""
    manager = CacheManager()
    return await manager.clear_expired_cache()
```

## Memory Cache

The memory cache uses an LRU (Least Recently Used) strategy to manage cache size:

```python
class LRUCache:
    """LRU cache implementation."""
    
    def __init__(self, maxsize: int = 128):
        self.cache = OrderedDict()
        self.maxsize = maxsize
    
    def get(self, key: str):
        """Get item from cache."""
        if key not in self.cache:
            return None
        
        # Move to end (most recently used)
        value = self.cache.pop(key)
        self.cache[key] = value
        return value
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.maxsize:
            # Remove least recently used item
            self.cache.popitem(last=False)
        
        self.cache[key] = value
```

## Disk Cache

The disk cache provides persistence across application restarts:

```python
class DiskCache:
    """Disk cache implementation."""
    
    def __init__(self, directory: str = ".cache", ttl: int = 86400):
        self.directory = directory
        self.ttl = ttl
        os.makedirs(directory, exist_ok=True)
    
    def _get_cache_path(self, key: str):
        """Get cache file path for a key."""
        # Use hash to avoid invalid filename characters
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.directory, f"{hashed_key}.json")
    
    async def get(self, key: str):
        """Get item from cache."""
        path = self._get_cache_path(key)
        if not os.path.exists(path):
            return None
        
        # Check if expired
        if time.time() - os.path.getmtime(path) > self.ttl:
            os.remove(path)
            return None
        
        # Read from file
        try:
            async with aiofiles.open(path, "r") as f:
                content = await f.read()
                return json.loads(content)
        except:
            return None
    
    async def put(self, key: str, value: Any):
        """Put item in cache."""
        path = self._get_cache_path(key)
        try:
            async with aiofiles.open(path, "w") as f:
                await f.write(json.dumps(value))
        except:
            pass
```

## Usage Example

```python
import asyncio
from modernrag import get_cached_result, cache_result, clear_cache

async def main():
    # Try to get cached result
    query = "What is Retrieval-Augmented Generation?"
    cached_result = await get_cached_result(query)
    
    if cached_result:
        print("Found in cache:", cached_result)
    else:
        # Generate result (in a real application, this would call the RAG pipeline)
        result = "Retrieval-Augmented Generation is a technique that enhances LLMs with external knowledge."
        
        # Cache the result
        await cache_result(query, result)
        print("Cached new result:", result)
    
    # Clear cache (for demonstration)
    await clear_cache()

if __name__ == "__main__":
    asyncio.run(main())
```
