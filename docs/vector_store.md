# Vector Store

The Vector Store module provides functionality for storing, retrieving, and searching document embeddings.

## Configuration

The `VectorStoreConfig` class manages configuration settings for the vector store:

```python
class VectorStoreConfig(BaseSettings):
    """Configuration settings for the vector store."""
    
    # Pinecone configuration
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str = "modernrag"
    
    # Embedding configuration
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536
    
    # Search configuration
    search_top_k: int = 5
    search_score_threshold: float = 0.7
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

## Vector Store Manager

The `VectorStoreManager` class implements the singleton pattern for efficient client reuse:

```python
class VectorStoreManager:
    """Manages vector store operations with singleton pattern."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._config = get_vector_store_config()
        self._pinecone_client = None
        self._index = None
        self._embedding_client = None
        self._initialized = True
    
    async def initialize(self):
        """Initialize the vector store asynchronously."""
        # Implementation details...
```

## Public API Functions

```python
async def initialize_vector_store():
    """Initialize the vector store."""
    manager = VectorStoreManager()
    await manager.initialize()

async def add_documents(documents: List[Document]):
    """Add documents to the vector store."""
    manager = VectorStoreManager()
    return await manager.add_documents(documents)

async def search_documents(query: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None):
    """Search for documents similar to the query."""
    manager = VectorStoreManager()
    return await manager.search_documents(query, top_k, score_threshold)
```

## Usage Example

```python
import asyncio
from modernrag import add_documents, search_documents, initialize_vector_store
from langchain.schema import Document

async def main():
    # Initialize vector store
    await initialize_vector_store()
    
    # Add documents
    documents = [
        Document(page_content="Retrieval-Augmented Generation enhances LLMs with external knowledge"),
        Document(page_content="Vector databases store embeddings for semantic search")
    ]
    await add_documents(documents)
    
    # Search documents
    results = await search_documents("What is RAG?")
    for doc in results:
        print(doc.page_content, doc.metadata.get("score"))

if __name__ == "__main__":
    asyncio.run(main())
```
