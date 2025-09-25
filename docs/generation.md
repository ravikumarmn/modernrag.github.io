# Generation

The Generation module provides functionality for augmenting retrieved documents and generating responses.

## Configuration

The `GenerationConfig` class manages configuration settings for document augmentation and response generation:

```python
class GenerationConfig(BaseSettings):
    """Configuration settings for document augmentation and response generation."""
    
    # LLM configuration
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1000
    
    # System prompt
    system_prompt: str = "You are a helpful assistant that provides accurate information based on the provided context."
    
    # Augmentation settings
    rerank_documents: bool = True
    synthesize_documents: bool = True
    max_documents_for_reranking: int = 10
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

## Document Augmentation

The document augmentation process involves:

1. **Reranking**: Using an LLM to prioritize the most relevant documents
2. **Synthesis**: Extracting and combining information from multiple sources

```python
async def rerank_documents(query: str, documents: List[Document], config: Optional[GenerationConfig] = None):
    """Rerank documents based on relevance to the query using an LLM."""
    # Implementation details...

async def synthesize_documents(query: str, documents: List[Document], config: Optional[GenerationConfig] = None):
    """Synthesize information from multiple documents."""
    # Implementation details...
```

## Response Generation

The response generation process uses the retrieved and augmented documents to generate a comprehensive answer:

```python
async def generate_response(query: str, documents: List[Document], config: Optional[GenerationConfig] = None):
    """Generate a response based on the query and documents."""
    # Implementation details...
```

## Complete RAG Pipeline

The `retrieve_augment_generate` function implements the complete RAG pipeline:

```python
async def retrieve_augment_generate(query: str, config: Optional[GenerationConfig] = None):
    """
    Implement the complete RAG pipeline: retrieve → rerank → augment → generate.
    
    Args:
        query: The user query
        config: Optional generation configuration
        
    Returns:
        Generated response based on retrieved and augmented documents
    """
    # Check cache first
    cached_result = await get_cached_result(query)
    if cached_result:
        return cached_result
    
    # Retrieve documents
    documents = await search_documents(query)
    
    # Rerank documents if enabled
    if config is None:
        config = get_generation_config()
    
    if config.rerank_documents and len(documents) > 0:
        documents = await rerank_documents(query, documents, config)
    
    # Synthesize documents if enabled
    if config.synthesize_documents and len(documents) > 0:
        synthesized_content = await synthesize_documents(query, documents, config)
    else:
        synthesized_content = None
    
    # Generate response
    response = await generate_response(query, documents, config, synthesized_content)
    
    # Cache result
    await cache_result(query, response)
    
    return response
```

## Usage Example

```python
import asyncio
from modernrag import retrieve_augment_generate, GenerationConfig

async def main():
    # Create custom configuration
    config = GenerationConfig(
        llm_model="gpt-4",
        llm_temperature=0.2,
        rerank_documents=True,
        synthesize_documents=True
    )
    
    # Process query
    query = "What are the advantages of Retrieval-Augmented Generation?"
    response = await retrieve_augment_generate(query, config)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```
