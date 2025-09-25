# ModernRAG Documentation

Welcome to the ModernRAG documentation. ModernRAG is a modern Retrieval-Augmented Generation (RAG) system built with async operations and production-level code.

## Overview

ModernRAG enhances large language models by retrieving relevant information from external knowledge sources before generating responses. This approach combines the strengths of retrieval-based and generation-based methods.

## Key Features

- **Asynchronous API**: All operations support async/await syntax for non-blocking I/O
- **Vector Store**: Efficient document storage and semantic search capabilities
- **Caching System**: Memory and disk caching for improved performance
- **Document Augmentation**: Reranking and synthesis of retrieved documents
- **Generation**: Context-aware response generation with configurable parameters

## Installation

```bash
pip install modernrag
```

## Quick Start

```python
import asyncio
from modernrag import retrieve_augment_generate

async def main():
    query = "What is Retrieval-Augmented Generation?"
    response = await retrieve_augment_generate(query)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

## API Reference

- [Vector Store](vector_store.md): Vector storage and retrieval functionality
- [Generation](generation.md): Document augmentation and response generation
- [Caching](caching.md): Query caching system for improved performance
