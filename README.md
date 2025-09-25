# ModernRAG

A modern Retrieval-Augmented Generation (RAG) system built with async operations and production-level code.

## Features

- **Asynchronous API**: All operations support async/await syntax for non-blocking I/O
- **Vector Store**: Efficient document storage and semantic search capabilities
- **Caching System**: Memory and disk caching for improved performance
- **Document Augmentation**: Reranking and synthesis of retrieved documents
- **Generation**: Context-aware response generation with configurable parameters

## Installation

```bash
# Clone the repository
git clone https://github.com/modernrag/modernrag.github.io.git
cd modernrag.github.io

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
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

## Examples

Check out the `examples` directory for more detailed usage examples:

- `rag_example.py`: Basic RAG pipeline usage
- `advanced_rag.py`: Advanced features and customization

## Configuration

Copy the `.env.example` file to `.env` and update the values with your API keys and configuration settings.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# modernrag.github.io
# modernrag.github.io
