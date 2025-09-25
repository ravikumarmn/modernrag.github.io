#!/usr/bin/env python
"""
Advanced example demonstrating the ModernRAG pipeline with custom configurations.

This script shows how to use the ModernRAG library with custom configurations
for retrieval, augmentation, and generation.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Import ModernRAG components
from modernrag.vector_store import (
    check_index_exists,
    similarity_search,
    split_and_upsert_documents,
    VectorStoreConfig
)
from modernrag.generation import (
    retrieve_augment_generate,
    rerank_documents,
    augment_documents,
    generate_response,
    GenerationConfig,
    AugmentationManager,
    GenerationManager
)
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class CustomAugmentationManager(AugmentationManager):
    """Custom augmentation manager with specialized reranking."""
    
    async def rerank_documents(self, query: str, documents, top_k: int = 3):
        """Custom reranking method that prioritizes document recency."""
        # First use the standard reranking
        reranked_docs = await super().rerank_documents(query, documents, top_k)
        
        # Then apply additional custom logic (e.g., prioritize by recency)
        # This is just a placeholder - in a real application, you might
        # sort by document date or other metadata
        
        logger.info("Applied custom reranking logic")
        return reranked_docs


class CustomGenerationManager(GenerationManager):
    """Custom generation manager with specialized response formatting."""
    
    def __init__(self):
        """Initialize with custom configuration."""
        super().__init__()
        # Override the default LLM with a custom configuration
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.5,  # Lower temperature for more focused responses
            max_tokens=2048    # Allow for longer responses
        )
    
    async def generate_response(self, query: str, context: str) -> str:
        """Custom response generation with structured formatting."""
        try:
            # Generate the base response using the parent method
            base_response = await super().generate_response(query, context)
            
            # Add custom formatting or post-processing
            formatted_response = (
                f"📝 **Response Summary**\n\n{base_response}\n\n"
                f"🔍 **Query**: {query}\n"
                f"⏱️ Generated at: {asyncio.get_event_loop().time():.2f}s"
            )
            
            logger.info("Applied custom response formatting")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error in custom response generation: {str(e)}")
            return f"Error generating response: {str(e)}"


async def load_sample_documents() -> List[Document]:
    """Load sample documents for the demo."""
    return [
        Document(
            page_content="""
            Hybrid search combines vector search with traditional keyword search to improve
            retrieval quality. By using both semantic understanding and exact keyword matching,
            hybrid search can handle a wider range of queries effectively.
            """,
            metadata={"source": "Hybrid Search", "page": 1, "date": "2025-08-15"}
        ),
        Document(
            page_content="""
            Query routing is a technique that directs different types of queries to the most
            appropriate retrieval method. For example, factual queries might be routed to a
            knowledge graph, while conceptual queries might use vector search.
            """,
            metadata={"source": "Query Routing", "page": 1, "date": "2025-08-20"}
        ),
        Document(
            page_content="""
            Self-querying retrievers can analyze a user query and automatically generate
            structured filters to narrow down the search space before performing semantic search.
            This combines the benefits of metadata filtering with vector similarity.
            """,
            metadata={"source": "Self-Querying Retrievers", "page": 1, "date": "2025-09-01"}
        ),
        Document(
            page_content="""
            Hypothetical Document Embeddings (HyDE) is a technique where the LLM first generates
            a hypothetical answer document, which is then embedded and used for retrieval.
            This helps bridge the gap between query and document semantic spaces.
            """,
            metadata={"source": "HyDE Technique", "page": 1, "date": "2025-09-10"}
        ),
        Document(
            page_content="""
            Recursive retrieval involves using initial search results to formulate follow-up
            queries, enabling the system to progressively refine its understanding and gather
            more specific information through multiple retrieval steps.
            """,
            metadata={"source": "Recursive Retrieval", "page": 1, "date": "2025-09-15"}
        ),
    ]


async def run_advanced_rag_example():
    """Run an advanced RAG example with custom components."""
    # Define index name
    index_name = "advanced-rag-example-index"
    
    # Ensure index exists
    await check_index_exists(index_name)
    
    # Load and upsert sample documents
    documents = await load_sample_documents()
    logger.info(f"Upserting {len(documents)} sample documents to index {index_name}")
    await split_and_upsert_documents(documents, index_name)
    logger.info("Sample documents upserted successfully")
    
    # Create custom managers
    custom_augmentation_manager = CustomAugmentationManager()
    custom_generation_manager = CustomGenerationManager()
    
    # Example query
    query = "What are advanced retrieval techniques in modern RAG systems?"
    
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    
    # Step 1: Retrieve documents
    retrieved_docs = await similarity_search(
        query=query,
        index_name=index_name,
        k=4,
        score_threshold=0.3
    )
    
    print(f"\nRetrieved {len(retrieved_docs)} documents")
    
    # Step 2: Apply custom reranking
    reranked_docs = await custom_augmentation_manager.rerank_documents(
        query=query,
        documents=retrieved_docs,
        top_k=3
    )
    
    print(f"Reranked to {len(reranked_docs)} most relevant documents")
    
    # Step 3: Apply custom augmentation
    augmented_context = await custom_augmentation_manager.augment_documents(
        query=query,
        documents=reranked_docs
    )
    
    print("\n" + "-" * 40)
    print("AUGMENTED CONTEXT (excerpt):")
    print("-" * 40)
    print(augmented_context[:300] + "..." if len(augmented_context) > 300 else augmented_context)
    
    # Step 4: Generate custom response
    response = await custom_generation_manager.generate_response(
        query=query,
        context=augmented_context
    )
    
    print("\n" + "-" * 40)
    print("GENERATED RESPONSE:")
    print("-" * 40)
    print(response)
    print("=" * 80)


if __name__ == "__main__":
    # Run the async example
    asyncio.run(run_advanced_rag_example())
