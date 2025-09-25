"""
Main entry point for the ModernRAG application.

This module demonstrates how to use the async vector store and generation modules
for retrieval-augmented generation tasks.
"""

import os
import asyncio
import logging
import argparse
from dotenv import load_dotenv

# Import our async vector store module
from modernrag.vector_store import (
    check_index_exists,
    get_vector_store,
    similarity_search
)

# Import our async generation module
from modernrag.generation import (
    retrieve_augment_generate,
    rerank_documents,
    augment_documents,
    generate_response
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def search_only(query: str, index_name: str, k: int, score_threshold: float):
    """Perform only the search step of RAG."""
    logger.info(f"Searching for: {query}")
    
    # Perform similarity search
    results = await similarity_search(
        query=query,
        index_name=index_name,
        k=k,
        score_threshold=score_threshold
    )
    
    # Display results
    logger.info(f"Found {len(results)} results")
    
    for i, doc in enumerate(results):
        # Check if the result is a tuple (document, score) or just a document
        if isinstance(doc, tuple) and len(doc) == 2:
            document, score = doc
            score_display = f"Score: {score:.4f}"
        else:
            document = doc
            score_display = "Score: N/A"
            
        print(f"\nResult {i+1} ({score_display}):")
        print("-" * 40)
        print(document.page_content[:200] + "..." if len(document.page_content) > 200 else document.page_content)
        print(f"Source: {document.metadata.get('source', 'Unknown')}, Page: {document.metadata.get('page', 'Unknown')}")
        print("-" * 40)
    
    return results


async def full_rag_pipeline(query: str, index_name: str, k: int, score_threshold: float, rerank_top_k: int):
    """Run the complete RAG pipeline: retrieve, augment, generate."""
    logger.info(f"Running full RAG pipeline for query: {query}")
    
    # Run the complete RAG pipeline
    result = await retrieve_augment_generate(
        query=query,
        index_name=index_name,
        k=k,
        score_threshold=score_threshold,
        rerank_top_k=rerank_top_k
    )
    
    # Display the results
    print("\n" + "=" * 80)
    print(f"QUERY: {result['query']}")
    print("=" * 80)
    
    # Display retrieved documents count
    print(f"\nRetrieved {len(result['retrieved_docs'])} documents")
    
    # Display augmented context
    print("\n" + "-" * 40)
    print("AUGMENTED CONTEXT:")
    print("-" * 40)
    print(result['augmented_context'][:500] + "..." if len(result['augmented_context']) > 500 else result['augmented_context'])
    
    # Display the generated response
    print("\n" + "-" * 40)
    print("GENERATED RESPONSE:")
    print("-" * 40)
    print(result['response'])
    print("\n" + "=" * 80)
    
    return result


async def main():
    """Main async function that demonstrates the RAG pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ModernRAG Demo')
    parser.add_argument('--query', type=str, default="LM-generated results as supervisory signals to fine-tune the embedding model during the RAG process.",
                        help='Query to search for')
    parser.add_argument('--index', type=str, default="langchain-test-index",
                        help='Index name to use')
    parser.add_argument('--k', type=int, default=4,
                        help='Number of documents to retrieve')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Similarity score threshold')
    parser.add_argument('--rerank', type=int, default=3,
                        help='Number of documents to keep after reranking')
    parser.add_argument('--mode', type=str, choices=['search', 'full'], default='full',
                        help='Mode to run: search only or full RAG pipeline')
    
    args = parser.parse_args()
    
    try:
        # Ensure index exists
        await check_index_exists(args.index)
        
        if args.mode == 'search':
            # Run search only
            await search_only(args.query, args.index, args.k, args.threshold)
        else:
            # Run full RAG pipeline
            await full_rag_pipeline(args.query, args.index, args.k, args.threshold, args.rerank)
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())