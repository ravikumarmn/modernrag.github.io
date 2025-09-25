#!/usr/bin/env python
"""
Example script demonstrating the ModernRAG pipeline.

This script shows how to use the ModernRAG library to perform
retrieval-augmented generation on a set of documents.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

# Import ModernRAG components
from modernrag.vector_store import (
    check_index_exists,
    similarity_search,
    split_and_upsert_documents
)
from modernrag.generation import retrieve_augment_generate
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def setup_demo_index(index_name: str):
    """Set up a demo index with sample documents."""
    # Sample documents - using clean text without extra whitespace to improve embedding quality
    documents = [
        Document(
            page_content="Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by retrieving relevant information from external knowledge sources before generating responses. This approach helps ground the model's outputs in factual information and reduces hallucinations.",
            metadata={"source": "RAG Overview", "page": 1}
        ),
        Document(
            page_content="The RAG architecture consists of two main components: a retriever and a generator. The retriever identifies relevant documents from a knowledge base, while the generator uses these documents along with the query to produce an informed response.",
            metadata={"source": "RAG Architecture", "page": 1}
        ),
        Document(
            page_content="Modern RAG systems often incorporate additional components such as rerankers, which help prioritize the most relevant documents, and document augmentation, which synthesizes or transforms retrieved information to make it more useful for generation.",
            metadata={"source": "Advanced RAG", "page": 1}
        ),
        Document(
            page_content="One key challenge in RAG systems is balancing retrieval quality with computational efficiency. Retrieving too many documents can introduce noise and slow down the system, while retrieving too few might miss important information.",
            metadata={"source": "RAG Challenges", "page": 1}
        ),
        Document(
            page_content="Recent advancements in RAG include adaptive retrieval, which dynamically determines how many documents to retrieve based on query complexity, and multi-hop retrieval, which performs sequential searches to find information that requires multiple steps of reasoning.",
            metadata={"source": "RAG Advancements", "page": 1}
        ),
    ]
    
    try:
        # Delete the index if it exists to start fresh
        from modernrag.vector_store import delete_index
        try:
            logger.info(f"Attempting to delete existing index: {index_name}")
            await delete_index(index_name)
            logger.info(f"Successfully deleted index: {index_name}")
        except Exception as e:
            logger.info(f"Index deletion failed or index didn't exist: {str(e)}")
        
        # Ensure index exists (will create a new one)
        logger.info(f"Creating fresh index: {index_name}")
        await check_index_exists(index_name)
        
        # Split and upsert documents
        logger.info(f"Upserting {len(documents)} sample documents to index {index_name}")
        success = await split_and_upsert_documents(documents, index_name)
        
        if success:
            logger.info("Sample documents upserted successfully")
            
            # Verify documents were indexed by doing a simple search
            logger.info("Verifying document indexing with a test search...")
            from modernrag.vector_store import similarity_search
            results = await similarity_search(
                query="RAG",
                index_name=index_name,
                k=1
            )
            logger.info(f"Verification search found {len(results)} documents")
            if results:
                logger.info(f"First result content: {results[0][0].page_content[:50]}...")
        else:
            logger.error("Failed to upsert documents")
    except Exception as e:
        logger.error(f"Error setting up demo index: {str(e)}")
        raise


async def run_rag_example():
    """Run a complete RAG example."""
    # Define index name
    index_name = "rag-example-index"
    
    # Set up the demo index with sample documents
    await setup_demo_index(index_name)
    
    # Example queries to test
    queries = [
        "What is Retrieval-Augmented Generation?",
        "What are the main components of a RAG system?",
        "What are some recent advancements in RAG technology?",
        "What challenges do RAG systems face?",
    ]
    
    # Process each query
    for query in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)
        
        try:
            # First do a direct search to verify document retrieval
            logger.info(f"Testing direct search for query: {query}")
            direct_results = await similarity_search(
                query=query,
                index_name=index_name,
                k=3,
                score_threshold=0.3
            )
            
            logger.info(f"Direct search found {len(direct_results)} documents")
            if direct_results:
                for i, (doc, score) in enumerate(direct_results):
                    logger.info(f"Result {i+1} - Score: {score:.4f} - Content: {doc.page_content[:50]}...")
            
            # Run the complete RAG pipeline
            logger.info(f"Running full RAG pipeline for query: {query}")
            result = await retrieve_augment_generate(
                query=query,
                index_name=index_name,
                k=3,
                score_threshold=0.3,
                rerank_top_k=2
            )
            
            # Display retrieved documents count
            print(f"\nRetrieved {len(result['retrieved_docs'])} documents")
            
            # Display detailed information about retrieved documents
            if result['retrieved_docs']:
                print("\n" + "-" * 40)
                print("RETRIEVED DOCUMENTS:")
                print("-" * 40)
                for i, (doc, score) in enumerate(result['retrieved_docs']):
                    print(f"Document {i+1} (Score: {score:.4f})")
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    print(f"Content: {doc.page_content[:100]}..." if len(doc.page_content) > 100 else doc.page_content)
                    print("-" * 20)
            
            # Display augmented context
            if 'augmented_context' in result and result['augmented_context']:
                print("\n" + "-" * 40)
                print("AUGMENTED CONTEXT:")
                print("-" * 40)
                print(result['augmented_context'][:300] + "..." if len(result['augmented_context']) > 300 else result['augmented_context'])
            
            # Display the generated response
            print("\n" + "-" * 40)
            print("GENERATED RESPONSE:")
            print("-" * 40)
            print(result['response'])
        
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            print(f"\nError: {str(e)}")
        
        # Add a delay between queries
        await asyncio.sleep(1)


if __name__ == "__main__":
    # Run the async example
    asyncio.run(run_rag_example())
