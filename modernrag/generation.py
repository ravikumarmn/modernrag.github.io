"""
Generation Module for Modern RAG Application

This module provides async interfaces for augmenting retrieved documents
and generating responses using LLMs with the retrieved context.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from functools import lru_cache

# Third-party imports
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field
from pydantic_settings import BaseSettings

# Import our vector store module
from modernrag.vector_store import vector_store_manager, similarity_search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class GenerationConfig(BaseSettings):
    """Configuration settings for generation operations."""
    llm_model: str = Field("gpt-4o", env="LLM_MODEL")
    temperature: float = Field(0.7, env="LLM_TEMPERATURE")
    max_tokens: int = Field(1024, env="LLM_MAX_TOKENS")
    system_prompt: str = Field(
        "You are a helpful AI assistant that provides accurate information based on the context provided.",
        env="SYSTEM_PROMPT"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_generation_config() -> GenerationConfig:
    """Get the generation configuration."""
    return GenerationConfig()


@lru_cache()
def get_llm() -> ChatOpenAI:
    """Get the LLM instance."""
    config = get_generation_config()
    return ChatOpenAI(
        model=config.llm_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )


class AugmentationManager:
    """Manages document augmentation operations."""
    
    def __init__(self):
        """Initialize the augmentation manager."""
        self.config = get_generation_config()
        self.llm = get_llm()
    
    async def rerank_documents(
        self, 
        query: str, 
        documents: List[Tuple[Document, float]],
        top_k: int = 3
    ) -> List[Document]:
        """Rerank documents based on relevance to the query.
        
        Args:
            query: The user query
            documents: List of (document, score) tuples from vector search
            top_k: Number of documents to return after reranking
            
        Returns:
            List of reranked documents
        """
        try:
            # Extract just the documents from the tuples
            docs = [doc for doc, _ in documents]
            
            if len(docs) <= top_k:
                return docs
            
            # Create a prompt for reranking
            rerank_prompt = PromptTemplate.from_template(
                """You are an expert at determining relevance of documents to a query.
                Query: {query}
                
                Below are several documents with their content. Rank them by relevance to the query,
                with the most relevant first. Return only the document numbers in order of relevance,
                separated by commas (e.g., "3,1,4,2").
                
                {document_list}
                """
            )
            
            # Create the document list for the prompt
            document_texts = []
            for i, doc in enumerate(docs):
                document_texts.append(f"Document {i+1}:\n{doc.page_content}\n")
            
            document_list = "\n\n".join(document_texts)
            
            # Generate the reranking
            messages = [
                SystemMessage(content="You are an expert at determining relevance of documents to a query."),
                HumanMessage(content=rerank_prompt.format(
                    query=query,
                    document_list=document_list
                ))
            ]
            
            response = await asyncio.to_thread(
                self.llm.invoke,
                messages
            )
            
            # Parse the response to get the reranked order
            reranked_indices = [int(idx.strip()) - 1 for idx in response.content.split(',')]
            
            # Return the reranked documents, limited to top_k
            reranked_docs = [docs[idx] for idx in reranked_indices if idx < len(docs)][:top_k]
            
            logger.info(f"Reranked {len(docs)} documents to {len(reranked_docs)} most relevant")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Failed to rerank documents: {str(e)}")
            # Fall back to returning the original documents sorted by score
            sorted_docs = sorted(documents, key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in sorted_docs[:top_k]]
    
    async def augment_documents(
        self, 
        query: str, 
        documents: List[Document]
    ) -> str:
        """Augment documents by extracting and synthesizing relevant information.
        
        Args:
            query: The user query
            documents: List of documents to augment
            
        Returns:
            Augmented context as a string
        """
        try:
            # Create a prompt for augmentation
            augment_prompt = PromptTemplate.from_template(
                """You are an expert at extracting and synthesizing relevant information from documents.
                
                User Query: {query}
                
                Below are several documents with their content. Extract and synthesize the most relevant
                information from these documents that would help answer the user's query.
                
                {document_list}
                
                Synthesized Information:
                """
            )
            
            # Create the document list for the prompt
            document_texts = []
            for i, doc in enumerate(documents):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                document_texts.append(
                    f"Document {i+1} (Source: {source}, Page: {page}):\n{doc.page_content}\n"
                )
            
            document_list = "\n\n".join(document_texts)
            
            # Generate the augmented context
            messages = [
                SystemMessage(content="You are an expert at extracting and synthesizing relevant information."),
                HumanMessage(content=augment_prompt.format(
                    query=query,
                    document_list=document_list
                ))
            ]
            
            response = await asyncio.to_thread(
                self.llm.invoke,
                messages
            )
            
            augmented_context = response.content
            
            logger.info(f"Augmented {len(documents)} documents into synthesized context")
            return augmented_context
            
        except Exception as e:
            logger.error(f"Failed to augment documents: {str(e)}")
            # Fall back to a simple concatenation of documents
            return "\n\n".join(
                f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
                for doc in documents
            )


class GenerationManager:
    """Manages response generation operations."""
    
    def __init__(self):
        """Initialize the generation manager."""
        self.config = get_generation_config()
        self.llm = get_llm()
        self.augmentation_manager = AugmentationManager()
    
    async def generate_response(
        self, 
        query: str, 
        context: str
    ) -> str:
        """Generate a response to the query using the provided context.
        
        Args:
            query: The user query
            context: The context information to use for generation
            
        Returns:
            Generated response as a string
        """
        try:
            # Create a prompt for generation
            generation_prompt = PromptTemplate.from_template(
                """You are a helpful AI assistant that provides accurate information based on the context provided.
                
                Context information:
                {context}
                
                User Query: {query}
                
                Provide a helpful, accurate, and concise response to the user's query based on the context information.
                If the context doesn't contain relevant information to answer the query, acknowledge this and provide
                general information if possible, or suggest what additional information might be needed.
                
                Response:
                """
            )
            
            # Generate the response
            messages = [
                SystemMessage(content=self.config.system_prompt),
                HumanMessage(content=generation_prompt.format(
                    query=query,
                    context=context
                ))
            ]
            
            response = await asyncio.to_thread(
                self.llm.invoke,
                messages
            )
            
            logger.info(f"Generated response for query: {query[:50]}...")
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return f"I'm sorry, I encountered an error while generating a response: {str(e)}"
    
    async def retrieve_augment_generate(
        self, 
        query: str,
        index_name: Optional[str] = None,
        k: int = 4,
        score_threshold: Optional[float] = 0.4,
        rerank_top_k: int = 3,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve, augment, and generate.
        
        Args:
            query: The user query
            index_name: Name of the index to search in
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            rerank_top_k: Number of documents to keep after reranking
            use_cache: Whether to use caching for this query
            
        Returns:
            Dictionary containing the query, retrieved documents, augmented context, and generated response
        """
        try:
            # Try to get cached result if caching is enabled
            if use_cache:
                from modernrag.caching import get_cached_result, cache_result
                
                # Create cache parameters
                cache_params = {
                    "index_name": index_name,
                    "k": k,
                    "score_threshold": score_threshold,
                    "rerank_top_k": rerank_top_k
                }
                
                # Try to get cached result
                cached_result = await get_cached_result(query, **cache_params)
                if cached_result is not None:
                    logger.info(f"Using cached result for query: {query[:50]}...")
                    return cached_result
            
            # Step 1: Retrieve relevant documents
            retrieved_docs = await similarity_search(
                query=query,
                index_name=index_name,
                k=k,
                score_threshold=score_threshold
            )
            
            if not retrieved_docs:
                result = {
                    "query": query,
                    "retrieved_docs": [],
                    "augmented_context": "",
                    "response": "I couldn't find any relevant information to answer your query."
                }
                
                # Cache the result if caching is enabled
                if use_cache:
                    from modernrag.caching import cache_result
                    await cache_result(query, result, **cache_params)
                    
                return result
            
            # Step 2: Rerank documents
            reranked_docs = await self.augmentation_manager.rerank_documents(
                query=query,
                documents=retrieved_docs,
                top_k=rerank_top_k
            )
            
            # Step 3: Augment documents
            augmented_context = await self.augmentation_manager.augment_documents(
                query=query,
                documents=reranked_docs
            )
            
            # Step 4: Generate response
            response = await self.generate_response(
                query=query,
                context=augmented_context
            )
            
            import time
            result = {
                "query": query,
                "retrieved_docs": retrieved_docs,
                "augmented_context": augmented_context,
                "response": response,
                "cached": False,
                "timestamp": time.time()
            }
            
            # Cache the result if caching is enabled
            if use_cache:
                from modernrag.caching import cache_result
                await cache_result(query, result, **cache_params)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            result = {
                "query": query,
                "error": str(e),
                "response": f"I'm sorry, I encountered an error while processing your query: {str(e)}"
            }
            
            return result

# Create singleton instances
augmentation_manager = AugmentationManager()
generation_manager = GenerationManager()

# Async API functions
async def rerank_documents(
    query: str, 
    documents: List[Tuple[Document, float]],
    top_k: int = 3
) -> List[Document]:
    """Rerank documents based on relevance to the query."""
    return await augmentation_manager.rerank_documents(query, documents, top_k)


async def augment_documents(
    query: str, 
    documents: List[Document]
) -> str:
    """Augment documents by extracting and synthesizing relevant information."""
    return await augmentation_manager.augment_documents(query, documents)


async def generate_response(
    query: str, 
    context: str
) -> str:
    """Generate a response to the query using the provided context."""
    return await generation_manager.generate_response(query, context)


async def retrieve_augment_generate(
    query: str,
    index_name: Optional[str] = None,
    k: int = 4,
    score_threshold: Optional[float] = 0.4,
    rerank_top_k: int = 3,
    use_cache: bool = True
) -> Dict[str, Any]:
    """Complete RAG pipeline: retrieve, augment, and generate."""
    return await generation_manager.retrieve_augment_generate(
        query, index_name, k, score_threshold, rerank_top_k, use_cache
    )
