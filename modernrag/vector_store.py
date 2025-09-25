"""
Vector Store Module for Modern RAG Application

This module provides async interfaces for interacting with Pinecone vector database.
It handles vector storage operations including index management, document embedding,
and vector search operations.
"""

import os
import getpass
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from uuid import uuid4
from functools import lru_cache

# Third-party imports
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field
from pydantic_settings import BaseSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class VectorStoreConfig(BaseSettings):
    """Configuration settings for vector store operations."""
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    default_index_name: str = Field("langchain-test-index", env="PINECONE_INDEX_NAME")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    dimension: int = Field(1536, env="VECTOR_DIMENSION")
    metric: str = Field("cosine", env="SIMILARITY_METRIC")
    cloud_provider: str = Field("aws", env="CLOUD_PROVIDER")
    region: str = Field("us-east-1", env="CLOUD_REGION")
    chunk_size: int = Field(200, env="CHUNK_SIZE")
    chunk_overlap: int = Field(20, env="CHUNK_OVERLAP")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_config() -> VectorStoreConfig:
    """Get the application configuration, prompting for API key if needed."""
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
    
    return VectorStoreConfig()


@lru_cache()
def get_embeddings() -> OpenAIEmbeddings:
    """Get the embedding model instance."""
    config = get_config()
    return OpenAIEmbeddings(model=config.embedding_model)


class VectorStoreManager:
    """Manages vector store operations with async support."""
    
    def __init__(self):
        """Initialize the vector store manager."""
        self.config = get_config()
        self._pinecone_client = Pinecone(api_key=self.config.pinecone_api_key)
        self._embeddings = get_embeddings()
        self._index_cache = {}
        self._vector_store_cache = {}
    
    async def create_index(self, index_name: Optional[str] = None) -> str:
        """Create a new Pinecone index asynchronously.
        
        Args:
            index_name: Name of the index to create. Uses default if not provided.
            
        Returns:
            The name of the created index.
        
        Raises:
            Exception: If index creation fails.
        """
        index_name = index_name or self.config.default_index_name
        
        try:
            # Run the synchronous Pinecone operation in a thread pool
            await asyncio.to_thread(
                self._pinecone_client.create_index,
                name=index_name,
                dimension=self.config.dimension,
                metric=self.config.metric,
                spec=ServerlessSpec(
                    cloud=self.config.cloud_provider, 
                    region=self.config.region
                )
            )
            logger.info(f"Created index: {index_name}")
            return index_name
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {str(e)}")
            raise
    
    async def delete_index(self, index_name: Optional[str] = None) -> bool:
        """Delete a Pinecone index asynchronously.
        
        Args:
            index_name: Name of the index to delete. Uses default if not provided.
            
        Returns:
            True if deletion was successful.
            
        Raises:
            Exception: If index deletion fails.
        """
        index_name = index_name or self.config.default_index_name
        
        try:
            # Run the synchronous Pinecone operation in a thread pool
            await asyncio.to_thread(
                self._pinecone_client.delete_index,
                index_name
            )
            
            # Clear caches for this index
            if index_name in self._index_cache:
                del self._index_cache[index_name]
            if index_name in self._vector_store_cache:
                del self._vector_store_cache[index_name]
                
            logger.info(f"Deleted index: {index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete index {index_name}: {str(e)}")
            raise
    
    async def check_index_exists(self, index_name: Optional[str] = None) -> bool:
        """Check if an index exists, create it if it doesn't.
        
        Args:
            index_name: Name of the index to check. Uses default if not provided.
            
        Returns:
            True if the index exists or was created successfully.
        """
        index_name = index_name or self.config.default_index_name
        
        try:
            # Run the synchronous Pinecone operation in a thread pool
            has_index = await asyncio.to_thread(
                self._pinecone_client.has_index,
                index_name
            )
            
            if not has_index:
                logger.info(f"Index {index_name} does not exist, creating it...")
                await self.create_index(index_name)
                
            return True
        except Exception as e:
            logger.error(f"Failed to check/create index {index_name}: {str(e)}")
            raise
    
    async def get_index(self, index_name: Optional[str] = None):
        """Get a Pinecone index instance asynchronously.
        
        Args:
            index_name: Name of the index to get. Uses default if not provided.
            
        Returns:
            Pinecone index instance.
        """
        index_name = index_name or self.config.default_index_name
        
        # Check if index exists in cache
        if index_name not in self._index_cache:
            # Ensure the index exists
            await self.check_index_exists(index_name)
            
            # Get the index
            self._index_cache[index_name] = self._pinecone_client.Index(index_name)
            
        return self._index_cache[index_name]
    
    async def get_vector_store(self, index_name: Optional[str] = None) -> PineconeVectorStore:
        """Get a PineconeVectorStore instance asynchronously.
        
        Args:
            index_name: Name of the index to use. Uses default if not provided.
            
        Returns:
            PineconeVectorStore instance.
        """
        index_name = index_name or self.config.default_index_name
        
        # Check if vector store exists in cache
        if index_name not in self._vector_store_cache:
            # Get the index
            index = await self.get_index(index_name)
            
            # Create the vector store
            self._vector_store_cache[index_name] = PineconeVectorStore(
                index=index, 
                embedding=self._embeddings
            )
            
        return self._vector_store_cache[index_name]
    
    async def upsert_documents(
        self, 
        documents: List[Document], 
        index_name: Optional[str] = None,
        ids: Optional[List[str]] = None
    ) -> bool:
        """Upsert documents to the vector store asynchronously.
        
        Args:
            documents: List of documents to upsert.
            index_name: Name of the index to use. Uses default if not provided.
            ids: Optional list of IDs for the documents.
            
        Returns:
            True if upsert was successful.
            
        Raises:
            Exception: If document upsert fails.
        """
        try:
            # Generate UUIDs if not provided
            if ids is None:
                ids = [str(uuid4()) for _ in range(len(documents))]
                
            # Get the vector store
            vector_store = await self.get_vector_store(index_name)
            
            # Run the synchronous add_documents operation in a thread pool
            await asyncio.to_thread(
                vector_store.add_documents,
                documents=documents,
                ids=ids
            )
            
            logger.info(f"Upserted {len(documents)} documents to index {index_name or self.config.default_index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert documents: {str(e)}")
            raise
    
    async def split_and_upsert_documents(
        self,
        documents: List[Document],
        index_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        batch_size: int = 100
    ) -> bool:
        """Split documents into chunks and upsert them to the vector store.
        
        Args:
            documents: List of documents to split and upsert.
            index_name: Name of the index to use. Uses default if not provided.
            chunk_size: Size of each chunk. Uses config default if not provided.
            chunk_overlap: Overlap between chunks. Uses config default if not provided.
            batch_size: Number of documents to process in each batch.
            
        Returns:
            True if the operation was successful.
        """
        chunk_size = chunk_size or self.config.chunk_size
        chunk_overlap = chunk_overlap or self.config.chunk_overlap
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Split documents
        texts = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(texts)} chunks")
        
        # Process in batches
        if len(texts) <= batch_size:
            # Small enough to process in one batch
            return await self.upsert_documents(texts, index_name)
        else:
            # Process in batches
            success = True
            total_batches = (len(texts) + batch_size - 1) // batch_size  # Ceiling division
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} chunks")
                
                try:
                    batch_success = await self.upsert_documents(batch, index_name)
                    if not batch_success:
                        logger.error(f"Failed to upsert batch {batch_num}/{total_batches}")
                        success = False
                except Exception as e:
                    logger.error(f"Error upserting batch {batch_num}/{total_batches}: {str(e)}")
                    success = False
            
            return success
    
    async def similarity_search(
        self,
        query: str,
        index_name: Optional[str] = None,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search in the vector store.
        
        Args:
            query: The query string to search for.
            index_name: Name of the index to search in. Uses default if not provided.
            k: Number of results to return.
            score_threshold: Minimum similarity score threshold.
            
        Returns:
            List of (document, score) tuples.
        """
        try:
            # Get the vector store
            vector_store = await self.get_vector_store(index_name)
            
            # Configure search parameters
            search_kwargs = {"k": k}
            if score_threshold is not None:
                search_kwargs["score_threshold"] = score_threshold
            
            # Use similarity_search_with_score to get documents with scores
            if score_threshold is not None:
                results = await asyncio.to_thread(
                    vector_store.similarity_search_with_score,
                    query,
                    k=k,
                    score_threshold=score_threshold
                )
            else:
                results = await asyncio.to_thread(
                    vector_store.similarity_search_with_score,
                    query,
                    k=k
                )
            
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {str(e)}")
            raise


# Create a singleton instance
vector_store_manager = VectorStoreManager()


# Async API functions
async def create_index(index_name: Optional[str] = None) -> str:
    """Create a new Pinecone index."""
    return await vector_store_manager.create_index(index_name)


async def delete_index(index_name: Optional[str] = None) -> bool:
    """Delete a Pinecone index."""
    return await vector_store_manager.delete_index(index_name)


async def check_index_exists(index_name: Optional[str] = None) -> bool:
    """Check if an index exists, create it if it doesn't."""
    return await vector_store_manager.check_index_exists(index_name)


async def get_index(index_name: Optional[str] = None):
    """Get a Pinecone index instance."""
    return await vector_store_manager.get_index(index_name)


async def get_vector_store(index_name: Optional[str] = None) -> PineconeVectorStore:
    """Get a PineconeVectorStore instance."""
    return await vector_store_manager.get_vector_store(index_name)


async def upsert_documents(
    documents: List[Document], 
    index_name: Optional[str] = None,
    ids: Optional[List[str]] = None
) -> bool:
    """Upsert documents to the vector store."""
    return await vector_store_manager.upsert_documents(documents, index_name, ids)


async def split_and_upsert_documents(
    documents: List[Document],
    index_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    batch_size: int = 100
) -> bool:
    """Split documents into chunks and upsert them to the vector store."""
    return await vector_store_manager.split_and_upsert_documents(
        documents, index_name, chunk_size, chunk_overlap, batch_size
    )


async def similarity_search(
    query: str,
    index_name: Optional[str] = None,
    k: int = 4,
    score_threshold: Optional[float] = None
) -> List[Tuple[Document, float]]:
    """Perform a similarity search in the vector store."""
    return await vector_store_manager.similarity_search(
        query, index_name, k, score_threshold
    )
