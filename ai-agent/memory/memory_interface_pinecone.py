"""
Pinecone interface for long-term memory storage and retrieval.
"""

import os
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class PineconeMemoryInterface:
    """Interface for storing and retrieving memories using Pinecone vector database."""
    
    def __init__(
        self,
        index_name: str = "agent-memory-index",
        embedding_dimension: int = 1536,
        environment: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize Pinecone memory interface.
        
        Args:
            index_name: Name of the Pinecone index
            embedding_dimension: Dimension of embeddings (1536 for OpenAI ada-002)
            environment: Pinecone environment (if None, reads from PINECONE_ENVIRONMENT)
            api_key: Pinecone API key (if None, reads from PINECONE_API_KEY)
        """
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension
        
        # Get API credentials
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY environment variable.")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize or connect to index
        self._setup_index()
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="content"
        )
        
        logger.info(f"Initialized Pinecone memory interface with index: {index_name}")
    
    def _setup_index(self):
        """Set up Pinecone index, creating it if it doesn't exist."""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                
                # Create index with serverless spec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Created Pinecone index: {self.index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {e}")
            raise
    
    def store_memory(
        self,
        content: str,
        memory_type: str = "interaction",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a memory in Pinecone.
        
        Args:
            content: The content to store
            memory_type: Type of memory (e.g., 'interaction', 'fact', 'preference')
            metadata: Additional metadata to store with the memory
            
        Returns:
            The ID of the stored memory
        """
        try:
            # Generate unique ID
            memory_id = str(uuid.uuid4())
            
            # Prepare metadata
            full_metadata = {
                "memory_id": memory_id,
                "memory_type": memory_type,
                "timestamp": datetime.utcnow().isoformat(),
                "content_length": len(content)
            }
            
            if metadata:
                full_metadata.update(metadata)
            
            # Create document
            document = Document(
                page_content=content,
                metadata=full_metadata
            )
            
            # Store in vector store
            ids = self.vector_store.add_documents([document], ids=[memory_id])
            
            logger.info(f"Stored memory with ID: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
    
    def retrieve_memories(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        relevance_threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Retrieve relevant memories based on a query.
        
        Args:
            query: Query to search for relevant memories
            top_k: Number of top memories to retrieve
            memory_type: Filter by memory type (optional)
            relevance_threshold: Minimum relevance score (0-1)
            
        Returns:
            List of tuples (content, score, metadata)
        """
        try:
            # Prepare filter
            filter_dict = {}
            if memory_type:
                filter_dict["memory_type"] = memory_type
            
            # Search for similar documents
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=filter_dict if filter_dict else None
            )
            
            # Filter by relevance threshold and format results
            relevant_memories = []
            for document, score in results:
                # Convert distance to similarity (Pinecone uses cosine distance)
                similarity = 1 - score
                
                if similarity >= relevance_threshold:
                    relevant_memories.append((
                        document.page_content,
                        similarity,
                        document.metadata
                    ))
            
            logger.info(f"Retrieved {len(relevant_memories)} relevant memories for query")
            return relevant_memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        new_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of the memory to update
            new_content: New content for the memory
            new_metadata: New metadata (will be merged with existing)
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Delete old memory
            self.delete_memory(memory_id)
            
            # Store updated memory with same ID
            metadata = new_metadata or {}
            metadata["updated_at"] = datetime.utcnow().isoformat()
            
            document = Document(
                page_content=new_content,
                metadata=metadata
            )
            
            self.vector_store.add_documents([document], ids=[memory_id])
            
            logger.info(f"Updated memory with ID: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
            return False
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            self.index.delete(ids=[memory_id])
            logger.info(f"Deleted memory with ID: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory index.
        
        Returns:
            Dictionary containing index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}
    
    def search_memories_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        top_k: int = 10
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Search memories by metadata filters.
        
        Args:
            metadata_filter: Dictionary of metadata filters
            top_k: Maximum number of memories to return
            
        Returns:
            List of tuples (content, metadata)
        """
        try:
            # Use a broad query and filter by metadata
            results = self.vector_store.similarity_search(
                query="",  # Empty query to get all results
                k=top_k,
                filter=metadata_filter
            )
            
            return [(doc.page_content, doc.metadata) for doc in results]
            
        except Exception as e:
            logger.error(f"Error searching memories by metadata: {e}")
            return []