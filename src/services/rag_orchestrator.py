"""
RAG System Orchestrator - Enterprise Service Layer
Replaces the monolithic RAGRunner with clean separation of concerns
"""

import os
import time
from typing import List, Dict, Any, Optional
from loguru import logger

from ..interfaces.core_interfaces import (
    DocumentProcessorInterface,
    EmbeddingInterface,
    VectorStoreInterface, 
    TextProcessorInterface,
    RetrievalInterface,
    RAGPipelineInterface
)
from ..core.factories import RAGSystemFactory
from ..core.config import config


class RAGOrchestrator:
    """
    Enterprise RAG System Orchestrator
    
    Responsibilities:
    - System initialization and setup
    - Component lifecycle management
    - High-level business operations
    - Error handling and recovery
    """
    
    def __init__(self, data_dir: str = None, use_openai_embeddings: bool = True):
        """Initialize RAG Orchestrator with minimal dependencies"""
        self.data_dir = data_dir or config.DATA_DIR
        self.use_openai_embeddings = use_openai_embeddings
        self.is_initialized = False
        
        # Components will be injected via factory
        self._components: Optional[Dict[str, Any]] = None
    
    @property
    def components(self) -> Dict[str, Any]:
        """Get system components (lazy initialization)"""
        if self._components is None:
            self._components = RAGSystemFactory.create_system(
                use_openai_embeddings=self.use_openai_embeddings
            )
        return self._components
    
    @property
    def document_processor(self) -> DocumentProcessorInterface:
        return self.components['document_processor']
    
    @property
    def embedding_service(self) -> EmbeddingInterface:
        return self.components['embedding_service']
    
    @property
    def vector_store(self) -> VectorStoreInterface:
        return self.components['vector_store']
    
    @property
    def text_processor(self) -> TextProcessorInterface:
        return self.components['text_processor']
    
    @property
    def retriever(self) -> RetrievalInterface:
        return self.components['retriever']
    
    @property
    def pipeline(self) -> RAGPipelineInterface:
        return self.components['pipeline']
    
    def setup(self) -> None:
        """
        Initialize the RAG system with proper error handling
        
        Raises:
            FileNotFoundError: If data directory doesn't exist
            Exception: For other setup errors
        """
        try:
            start_time = time.time()
            logger.info("Starting RAG system setup...")
            
            # Validate data directory
            if not os.path.exists(self.data_dir):
                raise FileNotFoundError(f"Directory not found: {self.data_dir}")
            
            # Load and process documents
            logger.info(f"Loading documents from: {self.data_dir}")
            documents = self.document_processor.load_documents(self.data_dir)
            logger.info(f"Loaded {len(documents)} documents")
            
            if not documents:
                logger.warning("No documents found in data directory")
                self.is_initialized = True
                return
            
            # Split documents into chunks
            logger.info("Splitting documents into chunks...")
            chunks = self.text_processor.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            
            if not chunks:
                logger.warning("No text chunks created from documents")
                self.is_initialized = True
                return
            
            # Create embeddings
            logger.info("Creating embeddings...")
            texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(texts)
            
            # Store in vector database
            logger.info("Storing embeddings in vector database...")
            self.vector_store.add_documents(chunks, embeddings)
            
            # Save to disk
            embeddings_path = os.path.join(config.EMBEDDINGS_DIR, "faiss_index_simple.npy")
            self.vector_store.save_index(embeddings_path)
            
            setup_time = time.time() - start_time
            logger.info(f"RAG system setup completed successfully in {setup_time:.2f}s")
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Error during RAG setup: {str(e)}")
            raise
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Process a question through the RAG pipeline
        
        Args:
            question: User's question
            
        Returns:
            Dict containing answer and metadata
            
        Raises:
            ValueError: If system not initialized or empty question
            Exception: For processing errors
        """
        if not self.is_initialized:
            raise ValueError("RAG system not initialized. Call setup() first.")
        
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            return self.pipeline.process_question(question.strip())
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform system health check
        
        Returns:
            Dict containing system health information
        """
        return {
            'initialized': self.is_initialized,
            'data_directory': self.data_dir,
            'data_directory_exists': os.path.exists(self.data_dir),
            'embeddings_directory': config.EMBEDDINGS_DIR,
            'embeddings_directory_exists': os.path.exists(config.EMBEDDINGS_DIR),
            'components_loaded': self._components is not None,
            'openai_embeddings': self.use_openai_embeddings
        }
    
    def reset(self) -> None:
        """Reset system state for testing or reinitialization"""
        self.is_initialized = False
        self._components = None
        logger.info("RAG system reset completed")
