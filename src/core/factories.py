"""
Factory classes for creating RAG system components.
Implements Factory Pattern for loose coupling and testability.
"""

from typing import Optional, Dict, Any
from ..interfaces.core_interfaces import (
    DocumentProcessorInterface,
    EmbeddingInterface, 
    VectorStoreInterface,
    TextProcessorInterface,
    RetrievalInterface,
    RAGPipelineInterface
)
from ..services.document_service import FileLoader
from ..services.embedding_service import Embedder
from ..storage.vector_store_impl import SimpleVectorStore
from ..utils.text_processing import TextSplitter
from ..retriever import Retriever
from ..core.pipeline import RAGPipeline
from ..core.config import config


class ComponentFactory:
    """Factory for creating RAG system components with proper dependency injection"""
    
    @staticmethod
    def create_document_processor() -> DocumentProcessorInterface:
        """Create document processor instance"""
        return FileLoader()
    
    @staticmethod
    def create_embedding_service(use_openai: bool = True) -> EmbeddingInterface:
        """Create embedding service instance"""
        return Embedder(
            embedding_model=config.EMBEDDING_MODEL,
            use_openai=use_openai
        )
    
    @staticmethod
    def create_vector_store() -> VectorStoreInterface:
        """Create vector store instance"""
        return SimpleVectorStore(dimension=config.EMBEDDING_DIMENSION)
    
    @staticmethod
    def create_text_processor() -> TextProcessorInterface:
        """Create text processor instance"""
        return TextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
    
    @staticmethod
    def create_retriever(
        vector_store: VectorStoreInterface,
        embedder: EmbeddingInterface,
        top_k: int = 5
    ) -> RetrievalInterface:
        """Create retriever instance with dependencies"""
        return Retriever(
            vector_store=vector_store,
            embedder=embedder,
            top_k=top_k
        )
    
    @staticmethod
    def create_pipeline(
        retriever: RetrievalInterface,
        model_name: str = None
    ) -> RAGPipelineInterface:
        """Create RAG pipeline instance with dependencies"""
        return RAGPipeline(
            retriever=retriever,
            model_name=model_name or config.OPENAI_MODEL
        )


class RAGSystemFactory:
    """High-level factory for creating complete RAG system"""
    
    @staticmethod
    def create_system(
        use_openai_embeddings: bool = True,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Create complete RAG system with all components
        
        Returns:
            Dict containing all initialized components
        """
        # Create individual components
        document_processor = ComponentFactory.create_document_processor()
        embedding_service = ComponentFactory.create_embedding_service(use_openai_embeddings)
        vector_store = ComponentFactory.create_vector_store()
        text_processor = ComponentFactory.create_text_processor()
        
        # Create retriever with dependencies
        retriever = ComponentFactory.create_retriever(
            vector_store=vector_store,
            embedder=embedding_service,
            top_k=top_k
        )
        
        # Create pipeline with dependencies
        pipeline = ComponentFactory.create_pipeline(retriever=retriever)
        
        return {
            'document_processor': document_processor,
            'embedding_service': embedding_service,
            'vector_store': vector_store,
            'text_processor': text_processor,
            'retriever': retriever,
            'pipeline': pipeline
        }
