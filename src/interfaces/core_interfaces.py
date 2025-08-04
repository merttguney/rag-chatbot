"""
Core interfaces for RAG system components.
Following Interface Segregation Principle (SOLID-I)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple


class DocumentProcessorInterface(ABC):
    """Interface for document processing operations"""
    
    @abstractmethod
    def load_documents(self, directory: str) -> List[Dict[str, Any]]:
        """Load documents from directory"""
        pass
    
    @abstractmethod
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a single document"""
        pass


class EmbeddingInterface(ABC):
    """Interface for embedding operations"""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass


class VectorStoreInterface(ABC):
    """Interface for vector storage operations"""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Add documents with their embeddings"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def save_index(self, path: str) -> None:
        """Save vector index to disk"""
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> None:
        """Load vector index from disk"""
        pass


class TextProcessorInterface(ABC):
    """Interface for text processing operations"""
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        pass
    
    @abstractmethod
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split multiple documents into chunks"""
        pass


class RetrievalInterface(ABC):
    """Interface for document retrieval operations"""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for query"""
        pass


class RAGPipelineInterface(ABC):
    """Interface for end-to-end RAG pipeline"""
    
    @abstractmethod
    def process_question(self, question: str) -> Dict[str, Any]:
        """Process question and return answer with metadata"""
        pass
