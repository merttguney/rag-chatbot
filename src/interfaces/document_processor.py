"""
Document processing interface
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class IDocumentProcessor(ABC):
    """Interface for document processing operations"""
    
    @abstractmethod
    def load_documents(self, directory: str) -> List[Dict[str, Any]]:
        """Load documents from directory"""
        pass
    
    @abstractmethod
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks"""
        pass


class IEmbeddingService(ABC):
    """Interface for embedding operations"""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for single query"""
        pass


class IVectorStore(ABC):
    """Interface for vector storage operations"""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Add documents and embeddings to store"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search similar documents"""
        pass
    
    @abstractmethod
    def save_index(self, path: str) -> None:
        """Save index to disk"""
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> None:
        """Load index from disk"""
        pass


class IQueryProcessor(ABC):
    """Interface for query processing operations"""
    
    @abstractmethod
    def process_question(self, question: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process question with context and generate answer"""
        pass
