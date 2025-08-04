"""
Legacy Adapter for backward compatibility
Maintains the same interface as original RAGRunner
"""

from typing import Dict, Any, Optional
from .services.rag_orchestrator import RAGOrchestrator


class RAGRunner:
    """
    Legacy adapter for RAGRunner
    Maintains backward compatibility while using new enterprise architecture
    """
    
    def __init__(self, data_dir: str = None, use_openai_embeddings: bool = True):
        """Initialize with same interface as original RAGRunner"""
        self._orchestrator = RAGOrchestrator(
            data_dir=data_dir,
            use_openai_embeddings=use_openai_embeddings
        )
    
    @property
    def is_initialized(self) -> bool:
        """Check if system is initialized"""
        return self._orchestrator.is_initialized
    
    def setup(self) -> None:
        """Setup the RAG system"""
        self._orchestrator.setup()
    
    def ask_question(self, question: str) -> str:
        """
        Ask a question and return answer string (legacy format)
        
        Args:
            question: User's question
            
        Returns:
            Answer string (maintains legacy format)
        """
        result = self._orchestrator.ask_question(question)
        
        # Return in legacy format (just the answer string)
        if isinstance(result, dict):
            return result.get('answer', 'Cevap bulunamadı')
        return str(result)
    
    def ask_question_detailed(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and return detailed response (new format)
        
        Args:
            question: User's question
            
        Returns:
            Detailed response with metadata
        """
        return self._orchestrator.ask_question(question)
    
    def health_check(self) -> Dict[str, Any]:
        """Get system health information"""
        return self._orchestrator.health_check()
    
    def reset(self) -> None:
        """Reset system state"""
        self._orchestrator.reset()
    
    # Legacy property access for backward compatibility
    @property 
    def document_processor(self):
        """Access to document processor (legacy)"""
        return self._orchestrator.document_processor
    
    @property
    def vector_store(self):
        """Access to vector store (legacy)"""
        return self._orchestrator.vector_store
