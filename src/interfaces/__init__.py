"""
Interfaces package for SOLID design
Contains abstract base classes for all major components
"""

from .core_interfaces import (
    DocumentProcessorInterface,
    EmbeddingInterface,
    VectorStoreInterface,
    TextProcessorInterface,
    RetrievalInterface,
    RAGPipelineInterface
)

__all__ = [
    'DocumentProcessorInterface',
    'EmbeddingInterface', 
    'VectorStoreInterface',
    'TextProcessorInterface',
    'RetrievalInterface',
    'RAGPipelineInterface'
]
