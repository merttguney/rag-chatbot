"""
Simple Vector Store implementation using numpy and sklearn
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger


class SimpleVectorStore:
    """
    Simple vector store using numpy arrays and cosine similarity
    """
    
    def __init__(self, dimension: int = 1536):
        """
        Initialize SimpleVectorStore
        
        Args:
            dimension: Vector dimension (default 1536 for OpenAI embeddings)
        """
        self.dimension = dimension
        self.vectors = None
        self.documents = []
        self.metadata = []
        self.is_loaded = False
        logger.info(f"SimpleVectorStore initialized with dimension {dimension}")
    
    @property
    def is_initialized(self) -> bool:
        """Check if vector store is initialized with data"""
        return self.vectors is not None and len(self.documents) > 0
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]], 
                     metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add documents to the vector store
        
        Args:
            texts: List of document texts
            embeddings: List of embedding vectors
            metadata: Optional metadata for each document
        """
        if not texts or not embeddings:
            return
        
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")
        
        # Convert embeddings to numpy array
        new_vectors = np.array(embeddings, dtype=np.float32)
        
        if self.vectors is None:
            self.vectors = new_vectors
            self.documents = texts.copy()
            self.metadata = metadata.copy() if metadata else [{}] * len(texts)
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
            self.documents.extend(texts)
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{}] * len(texts))
        
        logger.info(f"Added {len(texts)} documents to vector store. Total: {len(self.documents)}")
    
    def add_embeddings(self, embeddings: List[List[float]], 
                      metadata: Optional[List[Dict[str, Any]]] = None,
                      texts: Optional[List[str]] = None,
                      metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add embeddings with metadata to the vector store
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries (for compatibility)
            texts: List of texts (optional, extracted from metadata if not provided)
            metadatas: List of metadata dictionaries (alternative naming)
            
        Returns:
            List of document IDs (indices as strings)
        """
        if not embeddings:
            return []
        
        # Handle different parameter naming conventions
        final_metadata = metadata or metadatas or []
        
        if texts:
            final_texts = texts
        else:
            # Extract texts from metadata
            final_texts = [meta.get('text', '') for meta in final_metadata] if final_metadata else [''] * len(embeddings)
        
        if len(embeddings) != len(final_texts):
            # If lengths don't match, pad or truncate
            if len(final_texts) < len(embeddings):
                final_texts.extend([''] * (len(embeddings) - len(final_texts)))
            else:
                final_texts = final_texts[:len(embeddings)]
        
        if len(embeddings) != len(final_metadata) and final_metadata:
            # If lengths don't match, pad or truncate metadata
            if len(final_metadata) < len(embeddings):
                final_metadata.extend([{}] * (len(embeddings) - len(final_metadata)))
            else:
                final_metadata = final_metadata[:len(embeddings)]
        elif not final_metadata:
            final_metadata = [{}] * len(embeddings)
        
        # Use existing add_documents method
        start_idx = len(self.documents)
        self.add_documents(final_texts, embeddings, final_metadata)
        
        # Return document IDs (indices as strings)
        return [str(i) for i in range(start_idx, len(self.documents))]
    
    def search(self, query_embedding: List[float], top_k: int = 5, 
              threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results with text, metadata, and similarity scores
        """
        if self.vectors is None or len(self.vectors) == 0:
            logger.warning("No vectors in store for search")
            return []
        
        # Convert query to numpy array
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            if similarity >= threshold:
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'similarity': similarity,
                    'index': int(idx)
                })
        
        logger.info(f"Found {len(results)} results for search query")
        return results
    
    def save(self) -> None:
        """
        Save the vector store using default paths
        """
        # Create default paths
        embeddings_dir = "embeddings"
        os.makedirs(embeddings_dir, exist_ok=True)
        
        vectors_path = os.path.join(embeddings_dir, "faiss_index_simple.npy")
        docstore_path = os.path.join(embeddings_dir, "docstore.pkl")
        
        self.save_index(vectors_path, docstore_path)
    
    def load(self) -> bool:
        """
        Load the vector store using default paths
        
        Returns:
            True if loaded successfully, False otherwise
        """
        # Create default paths
        embeddings_dir = "embeddings"
        vectors_path = os.path.join(embeddings_dir, "faiss_index_simple.npy")
        docstore_path = os.path.join(embeddings_dir, "docstore.pkl")
        
        return self.load_index(vectors_path, docstore_path)
    
    def save_index(self, vectors_path: str, docstore_path: str) -> None:
        """
        Save vectors and documents to files
        
        Args:
            vectors_path: Path to save vectors (.npy)
            docstore_path: Path to save document store (.pkl)
        """
        if self.vectors is not None:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(vectors_path), exist_ok=True)
            os.makedirs(os.path.dirname(docstore_path), exist_ok=True)
            
            # Save vectors
            np.save(vectors_path, self.vectors)
            logger.info(f"Saved vectors to {vectors_path}")
            
            # Save document store
            docstore = {
                'documents': self.documents,
                'metadata': self.metadata,
                'dimension': self.dimension
            }
            with open(docstore_path, 'wb') as f:
                pickle.dump(docstore, f)
            logger.info(f"Saved document store to {docstore_path}")
    
    def load_index(self, vectors_path: str, docstore_path: str) -> bool:
        """
        Load vectors and documents from files
        
        Args:
            vectors_path: Path to vectors file (.npy)
            docstore_path: Path to document store file (.pkl)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if os.path.exists(vectors_path) and os.path.exists(docstore_path):
                # Load vectors
                self.vectors = np.load(vectors_path)
                logger.info(f"Loaded vectors from {vectors_path}")
                
                # Load document store
                with open(docstore_path, 'rb') as f:
                    docstore = pickle.load(f)
                
                self.documents = docstore['documents']
                self.metadata = docstore['metadata']
                self.dimension = docstore.get('dimension', self.dimension)
                self.is_loaded = True
                
                logger.info(f"Loaded document store from {docstore_path}")
                return True
            else:
                logger.warning("Vector or docstore files not found")
                return False
                
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics
        
        Returns:
            Dictionary with store statistics
        """
        return {
            'total_documents': len(self.documents),
            'total_vectors': len(self.vectors) if self.vectors is not None else 0,
            'dimension': self.dimension,
            'is_loaded': self.is_loaded,
            'memory_usage_mb': self.vectors.nbytes / (1024 * 1024) if self.vectors is not None else 0
        }
    
    def clear(self) -> None:
        """Clear all data from the vector store"""
        self.vectors = None
        self.documents = []
        self.metadata = []
        self.is_loaded = False
        logger.info("Vector store cleared")
