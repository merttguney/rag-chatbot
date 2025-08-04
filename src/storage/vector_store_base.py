"""
Vector store module for FAISS integration
Handles storage and retrieval of document embeddings
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

# FAISS for vector similarity search
import faiss

from ..core.config import config


class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self, 
                 index_path: str = None,
                 docstore_path: str = None,
                 embedding_dimension: int = None):
        """
        Initialize vector store
        
        Args:
            index_path: Path to FAISS index file
            docstore_path: Path to document store pickle file  
            embedding_dimension: Dimension of embeddings
        """
        self.index_path = index_path or config.FAISS_INDEX_PATH
        self.docstore_path = docstore_path or config.DOCSTORE_PATH
        self.embedding_dimension = embedding_dimension or config.EMBEDDING_DIMENSION
        
        # Initialize FAISS index and document store
        self.index = None
        self.docstore = {}  # Maps index ID to document metadata
        self.next_id = 0
        
        # Load existing index if available
        self.load_index()
        
        logger.info(f"VectorStore initialized with dimension {self.embedding_dimension}")
    
    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index"""
        # Use IndexFlatIP for cosine similarity (Inner Product)
        # IndexFlatL2 for L2 distance, IndexHNSWFlat for approximate search
        index = faiss.IndexFlatIP(self.embedding_dimension)
        
        # Normalize vectors for cosine similarity
        index = faiss.IndexIDMap2(index)
        
        logger.info(f"Created new FAISS index with dimension {self.embedding_dimension}")
        return index
    
    def add_embeddings(self, 
                      embeddings: List[List[float]], 
                      metadata_list: List[Dict[str, Any]]) -> List[int]:
        """
        Add embeddings to the vector store
        
        Args:
            embeddings: List of embedding vectors
            metadata_list: List of metadata dictionaries for each embedding
            
        Returns:
            List of assigned IDs
        """
        if not embeddings or not metadata_list:
            logger.warning("Empty embeddings or metadata provided")
            return []
        
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Create index if it doesn't exist
        if self.index is None:
            self.index = self._create_index()
        
        # Convert embeddings to numpy array and normalize for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Generate IDs
        ids = list(range(self.next_id, self.next_id + len(embeddings)))
        ids_array = np.array(ids, dtype=np.int64)
        
        # Add to FAISS index
        self.index.add_with_ids(embeddings_array, ids_array)
        
        # Add metadata to docstore
        for i, metadata in enumerate(metadata_list):
            self.docstore[ids[i]] = metadata
        
        # Update next ID
        self.next_id += len(embeddings)
        
        logger.info(f"Added {len(embeddings)} embeddings to vector store")
        return ids
    
    def add_chunks(self, embedded_chunks: List[Dict[str, Any]]) -> List[int]:
        """
        Add embedded chunks to the vector store
        
        Args:
            embedded_chunks: List of chunks with 'embedding' and 'metadata' fields
            
        Returns:
            List of assigned IDs
        """
        embeddings = []
        metadata_list = []
        
        for chunk in embedded_chunks:
            if 'embedding' not in chunk:
                logger.warning("Chunk missing embedding, skipping")
                continue
            
            embeddings.append(chunk['embedding'])
            
            # Combine text and metadata
            metadata = chunk.get('metadata', {}).copy()
            metadata['text'] = chunk.get('text', '')
            metadata_list.append(metadata)
        
        return self.add_embeddings(embeddings, metadata_list)
    
    def search(self, 
               query_embedding: List[float], 
               k: int = None,
               score_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with metadata and scores
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("No embeddings in vector store")
            return []
        
        k = k or config.TOP_K_RESULTS
        score_threshold = score_threshold or config.SIMILARITY_THRESHOLD
        
        # Convert query to numpy array and normalize
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search
        scores, indices = self.index.search(query_array, k)
        
        # Process results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # No result found
                continue
            
            if score < score_threshold:
                continue
            
            # Get metadata
            metadata = self.docstore.get(int(idx), {})
            
            result = {
                'id': int(idx),
                'score': float(score),
                'text': metadata.get('text', ''),
                'metadata': {k: v for k, v in metadata.items() if k != 'text'}
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} results for search query")
        return results
    
    def save_index(self):
        """Save FAISS index and document store to disk"""
        try:
            # Ensure directories exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.docstore_path), exist_ok=True)
            
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
                logger.info(f"Saved FAISS index to {self.index_path}")
            
            # Save document store
            with open(self.docstore_path, 'wb') as f:
                pickle.dump({
                    'docstore': self.docstore,
                    'next_id': self.next_id,
                    'embedding_dimension': self.embedding_dimension
                }, f)
            logger.info(f"Saved document store to {self.docstore_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load_index(self):
        """Load FAISS index and document store from disk"""
        try:
            # Load FAISS index
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded FAISS index from {self.index_path}")
            
            # Load document store
            if os.path.exists(self.docstore_path):
                with open(self.docstore_path, 'rb') as f:
                    data = pickle.load(f)
                    self.docstore = data.get('docstore', {})
                    self.next_id = data.get('next_id', 0)
                    stored_dim = data.get('embedding_dimension', self.embedding_dimension)
                    
                    if stored_dim != self.embedding_dimension:
                        logger.warning(f"Dimension mismatch: stored {stored_dim}, expected {self.embedding_dimension}")
                
                logger.info(f"Loaded document store from {self.docstore_path}")
            
        except Exception as e:
            logger.warning(f"Could not load existing index: {str(e)}")
            logger.info("Starting with empty vector store")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'total_embeddings': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.embedding_dimension,
            'next_id': self.next_id,
            'docstore_entries': len(self.docstore),
            'index_path': self.index_path,
            'docstore_path': self.docstore_path
        }
    
    def is_initialized(self) -> bool:
        """Check if vector store is initialized with data"""
        return self.index is not None and self.index.ntotal > 0
    
    def clear(self):
        """Clear all data from vector store"""
        self.index = None
        self.docstore = {}
        self.next_id = 0
        
        # Remove files if they exist
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.docstore_path):
            os.remove(self.docstore_path)
        
        logger.info("Cleared vector store")
    
    def delete_by_source(self, source_file: str):
        """Delete all embeddings from a specific source file"""
        # This is complex with FAISS - would require rebuilding index
        # For now, we'll mark them in metadata and filter in search
        deleted_count = 0
        for doc_id, metadata in self.docstore.items():
            if metadata.get('source_file') == source_file:
                metadata['deleted'] = True
                deleted_count += 1
        
        logger.info(f"Marked {deleted_count} embeddings as deleted from {source_file}")
        return deleted_count
    
    def save(self):
        """Alias for save_index() for compatibility"""
        return self.save_index()
    
    def load(self):
        """Alias for load_index() for compatibility"""
        return self.load_index()
