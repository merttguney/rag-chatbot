"""
Embedder module for text embedding operations
Supports OpenAI embeddings and SentenceTransformers
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger

# OpenAI for embeddings
from openai import OpenAI

# Alternative: SentenceTransformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available. Using OpenAI embeddings only.")

from ..core.config import config


class Embedder:
    """Text embedder using OpenAI or SentenceTransformers"""
    
    def __init__(self, 
                 embedding_model: str = None,
                 use_openai: bool = True):
        """
        Initialize embedder
        
        Args:
            embedding_model: Model name for embeddings
            use_openai: Whether to use OpenAI embeddings (vs SentenceTransformers)
        """
        self.use_openai = use_openai
        self.embedding_model = embedding_model or config.OPENAI_EMBEDDING_MODEL
        
        if self.use_openai:
            if not config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.embedding_dimension = config.EMBEDDING_DIMENSION
            logger.info(f"Initialized OpenAI embedder with model: {self.embedding_model}")
            
        else:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("SentenceTransformers is required for local embeddings")
            
            # Default local model
            local_model = embedding_model or "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(local_model)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized SentenceTransformer embedder with model: {local_model}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Create embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.embedding_dimension
        
        try:
            if self.use_openai:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                return response.data[0].embedding
            else:
                embedding = self.model.encode(text, convert_to_tensor=False)
                return embedding.tolist()
                
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """
        Create embeddings for multiple texts with optimized batching
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (uses config default if None)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        batch_size = batch_size or config.BATCH_SIZE
        embeddings = []
        
        try:
            if self.use_openai:
                # Process in batches to avoid rate limits
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_embeddings = self._embed_openai_batch(batch)
                    embeddings.extend(batch_embeddings)
                    logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            else:
                # SentenceTransformers can handle larger batches efficiently
                embeddings = self.model.encode(texts, convert_to_tensor=False, batch_size=batch_size)
                embeddings = embeddings.tolist()
                
            logger.info(f"Created embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {str(e)}")
            raise
    
    def _embed_openai_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a batch using OpenAI API
        
        Args:
            texts: Batch of texts to embed
            
        Returns:
            List of embedding vectors for the batch
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {str(e)}")
            # Fallback to individual processing
            return [self.embed_text(text) for text in texts]
        
        embeddings = []
        
        try:
            if self.use_openai:
                # Process in batches for OpenAI API
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=batch
                    )
                    
                    batch_embeddings = [data.embedding for data in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            else:
                # SentenceTransformers can handle batch processing efficiently
                embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
                embeddings = [emb.tolist() for emb in embeddings]
            
            logger.info(f"Created {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {str(e)}")
            raise
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create embeddings for text chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            
        Returns:
            List of chunks with added 'embedding' field
        """
        if not chunks:
            return []
        
        # Extract texts from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = chunk.copy()
            embedded_chunk['embedding'] = embedding
            embedded_chunks.append(embedded_chunk)
        
        logger.info(f"Added embeddings to {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.embedding_dimension
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def test_embedding(self) -> bool:
        """Test if embedding service is working"""
        try:
            test_text = "This is a test sentence for embedding."
            embedding = self.embed_text(test_text)
            
            if len(embedding) == self.embedding_dimension:
                logger.info("Embedding test successful")
                return True
            else:
                logger.error(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding)}")
                return False
                
        except Exception as e:
            logger.error(f"Embedding test failed: {str(e)}")
            return False
