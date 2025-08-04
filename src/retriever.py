"""
Retriever module for semantic search and document retrieval
Handles top-k similarity search and result filtering
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from .storage.vector_store_base import VectorStore
from .services.embedding_service import Embedder
from .core.config import config


class Retriever:
    """Document retriever using semantic search"""
    
    def __init__(self, 
                 vector_store: VectorStore = None,
                 embedder: Embedder = None,
                 top_k: int = None,
                 similarity_threshold: float = None):
        """
        Initialize retriever
        
        Args:
            vector_store: Vector store instance
            embedder: Embedder instance
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or Embedder()
        self.top_k = top_k or config.TOP_K_RESULTS
        self.similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
        
        logger.info(f"Retriever initialized with top_k={self.top_k}, "
                   f"threshold={self.similarity_threshold}")
    
    def retrieve(self, 
                query: str, 
                top_k: int = None,
                similarity_threshold: float = None,
                filter_deleted: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            filter_deleted: Whether to filter out deleted documents
            
        Returns:
            List of relevant document chunks with metadata and scores
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embedder.embed_text(query)
            
            # Search vector store
            k = top_k or self.top_k
            threshold = similarity_threshold or self.similarity_threshold
            
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=k * 2,  # Get more results to allow for filtering
                threshold=threshold
            )
            
            # Filter results
            filtered_results = []
            for result in results:
                # Skip deleted documents if requested
                if filter_deleted and result.get('metadata', {}).get('deleted', False):
                    continue
                
                filtered_results.append(result)
                
                # Stop when we have enough results
                if len(filtered_results) >= k:
                    break
            
            logger.info(f"Retrieved {len(filtered_results)} relevant documents for query")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise
    
    def retrieve_by_source(self, 
                          query: str,
                          source_files: List[str],
                          top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents from specific source files
        
        Args:
            query: Search query
            source_files: List of source file names to search within
            top_k: Number of results to return
            
        Returns:
            Filtered results from specified sources
        """
        # Get all results first
        all_results = self.retrieve(query, top_k=top_k * 10)  # Get more to filter
        
        # Filter by source files
        filtered_results = []
        for result in all_results:
            source_file = result.get('metadata', {}).get('source_file', '')
            if source_file in source_files:
                filtered_results.append(result)
                
                if len(filtered_results) >= (top_k or self.top_k):
                    break
        
        logger.info(f"Retrieved {len(filtered_results)} results from specified sources")
        return filtered_results
    
    def retrieve_with_context(self, 
                             query: str,
                             context_window: int = 1,
                             top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents with surrounding context chunks
        
        Args:
            query: Search query
            context_window: Number of surrounding chunks to include
            top_k: Number of results to return
            
        Returns:
            Results with expanded context
        """
        # Get initial results
        results = self.retrieve(query, top_k=top_k)
        
        # For each result, try to get surrounding chunks
        # This is simplified - in practice, you'd need to track chunk relationships
        expanded_results = []
        
        for result in results:
            # Add the main result
            expanded_results.append(result)
            
            # Try to find related chunks from the same source
            source_file = result.get('metadata', {}).get('source_file', '')
            chunk_id = result.get('metadata', {}).get('chunk_id', -1)
            
            if source_file and chunk_id >= 0:
                # Look for neighboring chunks (this is a simplified approach)
                for doc_id, metadata in self.vector_store.docstore.items():
                    if (metadata.get('source_file') == source_file and 
                        metadata.get('chunk_id', -1) in range(
                            max(0, chunk_id - context_window),
                            chunk_id + context_window + 1) and
                        metadata.get('chunk_id', -1) != chunk_id):
                        
                        context_result = {
                            'id': doc_id,
                            'score': result['score'] * 0.8,  # Lower score for context
                            'text': metadata.get('text', ''),
                            'metadata': {k: v for k, v in metadata.items() if k != 'text'},
                            'is_context': True
                        }
                        expanded_results.append(context_result)
        
        # Sort by score and chunk order
        expanded_results.sort(key=lambda x: (
            x.get('metadata', {}).get('source_file', ''),
            x.get('metadata', {}).get('chunk_id', 0)
        ))
        
        logger.info(f"Expanded {len(results)} results to {len(expanded_results)} with context")
        return expanded_results
    
    def get_document_sources(self) -> List[Dict[str, Any]]:
        """
        Get list of all document sources in the vector store
        
        Returns:
            List of source file information
        """
        sources = {}
        
        for doc_id, metadata in self.vector_store.docstore.items():
            source_file = metadata.get('source_file', 'unknown')
            
            if source_file not in sources:
                sources[source_file] = {
                    'filename': source_file,
                    'chunk_count': 0,
                    'total_chars': 0,
                    'file_type': metadata.get('source_extension', 'unknown')
                }
            
            sources[source_file]['chunk_count'] += 1
            sources[source_file]['total_chars'] += len(metadata.get('text', ''))
        
        return list(sources.values())
    
    def search_by_keywords(self, keywords: List[str], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Simple keyword-based search within documents
        
        Args:
            keywords: List of keywords to search for
            top_k: Number of results to return
            
        Returns:
            Documents containing the keywords
        """
        results = []
        keywords_lower = [kw.lower() for kw in keywords]
        
        for doc_id, metadata in self.vector_store.docstore.items():
            text = metadata.get('text', '').lower()
            
            # Count keyword matches
            matches = sum(1 for kw in keywords_lower if kw in text)
            
            if matches > 0:
                score = matches / len(keywords)  # Simple scoring
                
                result = {
                    'id': doc_id,
                    'score': score,
                    'text': metadata.get('text', ''),
                    'metadata': {k: v for k, v in metadata.items() if k != 'text'},
                    'keyword_matches': matches
                }
                results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit results
        k = top_k or self.top_k
        results = results[:k]
        
        logger.info(f"Found {len(results)} documents matching keywords")
        return results
    
    def hybrid_search(self, 
                     query: str,
                     keywords: List[str] = None,
                     semantic_weight: float = 0.7,
                     top_k: int = None) -> List[Dict[str, Any]]:
        """
        Combine semantic and keyword search
        
        Args:
            query: Search query
            keywords: Optional keywords for keyword search
            semantic_weight: Weight for semantic search (0-1)
            top_k: Number of results to return
            
        Returns:
            Combined search results
        """
        k = top_k or self.top_k
        
        # Get semantic search results
        semantic_results = self.retrieve(query, top_k=k * 2)
        
        # Get keyword search results if keywords provided
        keyword_results = []
        if keywords:
            keyword_results = self.search_by_keywords(keywords, top_k=k * 2)
        
        # Combine and re-score results
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = result['id']
            combined_results[doc_id] = result.copy()
            combined_results[doc_id]['semantic_score'] = result['score']
            combined_results[doc_id]['keyword_score'] = 0.0
        
        # Add keyword results
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in combined_results:
                combined_results[doc_id]['keyword_score'] = result['score']
            else:
                combined_results[doc_id] = result.copy()
                combined_results[doc_id]['semantic_score'] = 0.0
                combined_results[doc_id]['keyword_score'] = result['score']
        
        # Calculate hybrid scores
        for doc_id, result in combined_results.items():
            semantic_score = result['semantic_score']
            keyword_score = result['keyword_score']
            
            hybrid_score = (semantic_weight * semantic_score + 
                           (1 - semantic_weight) * keyword_score)
            
            result['score'] = hybrid_score
            result['hybrid_score'] = hybrid_score
        
        # Sort and limit results
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x['score'], reverse=True)
        final_results = final_results[:k]
        
        logger.info(f"Hybrid search returned {len(final_results)} results")
        return final_results
