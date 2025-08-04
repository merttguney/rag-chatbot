"""
RAG Runner - Main orchestrator for the RAG system
Handles document processing, indexing, and query pipeline
"""

import os
import time
from typing import List, Dict, Any, Optional
from loguru import logger

from .services.document_service import FileLoader
from .utils.text_processing import TextSplitter
from .services.embedding_service import Embedder
from .storage.vector_store_impl import SimpleVectorStore
from .retriever import Retriever
from .core.pipeline import RAGPipeline
from .core.config import config


class RAGRunner:
    """Main class that orchestrates the entire RAG system"""
    
    def __init__(self, 
                 data_dir: str = None,
                 use_openai_embeddings: bool = True):
        """
        Initialize RAG Runner
        
        Args:
            data_dir: Directory containing documents to process
            use_openai_embeddings: Whether to use OpenAI embeddings vs local
        """
        self.data_dir = data_dir or config.DATA_DIR
        
        # Initialize components
        self.file_loader = FileLoader()
        self.text_splitter = TextSplitter()
        self.embedder = Embedder(use_openai=use_openai_embeddings)
        self.vector_store = SimpleVectorStore(dimension=self.embedder.embedding_dimension)
        self.retriever = Retriever(vector_store=self.vector_store, embedder=self.embedder)
        self.pipeline = RAGPipeline(retriever=self.retriever, embedder=self.embedder)
        
        # State tracking
        self.is_initialized = False
        self.indexed_files = set()
        self.processed_documents = {}
        self.conversation_history = []
        
        logger.info(f"RAG Runner initialized - Data dir: {self.data_dir}, OpenAI: {use_openai_embeddings}")
    
    def setup(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Setup the RAG system - load and index documents with progress tracking
        
        Args:
            force_reindex: Whether to force reindexing of all documents
            
        Returns:
            Statistics about the setup process
        """
        try:
            logger.info("Starting RAG system setup...")
            stats = {
                'documents_processed': 0,
                'chunks_created': 0,
                'embeddings_created': 0,
                'vector_ids': 0,
                'errors': [],
                'processing_time': 0
            }
            
            start_time = time.time()
            
            # Step 1: Check if we need to process documents
            if not force_reindex and self.vector_store.is_initialized:
                logger.info("Vector store already exists, loading existing index...")
                self.vector_store.load()
                self.is_initialized = True
                
                # Restore indexed files from vector store metadata
                try:
                    vector_stats = self.vector_store.get_stats()
                    if 'indexed_files' in vector_stats:
                        self.indexed_files = set(vector_stats['indexed_files'])
                    else:
                        # Fallback: extract filenames from docstore
                        vector_docstore = getattr(self.vector_store, 'docstore', {})
                        self.indexed_files = set()
                        for doc_data in vector_docstore.values():
                            if isinstance(doc_data, dict) and 'filename' in doc_data:
                                self.indexed_files.add(doc_data['filename'])
                            elif isinstance(doc_data, dict) and 'source_file' in doc_data:
                                self.indexed_files.add(doc_data['source_file'])
                    
                    # Restore processed documents info
                    self.processed_documents = {}
                    vector_docstore = getattr(self.vector_store, 'docstore', {})
                    for filename in self.indexed_files:
                        self.processed_documents[filename] = {
                            'path': f"{self.data_dir}/{filename}",
                            'size': 0,  # Size info not preserved, would need to re-read file
                            'chunks': sum(1 for doc in vector_docstore.values() 
                                        if isinstance(doc, dict) and 
                                        (doc.get('filename') == filename or doc.get('source_file') == filename))
                        }
                    
                    logger.info(f"Restored {len(self.indexed_files)} indexed files from vector store")
                except Exception as e:
                    logger.warning(f"Could not restore indexed files info: {str(e)}")
                    self.indexed_files = set()
                    self.processed_documents = {}
                
                # Get existing files info
                existing_stats = self.vector_store.get_stats()
                stats.update(existing_stats)
                stats['processing_time'] = time.time() - start_time
                return stats
            
            # Step 2: Load documents
            logger.info(f"Loading documents from: {self.data_dir}")
            documents = self.file_loader.load_documents_from_directory(self.data_dir)
            
            if not documents:
                logger.warning("No documents found to process")
                return stats
            
            stats['documents_processed'] = len(documents)
            logger.info(f"Loaded {len(documents)} documents")
            
            # Step 3: Split documents into chunks
            logger.info("Splitting documents into chunks...")
            all_chunks = self.text_splitter.split_documents(documents)
            
            stats['chunks_created'] = len(all_chunks)
            logger.info(f"Created {len(all_chunks)} chunks")
            
            if not all_chunks:
                logger.warning("No chunks created from documents")
                return stats
            
            # Step 4: Create embeddings in batches
            logger.info("Creating embeddings...")
            chunk_texts = [chunk['text'] for chunk in all_chunks]
            embeddings = self.embedder.embed_texts(chunk_texts)
            
            stats['embeddings_created'] = len(embeddings)
            
            # Step 5: Store in vector database
            logger.info("Storing embeddings in vector database...")
            
            # Prepare metadata with text content
            metadata_list = []
            for i, chunk in enumerate(all_chunks):
                metadata = chunk['metadata'].copy()
                metadata['text'] = chunk['text']
                metadata_list.append(metadata)
            
            vector_ids = self.vector_store.add_embeddings(
                embeddings=embeddings,
                metadata=metadata_list
            )
            
            stats['vector_ids'] = len(vector_ids) if vector_ids else 0
            
            # Step 6: Save vector store
            self.vector_store.save()
            
            # Update state
            self.is_initialized = True
            self.indexed_files = set(doc['metadata']['filename'] for doc in documents)
            self.processed_documents = {
                doc['metadata']['filename']: {
                    'path': doc['metadata']['filepath'],
                    'size': doc['metadata']['size_bytes'],
                    'chunks': len([c for c in all_chunks if c['metadata']['source_file'] == doc['metadata']['filename']])
                }
                for doc in documents
            }
            
            stats['processing_time'] = time.time() - start_time
            logger.info(f"RAG system setup completed successfully in {stats['processing_time']:.2f}s")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error during RAG setup: {str(e)}")
            stats['errors'].append(str(e))
            raise
    
    def add_documents(self, document_paths: List[str] = None) -> Dict[str, Any]:
        """
        Add new documents to the existing index
        
        Args:
            document_paths: Specific documents to add (optional)
            
        Returns:
            Processing statistics
        """
        try:
            logger.info("Adding new documents to index...")
            
            if document_paths:
                documents = []
                for path in document_paths:
                    doc = self.file_loader.load_single_document(path)
                    if doc:
                        documents.append(doc)
            else:
                documents = self.file_loader.load_documents_from_directory(self.data_dir)
                # Filter out already indexed documents
                documents = [doc for doc in documents 
                           if doc['metadata']['filename'] not in self.indexed_files]
            
            if not documents:
                return {'documents_processed': 0, 'message': 'No new documents to add'}
            
            # Process new documents
            all_chunks = self.text_splitter.split_documents(documents)
            chunk_texts = [chunk['text'] for chunk in all_chunks]
            embeddings = self.embedder.embed_texts(chunk_texts)
            
            # Add to vector store
            vector_ids = self.vector_store.add_embeddings(
                embeddings=embeddings,
                texts=chunk_texts,
                metadatas=[chunk['metadata'] for chunk in all_chunks]
            )
            
            # Update state
            for doc in documents:
                self.indexed_files.add(doc['metadata']['filename'])
            
            self.vector_store.save()
            
            return {
                'documents_processed': len(documents),
                'chunks_created': len(all_chunks),
                'embeddings_created': len(embeddings),
                'vector_ids': len(vector_ids) if vector_ids else 0
            }
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def ask_question(self, 
                    question: str, 
                    use_history: bool = True,
                    **kwargs) -> Dict[str, Any]:
        """
        Ask a question using the RAG system
        
        Args:
            question: User question
            use_history: Whether to use conversation history for context
            **kwargs: Additional parameters for the pipeline
            
        Returns:
            Answer with sources and metadata
        """
        if not self.is_initialized:
            logger.warning("RAG system not initialized, attempting setup...")
            try:
                self.setup()
            except Exception as e:
                return {
                    'answer': f"Sistem henüz hazır değil. Lütfen önce belgeleri yükleyin. Hata: {str(e)}",
                    'sources': [],
                    'confidence': 0.0,
                    'metadata': {'error': str(e)}
                }

        try:
            if use_history and self.conversation_history:
                result = self.pipeline.ask_with_history(
                    question, 
                    self.conversation_history,
                    **kwargs
                )
            else:
                result = self.pipeline.process_question(question, **kwargs)
            
            # Add to conversation history
            if use_history:
                self.conversation_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'timestamp': time.time()
                })
                
                # Keep only last 10 exchanges
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
            
            logger.info(f"Question answered with confidence: {result.get('confidence', 0):.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                'answer': f"Soruyu yanıtlarken bir hata oluştu: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and statistics
        
        Returns:
            System status information
        """
        # Auto-initialize if not already done and vector store exists
        if not self.is_initialized and self.vector_store.is_initialized:
            try:
                self.setup()
            except Exception as e:
                logger.warning(f"Could not auto-initialize for status check: {str(e)}")
        
        status = {
            'is_initialized': self.is_initialized,
            'indexed_files_count': len(self.indexed_files),
            'indexed_files': list(self.indexed_files),
            'data_directory': self.data_dir,
            'conversation_history_length': len(self.conversation_history)
        }
        
        if self.is_initialized:
            try:
                vector_stats = self.vector_store.get_stats()
                status['vector_store'] = vector_stats
            except Exception as e:
                status['vector_store'] = {'error': str(e)}
        
        return status
    
    def health_check(self) -> Dict[str, bool]:
        """
        Perform system health check
        
        Returns:
            Health status for each component
        """
        health = {
            'config_valid': True,
            'embedder_working': False,
            'vector_store_accessible': False,
            'documents_indexed': False,
            'overall_healthy': False
        }
        
        try:
            # Check config
            config.validate_config()
        except Exception:
            health['config_valid'] = False
        
        # Check embedder
        try:
            test_embedding = self.embedder.embed_text("test")
            health['embedder_working'] = len(test_embedding) > 0
        except Exception:
            health['embedder_working'] = False
        
        # Check vector store
        try:
            health['vector_store_accessible'] = self.vector_store.is_initialized
        except Exception:
            health['vector_store_accessible'] = False
        
        # Check if documents are indexed
        health['documents_indexed'] = len(self.indexed_files) > 0
        
        # Overall health
        health['overall_healthy'] = all([
            health['config_valid'],
            health['embedder_working'],
            health['vector_store_accessible'],
            health['documents_indexed']
        ])
        
        return health
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_document_info(self, filename: str = None) -> Dict[str, Any]:
        """
        Get information about processed documents
        
        Args:
            filename: Specific file to get info for (optional)
            
        Returns:
            Document information
        """
        if filename:
            return self.processed_documents.get(filename, {})
        else:
            return self.processed_documents
    
    def remove_document(self, filename: str) -> bool:
        """
        Remove a document from the index
        
        Args:
            filename: Name of file to remove
            
        Returns:
            Success status
        """
        try:
            if filename not in self.indexed_files:
                logger.warning(f"Document {filename} not found in index")
                return False
            
            # Remove from vector store (this would need implementation in vector_store)
            # For now, we'll mark it as removed
            self.indexed_files.discard(filename)
            
            if filename in self.processed_documents:
                del self.processed_documents[filename]
            
            logger.info(f"Document {filename} removed from index")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document {filename}: {str(e)}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            # Vector store stats
            vector_stats = self.vector_store.get_stats()
            
            # Document stats
            doc_count = len(self.indexed_files)
            processed_docs = len(self.processed_documents)
            
            return {
                'is_initialized': self.is_initialized,
                'total_documents': doc_count,
                'processed_documents': processed_docs,
                'total_chunks': vector_stats.get('docstore_entries', 0),
                'total_embeddings': vector_stats.get('total_embeddings', 0),
                'embedding_dimension': vector_stats.get('embedding_dimension', 0),
                'data_directory': self.data_dir,
                'llm_model': self.pipeline.llm_model if hasattr(self.pipeline, 'llm_model') else 'gpt-3.5-turbo',
                'embedding_model': self.embedder.model_name if hasattr(self.embedder, 'model_name') else 'text-embedding-3-small',
                'indexed_files': list(self.indexed_files),
                'conversation_count': len(self.conversation_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {
                'is_initialized': False,
                'total_documents': 0,
                'processed_documents': 0,
                'total_chunks': 0,
                'total_embeddings': 0,
                'embedding_dimension': 0,
                'data_directory': self.data_dir,
                'llm_model': 'unknown',
                'embedding_model': 'unknown',
                'indexed_files': [],
                'conversation_count': 0
            }
