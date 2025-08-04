"""
Text splitter module for chunking documents
Uses RecursiveCharacterTextSplitter for optimal chunking
"""

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from ..core.config import config


class TextSplitter:
    """Text splitter for creating manageable chunks from documents"""
    
    def __init__(self, 
                 chunk_size: int = None, 
                 chunk_overlap: int = None,
                 separators: List[str] = None):
        """
        Initialize text splitter
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            separators: List of separators to use for splitting
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        # Default separators for better text splitting
        self.separators = separators or [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            " ",     # Spaces
            ".",     # Sentences
            ",",     # Clauses
            ""       # Characters
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
        
        logger.info(f"TextSplitter initialized with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}")
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Text to split
            metadata: Original document metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for splitting")
            return []
        
        try:
            # Split the text into chunks
            chunks = self.splitter.split_text(text)
            
            # Create chunk objects with metadata
            chunk_objects = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    'chunk_id': i,
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks)
                }
                
                # Add original document metadata
                if metadata:
                    chunk_metadata.update({
                        'source_file': metadata.get('filename', 'unknown'),
                        'source_path': metadata.get('filepath', 'unknown'),
                        'source_extension': metadata.get('extension', 'unknown'),
                        'source_size': metadata.get('size_bytes', 0)
                    })
                
                chunk_objects.append({
                    'text': chunk,
                    'metadata': chunk_metadata
                })
            
            logger.info(f"Split text into {len(chunks)} chunks")
            return chunk_objects
            
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            raise
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split multiple documents into chunks
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            try:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                
                chunks = self.split_text(content, metadata)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error splitting document {metadata.get('filename', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Split {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'total_characters': 0
            }
        
        chunk_sizes = [len(chunk['text']) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes),
            'files_processed': len(set(chunk.get('metadata', {}).get('source_file', 'unknown') for chunk in chunks))
        }
    
    def update_settings(self, chunk_size: int = None, chunk_overlap: int = None):
        """Update splitter settings"""
        if chunk_size:
            self.chunk_size = chunk_size
        if chunk_overlap:
            self.chunk_overlap = chunk_overlap
        
        # Recreate splitter with new settings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
        
        logger.info(f"Updated TextSplitter settings: chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}")
    
    def optimize_chunk_size(self, text: str) -> int:
        """
        Optimize chunk size based on text characteristics
        
        Args:
            text: Input text to analyze
            
        Returns:
            Optimized chunk size
        """
        text_length = len(text)
        
        # For short texts, use smaller chunks
        if text_length < 1000:
            return min(self.chunk_size, 200)
        # For medium texts, use standard chunks
        elif text_length < 10000:
            return self.chunk_size
        # For long texts, use larger chunks for better context
        else:
            return min(self.chunk_size * 2, 2000)
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before splitting
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Fix common encoding issues
        text = text.replace('\xa0', ' ')  # Non-breaking space
        text = text.replace('\u200b', '')  # Zero-width space
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text
    
    def create_semantic_chunks(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Create semantic chunks that try to preserve meaning
        
        Args:
            text: Text to split
            metadata: Document metadata
            
        Returns:
            List of semantic chunks
        """
        preprocessed_text = self.preprocess_text(text)
        
        # Try to split by semantic boundaries first
        paragraphs = preprocessed_text.split('\n\n')
        semantic_chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:
                    semantic_chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Paragraph itself is too long, use regular splitting
                    regular_chunks = self.split_text(paragraph, metadata)
                    semantic_chunks.extend([chunk['text'] for chunk in regular_chunks])
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add the last chunk
        if current_chunk:
            semantic_chunks.append(current_chunk.strip())
        
        # Convert to chunk objects
        chunk_objects = []
        for i, chunk in enumerate(semantic_chunks):
            chunk_metadata = {
                'chunk_id': i,
                'chunk_size': len(chunk),
                'total_chunks': len(semantic_chunks),
                'chunk_type': 'semantic'
            }
            
            if metadata:
                chunk_metadata.update({
                    'source_file': metadata.get('filename', 'unknown'),
                    'source_path': metadata.get('filepath', 'unknown'),
                    'source_extension': metadata.get('extension', 'unknown'),
                    'source_size': metadata.get('size_bytes', 0)
                })
            
            chunk_objects.append({
                'text': chunk,
                'metadata': chunk_metadata
            })
        
        logger.info(f"Created {len(semantic_chunks)} semantic chunks")
        return chunk_objects
