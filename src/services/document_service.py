"""
File loader module for PDF & TXT files
Unified module for reading different file formats
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

# PDF processing
import PyPDF2
from PyPDF2 import PdfReader

# Text processing
import docx
from loguru import logger

from ..core.config import config


class FileLoader:
    """Unified file loader for PDF, TXT, and DOCX files"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf', '.txt', '.md', '.docx']
        logger.info("FileLoader initialized")
    
    def load_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load a single file and return its content with metadata
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            Dict containing content, metadata, and file info
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        try:
            if file_extension == '.pdf':
                content = self._load_pdf(file_path)
            elif file_extension in ['.txt', '.md']:
                content = self._load_text(file_path)
            elif file_extension == '.docx':
                content = self._load_docx(file_path)
            
            # Prepare metadata
            file_stats = os.stat(file_path)
            metadata = {
                'filename': os.path.basename(file_path),
                'filepath': file_path,
                'extension': file_extension,
                'size_bytes': file_stats.st_size,
                'modified_time': file_stats.st_mtime,
                'content_length': len(content)
            }
            
            logger.info(f"Successfully loaded file: {file_path}")
            
            return {
                'content': content,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def _load_pdf(self, file_path: str) -> str:
        """Load PDF file content"""
        content = ""
        
        try:
            # Try with pypdf first (more reliable)
            reader = PdfReader(file_path)
            for page in reader.pages:
                content += page.extract_text() + "\n"
        except Exception as e:
            logger.warning(f"pypdf failed for {file_path}, trying PyPDF2: {str(e)}")
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        content += page.extract_text() + "\n"
            except Exception as e2:
                logger.error(f"Both PDF readers failed for {file_path}: {str(e2)}")
                raise
        
        return content.strip()
    
    def _load_text(self, file_path: str) -> str:
        """Load text file content"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                return content
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file {file_path} with any supported encoding")
    
    def _load_docx(self, file_path: str) -> str:
        """Load DOCX file content"""
        doc = docx.Document(file_path)
        content = ""
        
        for paragraph in doc.paragraphs:
            content += paragraph.text + "\n"
        
        return content.strip()
    
    def load_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Load all supported files from a directory
        
        Args:
            directory_path (str): Path to the directory
            
        Returns:
            List of document dictionaries
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        documents = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = Path(file_path).suffix.lower()
                
                if file_extension in self.supported_extensions:
                    try:
                        doc = self.load_file(file_path)
                        documents.append(doc)
                    except Exception as e:
                        logger.error(f"Failed to load {file_path}: {str(e)}")
                        continue
        
        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents
    
    def load_documents_from_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Alias for load_directory method for compatibility
        
        Args:
            directory_path (str): Path to the directory
            
        Returns:
            List of document dictionaries
        """
        return self.load_directory(directory_path)
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information without loading content"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_stats = os.stat(file_path)
        return {
            'filename': os.path.basename(file_path),
            'filepath': file_path,
            'extension': Path(file_path).suffix.lower(),
            'size_bytes': file_stats.st_size,
            'modified_time': file_stats.st_mtime,
            'is_supported': Path(file_path).suffix.lower() in self.supported_extensions
        }
