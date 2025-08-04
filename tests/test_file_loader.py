"""
Unit tests for document_service module
"""

import os
import pytest
import tempfile
from pathlib import Path

from src.services.document_service import FileLoader


class TestFileLoader:
    """Test cases for FileLoader class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.file_loader = FileLoader()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_temp_file(self, content: str, extension: str) -> str:
        """Create a temporary file with given content and extension"""
        temp_file = os.path.join(self.temp_dir, f"test{extension}")
        
        if extension == '.txt':
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return temp_file
    
    def test_load_text_file(self):
        """Test loading a text file"""
        content = "Bu bir test dosyasıdır.\nTürkçe karakterler: ğüşıöç"
        temp_file = self.create_temp_file(content, '.txt')
        
        result = self.file_loader.load_file(temp_file)
        
        assert result['content'] == content
        assert result['metadata']['filename'] == 'test.txt'
        assert result['metadata']['extension'] == '.txt'
        assert result['metadata']['content_length'] == len(content)
    
    def test_unsupported_file_type(self):
        """Test loading unsupported file type"""
        temp_file = self.create_temp_file("test", '.xyz')
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            self.file_loader.load_file(temp_file)
    
    def test_file_not_found(self):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            self.file_loader.load_file("/non/existent/file.txt")
    
    def test_supported_extensions(self):
        """Test that all expected extensions are supported"""
        expected_extensions = ['.pdf', '.txt', '.md', '.docx']
        assert self.file_loader.supported_extensions == expected_extensions
    
    def test_get_file_info(self):
        """Test getting file information"""
        content = "Test content"
        temp_file = self.create_temp_file(content, '.txt')
        
        info = self.file_loader.get_file_info(temp_file)
        
        assert info['filename'] == 'test.txt'
        assert info['extension'] == '.txt'
        assert info['is_supported'] == True
        assert info['size_bytes'] > 0
    
    def test_load_directory_empty(self):
        """Test loading from empty directory"""
        result = self.file_loader.load_directory(self.temp_dir)
        assert result == []
    
    def test_load_directory_with_files(self):
        """Test loading from directory with files"""
        # Create test files
        self.create_temp_file("File 1 content", '.txt')
        self.create_temp_file("File 2 content", '.md')
        
        result = self.file_loader.load_directory(self.temp_dir)
        
        assert len(result) == 2
        assert all('content' in doc and 'metadata' in doc for doc in result)


if __name__ == "__main__":
    pytest.main([__file__])
