"""
Integration tests for RAG system
"""

import os
import pytest
import tempfile
from unittest.mock import Mock, patch

from src.rag_runner import RAGRunner
from src.core.config import config


class TestRAGIntegration:
    """Integration tests for complete RAG system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock OpenAI to avoid API calls in tests
        self.mock_openai_client = Mock()
        self.mock_embeddings_response = Mock()
        self.mock_embeddings_response.data = [Mock(embedding=[0.1] * 1536)]
        self.mock_openai_client.embeddings.create.return_value = self.mock_embeddings_response
        
        self.mock_chat_response = Mock()
        self.mock_chat_response.choices = [Mock(message=Mock(content="Test answer"))]
        self.mock_openai_client.chat.completions.create.return_value = self.mock_chat_response
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_document(self, content: str, filename: str = "test.txt") -> str:
        """Create a test document"""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    @patch('src.embedder.OpenAI')
    @patch('src.pipeline.OpenAI')
    def test_complete_rag_workflow(self, mock_pipeline_openai, mock_embedder_openai):
        """Test complete RAG workflow from document to answer"""
        # Setup mocks
        mock_embedder_openai.return_value = self.mock_openai_client
        mock_pipeline_openai.return_value = self.mock_openai_client
        
        # Create test document
        test_content = """
        Bu bir test belgesidir.
        Yapay zeka hakkında bilgiler içerir.
        Machine learning çok önemlidir.
        """
        doc_path = self.create_test_document(test_content)
        
        # Initialize RAG system
        rag_runner = RAGRunner(data_dir=self.temp_dir)
        
        # Process document
        stats = rag_runner.process_documents([doc_path])
        
        assert stats['documents_processed'] == 1
        assert stats['chunks_created'] > 0
        assert stats['embeddings_created'] > 0
        
        # Ask question
        question = "Yapay zeka hakkında ne biliyorsun?"
        result = rag_runner.ask_question(question)
        
        assert 'answer' in result
        assert 'sources' in result
        assert result['question'] == question
    
    @patch('src.embedder.OpenAI')
    def test_document_addition_and_removal(self, mock_openai):
        """Test adding and removing documents"""
        mock_openai.return_value = self.mock_openai_client
        
        # Create test documents
        doc1_path = self.create_test_document("Document 1 content", "doc1.txt")
        doc2_path = self.create_test_document("Document 2 content", "doc2.txt")
        
        # Initialize RAG system
        rag_runner = RAGRunner(data_dir=self.temp_dir)
        
        # Add first document
        stats1 = rag_runner.add_document(doc1_path)
        assert stats1['documents_processed'] == 1
        
        # Add second document
        stats2 = rag_runner.add_document(doc2_path)
        assert stats2['documents_processed'] == 1
        
        # Check system status
        status = rag_runner.get_system_status()
        assert status['indexed_files_count'] == 2
        assert 'doc1.txt' in status['indexed_files']
        assert 'doc2.txt' in status['indexed_files']
        
        # Remove first document
        removed_count = rag_runner.remove_document('doc1.txt')
        assert removed_count > 0
        
        # Check status after removal
        status_after = rag_runner.get_system_status()
        assert 'doc1.txt' not in status_after['indexed_files']
    
    @patch('src.embedder.OpenAI')
    def test_search_functionality(self, mock_openai):
        """Test different search modes"""
        mock_openai.return_value = self.mock_openai_client
        
        # Create test document with specific content
        test_content = """
        Python programlama dili çok popülerdir.
        Machine learning için sıkça kullanılır.
        Data science alanında vazgeçilmezdir.
        Web development için de uygundur.
        """
        doc_path = self.create_test_document(test_content)
        
        # Initialize and setup RAG system
        rag_runner = RAGRunner(data_dir=self.temp_dir)
        rag_runner.process_documents([doc_path])
        
        # Test semantic search
        semantic_results = rag_runner.search_documents(
            query="Python nedir?",
            search_type="semantic"
        )
        assert len(semantic_results) > 0
        
        # Test keyword search
        keyword_results = rag_runner.search_documents(
            query="Python programlama",
            search_type="keyword"
        )
        assert len(keyword_results) > 0
        
        # Test hybrid search
        hybrid_results = rag_runner.search_documents(
            query="Python programlama",
            search_type="hybrid"
        )
        assert len(hybrid_results) > 0
    
    def test_health_check(self):
        """Test system health check"""
        # This test doesn't need OpenAI mocking for basic health check
        rag_runner = RAGRunner(data_dir=self.temp_dir, use_openai_embeddings=False)
        
        health = rag_runner.health_check()
        
        assert 'config_valid' in health
        assert 'embedder_working' in health
        assert 'vector_store_accessible' in health
        assert 'documents_indexed' in health
        assert 'overall_healthy' in health
    
    @patch('src.embedder.OpenAI')
    def test_conversation_history(self, mock_openai):
        """Test conversation with history"""
        mock_openai.return_value = self.mock_openai_client
        
        # Create test document
        test_content = "Yapay zeka gelecekte çok önemli olacak."
        doc_path = self.create_test_document(test_content)
        
        # Initialize RAG system
        rag_runner = RAGRunner(data_dir=self.temp_dir)
        rag_runner.process_documents([doc_path])
        
        # First question
        result1 = rag_runner.chat("Yapay zeka nedir?")
        assert 'answer' in result1
        
        # Second question with history
        history = [{'question': "Yapay zeka nedir?", 'answer': result1['answer']}]
        result2 = rag_runner.chat("Gelecekte ne olacak?", conversation_history=history)
        assert 'answer' in result2


if __name__ == "__main__":
    pytest.main([__file__])
