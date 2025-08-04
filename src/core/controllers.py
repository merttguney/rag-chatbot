"""
Web API Controllers for RAG System
Clean separation of web concerns from business logic
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from typing import Dict, Any, Optional
from loguru import logger

from ..services.rag_orchestrator import RAGOrchestrator
from ..core.config import config


class RAGController:
    """Controller handling RAG-related web requests"""
    
    def __init__(self, orchestrator: RAGOrchestrator):
        self.orchestrator = orchestrator
    
    def query(self) -> Dict[str, Any]:
        """Handle query requests"""
        try:
            data = request.get_json()
            if not data or 'question' not in data:
                return {'error': 'Soru alanı gerekli'}, 400
            
            question = data['question'].strip()
            if not question:
                return {'error': 'Soru alanı boş olamaz'}, 400
            
            # Process question
            result = self.orchestrator.ask_question(question)
            
            return {
                'answer': result.get('answer', 'Cevap bulunamadı'),
                'confidence': result.get('confidence', 0.0),
                'sources': result.get('sources', []),
                'metadata': result.get('metadata', {})
            }
            
        except ValueError as e:
            logger.warning(f"Validation error: {str(e)}")
            return {'error': str(e)}, 400
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}")
            return {'error': 'İç sunucu hatası'}, 500
    
    def health(self) -> Dict[str, Any]:
        """Handle health check requests"""
        try:
            health_info = self.orchestrator.health_check()
            return {
                'status': 'healthy' if health_info['initialized'] else 'initializing',
                'service': 'rag-chatbot-enterprise',
                'version': '3.0.0',
                'details': health_info
            }
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }, 500
    
    def status(self) -> Dict[str, Any]:
        """Handle status requests"""
        try:
            health_info = self.orchestrator.health_check()
            return {
                'system': 'enterprise',
                'initialized': health_info['initialized'], 
                'components_loaded': health_info['components_loaded'],
                'data_directory': health_info['data_directory']
            }
        except Exception as e:
            logger.error(f"Status check error: {str(e)}")
            return {'error': str(e)}, 500


class StaticController:
    """Controller for static resources"""
    
    @staticmethod
    def favicon():
        """Serve favicon"""
        return send_from_directory('.', 'favicon.png')
    
    @staticmethod
    def index():
        """Serve main page - keeping original UI"""
        # Import the original HTML template here to maintain UI consistency
        from ...web_app import HTML_TEMPLATE
        return render_template_string(HTML_TEMPLATE)


def create_controllers(orchestrator: RAGOrchestrator) -> Dict[str, Any]:
    """Factory function to create all controllers"""
    return {
        'rag': RAGController(orchestrator),
        'static': StaticController()
    }
