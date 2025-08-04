"""
Enterprise Application Factory
Creates configured Flask application with proper dependency injection
"""

import os
from flask import Flask
from loguru import logger

from ..services.rag_orchestrator import RAGOrchestrator
from ..core.controllers import create_controllers
from ..core.config import config


def create_app(data_dir: str = None, use_openai_embeddings: bool = True) -> Flask:
    """
    Application factory for creating configured Flask app
    
    Args:
        data_dir: Data directory path
        use_openai_embeddings: Whether to use OpenAI embeddings
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Configure Flask
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['DEBUG'] = config.DEBUG
    
    # Initialize RAG system
    orchestrator = RAGOrchestrator(
        data_dir=data_dir,
        use_openai_embeddings=use_openai_embeddings
    )
    
    # Store orchestrator in app context for access in views
    app.orchestrator = orchestrator
    
    # Create controllers
    controllers = create_controllers(orchestrator)
    
    # Register routes
    register_routes(app, controllers)
    
    # Initialize system on first request
    with app.app_context():
        try:
            logger.info("⚙️  RAG sistemi kuruluyor...")
            orchestrator.setup()
            logger.info("✅ RAG sistemi başarıyla yüklendi")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
    
    return app


def register_routes(app: Flask, controllers: dict) -> None:
    """Register all application routes"""
    
    # Static routes
    app.add_url_rule('/', 'index', controllers['static'].index)
    app.add_url_rule('/favicon.png', 'favicon', controllers['static'].favicon)
    
    # API routes  
    app.add_url_rule('/api/query', 'api_query', 
                     controllers['rag'].query, methods=['POST'])
    app.add_url_rule('/api/health', 'api_health', 
                     controllers['rag'].health, methods=['GET'])
    app.add_url_rule('/api/status', 'api_status', 
                     controllers['rag'].status, methods=['GET'])
    
    # Legacy compatibility route
    @app.route('/ask', methods=['POST'])
    def legacy_ask():
        """Legacy endpoint for backward compatibility"""
        return controllers['rag'].query()


if __name__ == '__main__':
    # For direct execution
    app = create_app()
    
    print("🚀 Starting Enterprise RAG System v3.0")
    print("📊 Architecture: Clean Architecture + SOLID Principles")
    print("🔧 Components: Factory Pattern + Dependency Injection")
    print("📍 Web Interface: http://localhost:8086")
    
    app.run(
        host='0.0.0.0',
        port=8086,
        debug=config.DEBUG
    )
