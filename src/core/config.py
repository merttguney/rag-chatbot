"""
Configuration module for RAG Chatbot
Contains all model, chunking, and vector DB settings
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Flask Settings
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-not-for-production")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # OpenAI Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Text Splitting Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Vector Store Settings
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "simple")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    
    # Retrieval Settings
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    
    # File Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Go up to project root
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")
    LOGS_DIR = os.path.join(ROOT_DIR, "logs")
    
    # FAISS Settings
    FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_index.index")
    DOCSTORE_PATH = os.path.join(EMBEDDINGS_DIR, "docstore.pkl")
    
    # UI Settings
    GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
    GRADIO_SHARE = os.getenv("GRADIO_SHARE", "False").lower() == "true"
    
    # Logging Settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.path.join(LOGS_DIR, "rag_chatbot.log")
    
    # Admin Panel Chatbot Specific Settings
    USE_SEMANTIC_CHUNKING = os.getenv("USE_SEMANTIC_CHUNKING", "True").lower() == "true"
    MAX_TOKENS_PER_REQUEST = int(os.getenv("MAX_TOKENS_PER_REQUEST", "4000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))  # Low temperature for precise admin guidance
    
    # Performance Settings
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    
    # Cache Settings
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "True").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
    # Local Model Settings (for when not using OpenAI)
    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama2")
    
    # Admin Panel Specific Settings
    CHATBOT_NAME = os.getenv("CHATBOT_NAME", "Admin Panel Asistanı")
    SYSTEM_NAME = os.getenv("SYSTEM_NAME", "E-ticaret Yönetim Paneli")
    COMPANY_NAME = os.getenv("COMPANY_NAME", "Leancart Global")
    
    # Supported file types
    SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.md', '.docx']
    
    # UI Customization
    CHATBOT_THEME_COLOR = os.getenv("CHATBOT_THEME_COLOR", "#2563eb")
    CHATBOT_POSITION = os.getenv("CHATBOT_POSITION", "bottom-right")  # bottom-right, bottom-left
    ENABLE_VOICE_INPUT = os.getenv("ENABLE_VOICE_INPUT", "False").lower() == "true"
    
    @classmethod
    def validate_config(cls):
        """Validate essential configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Create directories if they don't exist
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.EMBEDDINGS_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)
        
        return True

# Default configuration instance
config = Config()
