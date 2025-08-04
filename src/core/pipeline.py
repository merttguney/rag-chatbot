"""
Pipeline module for end-to-end RAG processing
Handles: Question → Embed → Retrieve → Generate Answer
"""

from typing import List, Dict, Any, Optional
from loguru import logger

# OpenAI for chat completion
from openai import OpenAI

from ..services.embedding_service import Embedder
from ..retriever import Retriever
from .config import config


class RAGPipeline:
    """End-to-end RAG pipeline for question answering"""
    
    def __init__(self,
                 retriever: Retriever = None,
                 embedder: Embedder = None,
                 llm_model: str = None,
                 system_prompt: str = None):
        """
        Initialize RAG pipeline
        
        Args:
            retriever: Retriever instance
            embedder: Embedder instance
            llm_model: Language model for answer generation
            system_prompt: System prompt for the LLM
        """
        self.retriever = retriever or Retriever()
        self.embedder = embedder or Embedder()
        self.llm_model = llm_model or config.OPENAI_MODEL
        
        # Initialize OpenAI client
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Default system prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        logger.info(f"RAG Pipeline initialized with model: {self.llm_model}")
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for admin panel assistant"""
        return """Sen deneyimli ve yardımsever bir e-ticaret admin panel uzmanısın. Kullanıcılara admin panel konularında profesyonel destek sağlayacaksın.

Görevlerin:
• Admin panel işlemlerini basit ve anlaşılır şekilde açıklamak
• Sadece güvenilir kaynaklardan elde ettiğin bilgileri kullanmak  
• Açık, net ve Türkçe yanıtlar vermek
• Emin olmadığın durumlarda dürüstçe "Bu konuda kesin bilgim yok" demek

Uzmanlık Alanların:
• Ürün ve kategori yönetimi
• Sipariş süreçleri ve takibi
• Müşteri ilişkileri yönetimi
• Stok kontrol sistemleri
• Satış raporları ve analizler
• Sistem ayarları ve konfigürasyonlar

Yanıt Tarzın:
✓ Dostane ve profesyonel
✓ Emoji kullanarak daha samimi (📦, 📊, ✅, 💡, ⚡)
✓ Net ve pratik açıklamalar
✓ Pratik örnekler ve ipuçları
✓ Gerektiğinde adım adım rehberlik
✓ Sorunları çözmeye odaklı

Yanıtların doğal ve samimi olsun. Karmaşık işlemler için adım adım açıklama yap, basit sorular için kısa ve net cevap ver. Her yanıtta aynı formatı kullanmak zorunda değilsin - soruya göre en uygun şekilde yanıtla.

NOT: Kullanıcıya kaynak bilgisi, belge adları veya doğruluk skorları gösterme. Sadece yardımcı ve güvenilir cevaplar ver."""
    
    def process_question(self, 
                        question: str, 
                        top_k: int = None,
                        similarity_threshold: float = None,
                        max_context_length: int = None) -> Dict[str, Any]:
        """
        Process a question through the complete RAG pipeline
        
        Args:
            question: User question
            top_k: Number of top chunks to retrieve
            similarity_threshold: Minimum similarity for relevance
            max_context_length: Maximum context length for LLM
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Step 1: Get question embedding
            question_embedding = self.embedder.embed_text(question)
            
            # Step 2: Retrieve relevant chunks
            top_k = top_k or config.TOP_K_RESULTS
            similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
            
            relevant_chunks = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            if not relevant_chunks:
                return {
                    'answer': "Bu konuda belgelerden yeterli bilgi bulamadım.",
                    'sources': [],
                    'confidence': 0.0,
                    'metadata': {'chunks_found': 0, 'avg_similarity': 0.0}
                }
            
            # Step 3: Build context with smart truncation
            max_context_length = max_context_length or config.MAX_TOKENS_PER_REQUEST
            context, used_sources = self._build_optimized_context(
                relevant_chunks, 
                max_context_length
            )
            
            # Step 4: Generate answer
            answer = self._generate_answer(question, context)
            
            # Step 5: Calculate confidence and metadata
            avg_similarity = sum(chunk.get('similarity', 0) for chunk in relevant_chunks) / len(relevant_chunks)
            confidence = self._calculate_confidence(relevant_chunks, answer)
            
            result = {
                'answer': answer,
                'sources': used_sources,
                'confidence': confidence,
                'metadata': {
                    'chunks_found': len(relevant_chunks),
                    'chunks_used': len(used_sources),
                    'avg_similarity': avg_similarity,
                    'context_length': len(context)
                }
            }
            
            logger.info(f"Generated answer with confidence: {confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                'answer': f"Bir hata oluştu: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
    
    def _build_optimized_context(self, 
                               chunks: List[Dict[str, Any]], 
                               max_length: int) -> tuple[str, List[Dict[str, Any]]]:
        """
        Build optimized context from chunks with smart truncation
        
        Args:
            chunks: Retrieved chunks
            max_length: Maximum context length
            
        Returns:
            Tuple of (context_string, used_sources)
        """
        context_parts = []
        used_sources = []
        current_length = 0
        
        # Sort chunks by similarity (highest first)
        sorted_chunks = sorted(chunks, key=lambda x: x.get('similarity', 0), reverse=True)
        
        for chunk in sorted_chunks:
            chunk_text = chunk.get('text', '')
            chunk_metadata = chunk.get('metadata', {})
            
            # Estimate token count (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = len(chunk_text) // 4
            
            if current_length + estimated_tokens < max_length:
                source_info = {
                    'filename': chunk_metadata.get('source_file', 'unknown'),
                    'chunk_id': chunk_metadata.get('chunk_id', 0),
                    'similarity': chunk.get('similarity', 0),
                    'relevance_score': chunk.get('similarity', 0)
                }
                
                context_parts.append(chunk_text)
                used_sources.append(source_info)
                current_length += estimated_tokens
            else:
                # If we can't fit the whole chunk, try to fit a part of it
                remaining_chars = (max_length - current_length) * 4
                if remaining_chars > 200:  # Only if we can fit meaningful content
                    truncated_text = chunk_text[:remaining_chars-50] + "..."
                    source_info = {
                        'filename': chunk_metadata.get('source_file', 'unknown'),
                        'chunk_id': chunk_metadata.get('chunk_id', 0),
                        'similarity': chunk.get('similarity', 0),
                        'relevance_score': chunk.get('similarity', 0),
                        'truncated': True
                    }
                    context_parts.append(truncated_text)
                    used_sources.append(source_info)
                break
        
        context = "\n\n".join(context_parts)
        return context, used_sources
    
    def _generate_answer(self, question: str, context: str, temperature: float = None) -> str:
        """
        Generate answer using OpenAI LLM with optimized prompting
        
        Args:
            question: User question
            context: Retrieved and optimized context
            temperature: LLM temperature (optional, uses config default)
            
        Returns:
            Generated answer
        """
        try:
            # Build the prompt - clean and professional
            user_prompt = f"""Soru: {question}

Bilgi: {context}

Lütfen bu soruyu net ve anlaşılır şekilde yanıtla."""
            
            # Use provided temperature or config default
            temp = temperature if temperature is not None else config.TEMPERATURE
            
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                max_tokens=1000,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Post-process answer
            if not answer or len(answer.strip()) < 10:
                return "Bu konuda belgelerden yeterli bilgi bulamadım."
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Yanıt oluştururken bir hata oluştu: {str(e)}"
    
    def _calculate_confidence(self, chunks: List[Dict[str, Any]], answer: str) -> float:
        """
        Calculate confidence score for the generated answer
        
        Args:
            chunks: Retrieved chunks
            answer: Generated answer
            
        Returns:
            Confidence score between 0 and 1
        """
        if not chunks or not answer:
            return 0.0
        
        # Base confidence from average similarity
        avg_similarity = sum(chunk.get('similarity', 0) for chunk in chunks) / len(chunks)
        
        # Adjust based on number of relevant chunks
        chunk_factor = min(len(chunks) / config.TOP_K_RESULTS, 1.0)
        
        # Adjust based on answer length (very short answers are less confident)
        length_factor = min(len(answer) / 100, 1.0)
        
        # Check if answer indicates uncertainty
        uncertainty_phrases = [
            "yeterli bilgi bulamadım",
            "belirsiz",
            "emin değilim",
            "bilgi bulunamadı"
        ]
        
        uncertainty_penalty = 0.0
        for phrase in uncertainty_phrases:
            if phrase.lower() in answer.lower():
                uncertainty_penalty = 0.5
                break
        
        confidence = (avg_similarity * 0.6 + chunk_factor * 0.2 + length_factor * 0.2) - uncertainty_penalty
        return max(0.0, min(1.0, confidence))
    
    def ask_with_history(self, 
                        question: str, 
                        conversation_history: List[Dict[str, str]] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Ask question with conversation history for contextual understanding
        
        Args:
            question: Current question
            conversation_history: Previous Q&A pairs
            **kwargs: Additional parameters for process_question
            
        Returns:
            Answer with conversation context
        """
        # If we have history, enhance the question with context
        if conversation_history:
            context_aware_question = self._enhance_question_with_history(
                question, 
                conversation_history
            )
            result = self.process_question(context_aware_question, **kwargs)
        else:
            result = self.process_question(question, **kwargs)
        
        return result
    
    def _enhance_question_with_history(self, 
                                     question: str, 
                                     history: List[Dict[str, str]]) -> str:
        """
        Enhance current question with conversation history
        
        Args:
            question: Current question
            history: Conversation history
            
        Returns:
            Enhanced question with context
        """
        if not history:
            return question
        
        # Take last 2-3 exchanges for context
        recent_history = history[-3:] if len(history) > 3 else history
        
        context_parts = []
        for exchange in recent_history:
            if 'question' in exchange and 'answer' in exchange:
                context_parts.append(f"Önceki soru: {exchange['question']}")
                context_parts.append(f"Önceki cevap: {exchange['answer'][:200]}...")
        
        if context_parts:
            enhanced_question = f"""
Önceki konuşma bağlamı:
{chr(10).join(context_parts)}

Şu anki soru: {question}
"""
            return enhanced_question
        
        return question
    
    def answer_question(self, 
                       question: str,
                       top_k: int = None,
                       include_sources: bool = False,  # Changed default to False
                       temperature: float = 0.1) -> Dict[str, Any]:
        """
        Answer a question using RAG pipeline
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            include_sources: Whether to include source information
            temperature: LLM temperature for response generation
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not question or not question.strip():
            return {
                'answer': 'Lütfen geçerli bir soru sorun.',
                'sources': [],
                'question': question,
                'retrieved_docs': []
            }
        
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Step 1: Retrieve relevant documents
            relevant_docs = self.retriever.retrieve(
                query=question,
                top_k=top_k or config.TOP_K_RESULTS
            )
            
            if not relevant_docs:
                return {
                    'answer': 'Bu soruya cevap verebilmek için yeterli bilgi bulunamadı.',
                    'sources': [],
                    'question': question,
                    'retrieved_docs': []
                }
            
            # Step 2: Prepare context from retrieved documents
            context = self._prepare_context(relevant_docs)
            
            # Step 3: Generate answer using LLM
            answer = self._generate_answer(question, context, temperature)
            
            # Step 4: Prepare response
            response = {
                'answer': answer,
                'question': question,
                'retrieved_docs': len(relevant_docs)
            }
            
            if include_sources:
                response['sources'] = self._extract_sources(relevant_docs)
            
            logger.info("Successfully generated answer")
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return {
                'answer': f'Üzgünüm, sorunuzu yanıtlarken bir hata oluştu: {str(e)}',
                'sources': [],
                'question': question,
                'retrieved_docs': 0
            }
    
    def _prepare_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Prepare context from retrieved documents
        
        Args:
            relevant_docs: List of retrieved document chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for doc in relevant_docs:
            text = doc.get('text', '').strip()
            if text:  # Only add non-empty text
                context_parts.append(text)
        
        return '\n\n'.join(context_parts)
    
    def _extract_sources(self, relevant_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from retrieved documents"""
        sources = []
        seen_sources = set()
        
        for doc in relevant_docs:
            metadata = doc.get('metadata', {})
            source_file = metadata.get('source_file', 'Bilinmeyen')
            
            if source_file not in seen_sources:
                source_info = {
                    'filename': source_file,
                    'file_type': metadata.get('source_extension', ''),
                    'relevance_score': doc.get('score', 0.0)
                }
                sources.append(source_info)
                seen_sources.add(source_file)
        
        return sources
    
    def chat(self, 
             question: str,
             conversation_history: List[Dict[str, str]] = None,
             top_k: int = None) -> Dict[str, Any]:
        """
        Handle conversational question answering
        
        Args:
            question: Current question
            conversation_history: Previous Q&A pairs
            top_k: Number of documents to retrieve
            
        Returns:
            Answer with conversation context
        """
        # Enhance query with conversation context if available
        enhanced_query = question
        
        if conversation_history:
            # Add recent context to improve retrieval
            recent_context = "\n".join([
                f"Q: {qa['question']}\nA: {qa['answer']}"
                for qa in conversation_history[-2:]  # Last 2 Q&A pairs
            ])
            enhanced_query = f"{recent_context}\n\nCurrent question: {question}"
        
        # Get answer using enhanced query for retrieval but original question for generation
        relevant_docs = self.retriever.retrieve(
            query=enhanced_query,
            top_k=top_k or config.TOP_K_RESULTS
        )
        
        if not relevant_docs:
            return {
                'answer': 'Bu soruya cevap verebilmek için yeterli bilgi bulunamadı.',
                'sources': [],
                'question': question,
                'retrieved_docs': 0
            }
        
        # Prepare context
        context = self._prepare_context(relevant_docs)
        
        # Prepare messages with conversation history
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history
        if conversation_history:
            for qa in conversation_history[-3:]:  # Last 3 Q&A pairs
                messages.append({"role": "user", "content": qa['question']})
                messages.append({"role": "assistant", "content": qa['answer']})
        
        # Add current question with context
        messages.append({"role": "user", "content": f"""Soru: {question}

Bilgi: {context}

Lütfen bu soruyu önceki konuşma bağlamını da dikkate alarak yanıtla."""})
        
        try:
            # Generate response
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                'answer': answer,
                'question': question,
                'sources': self._extract_sources(relevant_docs),
                'retrieved_docs': len(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return {
                'answer': f'Üzgünüm, sorunuzu yanıtlarken bir hata oluştu: {str(e)}',
                'sources': [],
                'question': question,
                'retrieved_docs': 0
            }
    
    def update_system_prompt(self, new_prompt: str):
        """Update system prompt"""
        self.system_prompt = new_prompt
        logger.info("System prompt updated")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        vector_stats = self.retriever.vector_store.get_stats()
        
        return {
            'llm_model': self.llm_model,
            'embedding_model': self.embedder.embedding_model,
            'vector_store_stats': vector_stats,
            'retriever_top_k': self.retriever.top_k,
            'similarity_threshold': self.retriever.similarity_threshold
        }
