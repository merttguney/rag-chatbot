"""
Flask web arayüzü
"""

from flask import Flask, render_template_string, request, jsonify, send_from_directory
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.rag_runner import RAGRunner
from src.core.config import config
from loguru import logger

app = Flask(__name__)

# Global RAG runner instance
rag_runner = None

def initialize_rag():
    """Initialize RAG runner"""
    global rag_runner
    if rag_runner is None:
        try:
            rag_runner = RAGRunner(data_dir=config.DATA_DIR)
            # Try to set up if not already done
            if not rag_runner.is_initialized:
                logger.info("⚙️  RAG sistemi kuruluyor...")
                rag_runner.setup()
            logger.info("✅ RAG sistemi başarıyla yüklendi")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            rag_runner = None
    return rag_runner

# HTML template - Modern Dark Theme
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LeanCart Global Chatbot</title>
    <link rel="icon" type="image/png" href="/favicon.png">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0a1f18 0%, #1a2a24 100%);
            color: #ffffff;
            height: 100vh;
            overflow: hidden;
        }
        
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            max-width: 380px;
            margin: 0 auto;
            border-radius: 12px;
            overflow: hidden;
            background: #1a2a24;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
            position: relative;
        }
        
        /* Demo sayfası için özel düzenlemeler */
        body.demo-mode .chat-container {
            height: 100%;
            box-shadow: none;
            border-radius: 0;
        }
        
        /* Demo modunda iframe içindeki refresh butonunu gizle */
        body.demo-mode .refresh-btn {
            display: none !important;
        }
        
        /* Demo modunda typing indicator düzeltmesi */
        body.demo-mode .typing-indicator {
            position: static;
            margin: 8px 0;
            margin-left: 0;
            align-self: flex-start;
        }
        
        body.demo-mode .typing-indicator::before {
            position: absolute;
            top: -16px;
            left: 4px;
            background: #0f221a;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #155342 0%, #0f3d2e 100%);
            padding: 24px 20px;
            text-align: center;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .chat-header h1 {
            font-size: 16px;
            font-weight: 600;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 8px;
            flex: 1;
            justify-content: center;
        }
        
        .header-controls {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .demo-refresh-btn {
            background: rgba(255,255,255,0.1);
            border: none;
            border-radius: 6px;
            width: 32px;
            height: 32px;
            color: white;
            cursor: pointer;
            display: none;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
            font-size: 14px;
        }
        
        .demo-refresh-btn:hover {
            background: rgba(255,255,255,0.2);
            transform: scale(1.05);
        }
        
        .demo-refresh-btn::before {
            content: "🔄";
        }
        
        .demo-close-btn {
            background: rgba(255,255,255,0.1);
            border: none;
            border-radius: 6px;
            width: 32px;
            height: 32px;
            color: white;
            cursor: pointer;
            display: none;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .demo-close-btn:hover {
            background: rgba(255,255,255,0.2);
            transform: scale(1.05);
        }
        
        /* Demo modunda butonları göster */
        body.demo-mode .demo-refresh-btn,
        body.demo-mode .demo-close-btn {
            display: flex;
        }
        
        /* Demo modunda header padding artır */
        body.demo-mode .chat-header {
            padding: 24px 20px;
            min-height: 64px;
        }
        
        .chat-header h1 {
            font-size: 16px;
            font-weight: 600;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 8px;
            flex: 1;
            justify-content: center;
        }
        
        .chat-header .logo {
            width: 24px;
            height: 24px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        
        .chat-header .logo img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        
        .refresh-btn {
            background: rgba(255,255,255,0.1);
            border: none;
            border-radius: 6px;
            width: 32px;
            height: 32px;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }
        
        .refresh-btn:hover {
            background: rgba(255,255,255,0.2);
        }
        
        .refresh-btn::before {
            content: "🔄";
            font-size: 14px;
        }
        
        .chat-messages {
            flex: 1;
            padding: 16px;
            overflow-y: auto;
            background: #0f221a;
            display: flex;
            flex-direction: column;
            gap: 12px;
            position: relative;
        }
        
        /* Yazıyor göstergesi için ek alan */
        .chat-messages::after {
            content: '';
            height: 20px;
            flex-shrink: 0;
        }
        
        /* Quick Suggestions */
        .quick-suggestions {
            padding: 12px 16px;
            background: #152520;
            border-top: 1px solid rgba(124, 160, 133, 0.2);
        }
        
        .suggestions-title {
            font-size: 11px;
            color: #7ca085;
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        .suggestion-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        
        .suggestion-btn {
            background: rgba(21, 83, 66, 0.15);
            border: 1px solid rgba(124, 160, 133, 0.2);
            color: #9bc3a3;
            padding: 8px 14px;
            border-radius: 20px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
            font-weight: 500;
        }
        
        .suggestion-btn:hover {
            background: rgba(21, 83, 66, 0.3);
            border-color: rgba(124, 160, 133, 0.4);
            color: #b4d4ba;
            transform: translateY(-1px);
        }
        
        /* Typing Indicator */
        .typing-indicator {
            display: none;
            padding: 12px 16px;
            background: linear-gradient(135deg, #1a332a 0%, #2a4a3a 100%);
            border-radius: 18px;
            align-self: flex-start;
            max-width: 200px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            border: 1px solid rgba(124, 160, 133, 0.2);
            position: relative;
            margin: 8px 0;
            margin-left: 0;
            animation: slideInUp 0.3s ease-out;
            order: 999; /* Her zaman en altta görünsün */
        }
        
        .typing-indicator::before {
            content: 'Yazıyor...';
            font-size: 10px;
            color: #7ca085;
            opacity: 0.8;
            position: absolute;
            top: -16px;
            left: 4px;
            font-weight: 500;
            letter-spacing: 0.2px;
            background: #0f221a;
            padding: 2px 6px;
            border-radius: 4px;
        }
        
        .typing-dots {
            display: flex;
            gap: 6px;
            align-items: center;
            justify-content: center;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            background: linear-gradient(45deg, #7ca085, #a8d4b0);
            border-radius: 50%;
            animation: typingAnimation 1.6s infinite;
            box-shadow: 0 0 6px rgba(124, 160, 133, 0.3);
        }
        
        .typing-dot:nth-child(2) { animation-delay: 0.3s; }
        .typing-dot:nth-child(3) { animation-delay: 0.6s; }
        
        @keyframes typingAnimation {
            0%, 70%, 100% { 
                opacity: 0.4; 
                transform: scale(0.7) translateY(0px);
            }
            35% { 
                opacity: 1; 
                transform: scale(1.1) translateY(-4px);
                box-shadow: 0 0 12px rgba(124, 160, 133, 0.6);
            }
        }
        
        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideOutDown {
            from {
                opacity: 1;
                transform: translateY(0);
            }
            to {
                opacity: 0;
                transform: translateY(20px);
            }
        }
        
        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 20px;
            font-size: 14px;
            line-height: 1.5;
            word-wrap: break-word;
            animation: fadeIn 0.4s ease-out;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .message strong {
            font-weight: bold;
            color: #4ade80;
        }
        
        .message.bot strong {
            color: #60d394;
        }
        
        .message.user strong {
            color: #ffffff;
            font-weight: 600;
        }
        
        .message.bot {
            background: linear-gradient(135deg, #1a3428 0%, #1e3a2c 100%);
            align-self: flex-start;
            border-bottom-left-radius: 6px;
            border: 1px solid rgba(124, 160, 133, 0.2);
            color: #e8f5e8;
        }
        
        .message.user {
            background: linear-gradient(135deg, #155342 0%, #0f3d2e 100%);
            align-self: flex-end;
            border-bottom-right-radius: 6px;
            color: white;
            border: 1px solid rgba(21, 83, 66, 0.3);
        }
        
        .chat-input-container {
            padding: 16px;
            background: #1a2a24;
            border-top: 1px solid rgba(21, 83, 66, 0.2);
        }
        
        .input-wrapper {
            display: flex;
            gap: 8px;
            align-items: flex-end;
        }
        
        .chat-input {
            flex: 1;
            background: #0f221a;
            border: 2px solid rgba(124, 160, 133, 0.2);
            border-radius: 22px;
            padding: 12px 18px;
            color: #ffffff;
            font-size: 14px;
            resize: none;
            min-height: 40px;
            max-height: 100px;
            outline: none;
            transition: all 0.2s ease;
            font-family: inherit;
        }
        
        .chat-input:focus {
            border-color: #7ca085;
            box-shadow: 0 0 0 3px rgba(124, 160, 133, 0.1);
        }
        
        .chat-input::placeholder {
            color: #666;
        }
        
        .send-button {
            background: linear-gradient(135deg, #155342 0%, #0f3d2e 100%);
            border: none;
            border-radius: 50%;
            width: 44px;
            height: 44px;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
            outline: none;
            box-shadow: 0 2px 8px rgba(21, 83, 66, 0.3);
        }
        
        .send-button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(21, 83, 66, 0.5);
        }
        
        .send-button:active {
            transform: scale(0.95);
        }
        
        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            box-shadow: 0 2px 8px rgba(21, 83, 66, 0.2);
        }
        
        .send-button::before {
            content: "↗";
            font-size: 16px;
            font-weight: bold;
        }
        
        .footer {
            padding: 10px;
            text-align: center;
            background: #1a2a24;
            font-size: 10px;
            color: #555;
            border-top: 1px solid rgba(21, 83, 66, 0.2);
        }
        
        .footer a {
            color: #155342;
            text-decoration: none;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Scrollbar styling */
        .chat-messages::-webkit-scrollbar {
            width: 4px;
        }
        
        .chat-messages::-webkit-scrollbar-track {
            background: #0f221a;
        }
        
        .chat-messages::-webkit-scrollbar-thumb {
            background: rgba(21, 83, 66, 0.5);
            border-radius: 2px;
        }
        
        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: rgba(21, 83, 66, 0.8);
        }
        
        /* Mobile responsive */
        @media (max-width: 480px) {
            .chat-container {
                max-width: 100%;
                height: 100vh;
                border-radius: 0;
            }
        }
        
        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message.bot {
            animation: fadeInUp 0.3s ease;
        }
        
        /* Animations */
        @keyframes messageSlideIn {
            from {
                opacity: 0;
                transform: translateY(12px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        
        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideOutDown {
            from {
                opacity: 1;
                transform: translateY(0);
            }
            to {
                opacity: 0;
                transform: translateY(20px);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(16px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Scrollbar Styling */
        .chat-messages::-webkit-scrollbar {
            width: 4px;
        }
        
        .chat-messages::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .chat-messages::-webkit-scrollbar-thumb {
            background: var(--border-default);
            border-radius: var(--radius-full);
        }
        
        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: var(--border-strong);
        }
        
        /* Mobile Responsive */
        @media (max-width: 480px) {
            .chat-container {
                max-width: 100%;
                height: 100vh;
                border-radius: 0;
            }
            
            .chat-header {
                padding: 20px var(--space-5);
            }
            
            .chat-messages {
                padding: var(--space-5) var(--space-5) var(--space-4);
            }
            
            .chat-input-container {
                padding: var(--space-4) var(--space-5) var(--space-5);
            }
            
            .quick-suggestions {
                padding: var(--space-4) var(--space-5);
            }
            
            .message {
                max-width: 90%;
            }
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            :root {
                --background-primary: #0f172a;
                --background-secondary: #1e293b;
                --background-tertiary: #334155;
                --surface-primary: #1e293b;
                --surface-secondary: #334155;
                --surface-elevated: #475569;
                
                --text-primary: #f8fafc;
                --text-secondary: #cbd5e1;
                --text-tertiary: #94a3b8;
                --text-muted: #64748b;
                --text-inverse: #0f172a;
                
                --border-subtle: #334155;
                --border-default: #475569;
                --border-strong: #64748b;
            }
        }
        
        /* Focus management and accessibility */
        .suggestion-btn:focus,
        .refresh-btn:focus,
        .send-button:focus {
            outline: 2px solid var(--primary-500);
            outline-offset: 2px;
        }
        
        .chat-input:focus {
            outline: none;
        }
        
        /* Loading state */
        .loading-shimmer {
            background: linear-gradient(90deg, var(--surface-secondary) 25%, var(--background-tertiary) 50%, var(--surface-secondary) 75%);
            background-size: 200% 100%;
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% {
                background-position: -200% 0;
            }
            100% {
                background-position: 200% 0;
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>
                <div class="logo">
                    <img src="/favicon.png" alt="Logo">
                </div>
                LeanCart Global Chatbot
            </h1>
            <div class="header-controls">
                <button class="demo-refresh-btn" onclick="parent.refreshChatbot && parent.refreshChatbot()" title="Sohbeti Yenile"></button>
                <button class="demo-close-btn" onclick="parent.toggleChatbot && parent.toggleChatbot()" title="Kapat">×</button>
                <button class="refresh-btn" onclick="refreshChat()"></button>
            </div>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                Merhaba! 👋 LeanCart Global Admin Panel Asistanı'na hoş geldiniz.<br><br>
                Size admin panel konularında yardımcı olmaktan mutluluk duyarım. Aşağıdaki popüler sorulardan birini seçebilir veya kendi sorunuzu yazabilirsiniz.
            </div>
            
            <div class="typing-indicator" id="typingIndicator">
                <div class="typing-dots">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        </div>
        
        <div class="quick-suggestions" id="quickSuggestions">
            <div class="suggestions-title">💡 Popüler sorular</div>
            <div class="suggestion-buttons">
                <button class="suggestion-btn" onclick="askSuggestion('Ürün nasıl eklenir?')">📦 Ürün ekleme</button>
                <button class="suggestion-btn" onclick="askSuggestion('Sipariş durumu nasıl değiştirilir?')">📋 Sipariş yönetimi</button>
                <button class="suggestion-btn" onclick="askSuggestion('Stok takibi nasıl yapılır?')">📊 Stok kontrol</button>
                <button class="suggestion-btn" onclick="askSuggestion('Müşteri bilgileri nasıl güncellenir?')">👤 Müşteri yönetimi</button>
                <button class="suggestion-btn" onclick="askSuggestion('Satış raporları nerede görülür?')">📈 Raporlar</button>
                <button class="suggestion-btn" onclick="askSuggestion('Kategori düzenleme nasıl yapılır?')">🏷️ Kategori yönetimi</button>
            </div>
        </div>
        
        <div class="chat-input-container">
            <div class="input-wrapper">
                <textarea 
                    id="messageInput" 
                    class="chat-input" 
                    placeholder="Admin panel hakkında soru sorabilirsiniz..."
                    rows="1"
                    onkeydown="handleEnter(event)"
                    oninput="autoResize(this)"
                ></textarea>
                <button onclick="sendMessage()" id="sendBtn" class="send-button"></button>
            </div>
        </div>
        
        <div class="footer">
            LeanCart Global E-Commerce Solutions
        </div>
    </div>
                    rows="1"
                    onkeydown="handleEnter(event)"
                    oninput="autoResize(this)"
                    aria-label="Mesaj yazın"
                    role="textbox"
                    aria-multiline="true"
                ></textarea>
                <button 
                    onclick="sendMessage()" 
                    id="sendBtn" 
                    class="send-button"
                    aria-label="Mesajı gönder"
                    type="button"
                ></button>
            </div>
        </div>
        
        <div class="footer">
            <a href="#" onclick="return false;">LeanCart Global</a> E-Commerce Solutions
        </div>
    </div>

    <script>
        let isTyping = false;
        
        function autoResize(textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 100) + 'px';
        }
        
        function handleEnter(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        function showTypingIndicator() {
            const indicator = document.getElementById('typingIndicator');
            indicator.style.display = 'flex';
            
            setTimeout(() => {
                scrollToBottom();
            }, 100);
        }
        
        function hideTypingIndicator() {
            const indicator = document.getElementById('typingIndicator');
            
            indicator.style.animation = 'slideOutDown 0.3s ease-in forwards';
            
            setTimeout(() => {
                indicator.style.display = 'none';
                indicator.style.animation = '';
            }, 300);
        }
        
        function scrollToBottom() {
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function askSuggestion(question) {
            const input = document.getElementById('messageInput');
            input.value = question;
            input.focus();
            
            const event = new Event('input', { bubbles: true });
            input.dispatchEvent(event);
            
            sendMessage();
        }
        
        function refreshChat() {
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.innerHTML = `
                <div class="message bot">
                    Merhaba! 👋 Size nasıl yardımcı olabilirim? Admin panel işlemleri hakkında sorularınızı sorabilirsiniz.
                </div>
                
                <div class="typing-indicator" id="typingIndicator">
                    <div class="typing-dots">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                    </div>
                </div>
            `;
            
            const quickSuggestions = document.getElementById('quickSuggestions');
            quickSuggestions.style.display = 'block';
            quickSuggestions.style.opacity = '0';
            quickSuggestions.style.transform = 'translateY(10px)';
            
            setTimeout(() => {
                quickSuggestions.style.transition = 'all 0.4s ease-out';
                quickSuggestions.style.opacity = '1';
                quickSuggestions.style.transform = 'translateY(0)';
            }, 100);
        }
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message || isTyping) return;
            
            addMessage(message, 'user');
            
            const quickSuggestions = document.getElementById('quickSuggestions');
            if (quickSuggestions.style.display !== 'none') {
                quickSuggestions.style.transition = 'all 0.3s ease-out';
                quickSuggestions.style.opacity = '0';
                quickSuggestions.style.transform = 'translateY(-10px)';
                setTimeout(() => {
                    quickSuggestions.style.display = 'none';
                }, 300);
            }
            
            input.value = '';
            input.style.height = 'auto';
            const sendBtn = document.getElementById('sendBtn');
            sendBtn.disabled = true;
            isTyping = true;
            
            showTypingIndicator();
            
            fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({question: message})
            })
            .then(response => response.json())
            .then(data => {
                hideTypingIndicator();
                if (data.success) {
                    const delay = Math.min(1000, Math.max(300, data.answer.length * 15));
                    setTimeout(() => {
                        addMessage(data.answer, 'bot', true);
                    }, delay);
                } else {
                    const errorMsg = getErrorMessage(data.error);
                    addMessage(errorMsg, 'bot', true);
                }
            })
            .catch(error => {
                hideTypingIndicator();
                console.error('Error:', error);
                const errorMsg = '😔 Özür dilerim, şu anda bir teknik sorun yaşıyorum. Lütfen sorunuzu farklı şekilde sormayı deneyin veya biraz sonra tekrar deneyin.';
                addMessage(errorMsg, 'bot', true);
            })
            .finally(() => {
                setTimeout(() => {
                    sendBtn.disabled = false;
                    isTyping = false;
                }, 1000);
            });
        }
        
        function addMessage(text, sender, animate = false) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const processedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            
            chatMessages.appendChild(messageDiv);
            
            if (animate && sender === 'bot') {
                messageDiv.innerHTML = '';
                
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = processedText;
                const plainText = tempDiv.textContent || tempDiv.innerText || '';
                
                let charIndex = 0;
                const delay = 25;
                
                function typeWriter() {
                    if (charIndex <= plainText.length) {
                        const currentChar = plainText.substring(0, charIndex);
                        const formattedText = currentChar.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                        messageDiv.innerHTML = formattedText;
                        
                        charIndex++;
                        setTimeout(typeWriter, delay);
                    }
                }
                
                setTimeout(typeWriter, 150);
            } else {
                messageDiv.innerHTML = processedText;
            }
            
            scrollToBottom();
        }
        
        function getErrorMessage(error) {
            const errorMessages = [
                '🤔 Bu konuda biraz daha bilgiye ihtiyacım var. Sorunuzu daha detaylı sorabilir misiniz?',
                '💭 Hmm, bu soruya daha iyi cevap verebilmek için farklı bir şekilde sormayı dener misiniz?',
                '🔍 Bu konuda elimdeki bilgiler yeterli değil. Başka bir şekilde formülize edebilir misiniz?',
            ];
            return errorMessages[Math.floor(Math.random() * errorMessages.length)];
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            const messageInput = document.getElementById('messageInput');
            
            messageInput.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = this.scrollHeight + 'px';
            });
            
            messageInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter') {
                    if (e.shiftKey) {
                        return;
                    } else {
                        e.preventDefault();
                        sendMessage();
                    }
                }
            });
            
            messageInput.focus();
        });
        
        window.onload = function() {
            setTimeout(() => {
                const welcomeMsg = document.querySelector('.message.bot');
                if (welcomeMsg) {
                    welcomeMsg.style.opacity = '0';
                    welcomeMsg.style.animation = 'fadeInUp 0.6s ease forwards';
                }
            }, 200);
        }
    </script>
</body>
</html>
"""

@app.route('/')
@app.route('/iframe')
def index():
    """Ana sayfa"""
    # Check if this is being loaded in iframe (demo mode)
    is_iframe = 'iframe' in request.endpoint or request.args.get('demo') == 'true'
    
    # Add demo-mode class if in iframe
    template = HTML_TEMPLATE
    if is_iframe:
        template = template.replace('<body>', '<body class="demo-mode">')
    
    return render_template_string(template)

@app.route('/demo')
def demo():
    """Demo sayfası - Admin panel entegrasyonu gösterimi"""
    with open('demo.html', 'r', encoding='utf-8') as f:
        demo_content = f.read()
    return demo_content

@app.route('/favicon.ico')
@app.route('/favicon.png')
def favicon():
    """Favicon ve logo servisi"""
    return send_from_directory('.', 'favicon.png')

@app.route('/ask', methods=['POST'])
def ask():
    """Soru cevaplama endpoint'i"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'success': False, 'error': 'Boş soru gönderildi'})
        
        # Initialize RAG if needed
        runner = initialize_rag()
        if not runner:
            return jsonify({'success': False, 'error': 'Sistem henüz hazır değil'})
        
        # Get answer
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 KULLANICI SORUSU: {question}")
            logger.info(f"{'='*60}")
            
            result = runner.ask_question(question)
            if isinstance(result, dict):
                answer = result.get('answer', 'Cevap alınamadı')
            else:
                answer = str(result)
            
            logger.info(f"\n✅ CEVAP:")
            logger.info(f"{'-'*40}")
            logger.info(f"{answer}")
            logger.info(f"{'-'*40}\n")
            
            # Extract confidence and metadata if available
            confidence = 0.0
            metadata = {}
            sources_count = 0
            
            if isinstance(result, dict):
                confidence = result.get('confidence', 0.0)
                metadata = result.get('metadata', {})
                sources_count = len(result.get('sources', []))
                answer = result.get('answer', str(result))
            
            return jsonify({
                'success': True,
                'answer': answer,
                'question': question,
                'confidence': confidence,
                'sources_count': sources_count,
                'metadata': {
                    'chunks_found': metadata.get('chunks_found', 0),
                    'response_quality': 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low'
                }
            })
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return jsonify({
                'success': False, 
                'error': f'Cevap oluştururken hata: {str(e)}'
            })
            
    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stats')
def stats():
    """Sistem istatistikleri endpoint'i"""
    try:
        runner = initialize_rag()
        if not runner:
            return jsonify({
                'success': False, 
                'data': {'is_ready': False, 'documents': 0, 'chunks': 0, 'embeddings': 0}
            })
        
        # Get system stats
        try:
            system_stats = runner.get_system_stats()
            return jsonify({
                'success': True,
                'data': {
                    'is_ready': rag_runner.is_initialized,
                    'documents': system_stats.get('total_documents', 0),
                    'chunks': system_stats.get('total_chunks', 0),
                    'embeddings': system_stats.get('total_embeddings', 0)
                }
            })
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return jsonify({
                'success': False,
                'data': {'is_ready': False, 'documents': 0, 'chunks': 0, 'embeddings': 0}
            })
            
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 LeanCart Global Admin Panel Asistanı")
    print("="*60)
    print(f"📍 Web Arayüzü: http://localhost:8086")
    print(f"📊 Sistem durumu: Başlatılıyor...")
    print("="*60 + "\n")
    
    # Initialize RAG system
    initialize_rag()
    
    print("✅ Sistem hazır! Web arayüzüne bağlanabilirsiniz.\n")
    
    app.run(debug=True, host='0.0.0.0', port=8086)
