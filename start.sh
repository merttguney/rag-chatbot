#!/bin/bash

# RAG Chatbot Startup Script
# This script helps start the RAG chatbot with proper environment setup

set -e

echo "🚀 Starting RAG Chatbot..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "⚙️  Activating virtual environment..."
source .venv/bin/activate

# Check if requirements are installed
if [ ! -f ".venv/installed" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    touch .venv/installed
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Creating .env from template..."
    cp .env.example .env
    echo "🔑 Please edit .env file and add your OpenAI API key!"
    echo "📝 Then run this script again."
    exit 1
fi

# Check if OpenAI API key is set
if grep -q "your_openai_api_key_here" .env; then
    echo "⚠️  Please add your OpenAI API key to .env file!"
    echo "📝 Edit .env and replace 'your_openai_api_key_here' with your actual API key."
    exit 1
fi

echo "✅ Starting web application..."
echo "🌐 Access the chatbot at: http://localhost:8086"
echo "⏹️  Press Ctrl+C to stop"

python web_app.py
