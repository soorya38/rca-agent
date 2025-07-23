#!/bin/bash

# Helios Setup Script
# This script initializes the Helios platform and sets up the default Ollama model

set -e

echo "🌟 Helios Setup Script"
echo "======================"

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "✅ .env file created. You can modify it to customize your configuration."
else
    echo "📝 .env file already exists."
fi

# Build and start services
echo "🚀 Building and starting Helios services..."
docker-compose up -d

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama service to be ready..."
timeout 300 bash -c '
    until curl -s http://localhost:11434/api/version > /dev/null; do
        echo "Waiting for Ollama..."
        sleep 5
    done
'

if [ $? -eq 0 ]; then
    echo "✅ Ollama service is ready!"
else
    echo "❌ Timeout waiting for Ollama service. Please check the logs."
    exit 1
fi

# Pull the default model
MODEL=${OLLAMA_MODEL:-llama3.2:latest}
echo "🤖 Pulling Ollama model: $MODEL"
echo "This may take several minutes for the first time..."

docker exec helios-ollama ollama pull $MODEL

if [ $? -eq 0 ]; then
    echo "✅ Model $MODEL pulled successfully!"
else
    echo "❌ Failed to pull model $MODEL. Please check your internet connection."
    exit 1
fi

# Check service health
echo "🏥 Checking service health..."
sleep 10

# Check API Gateway
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API Gateway is healthy"
else
    echo "⚠️  API Gateway health check failed"
fi

# Check AI Core
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ AI Core is healthy"
else
    echo "⚠️  AI Core health check failed"
fi

# Check Frontend
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Frontend is accessible"
else
    echo "⚠️  Frontend accessibility check failed"
fi

echo ""
echo "🎉 Helios setup complete!"
echo ""
echo "Access your Helios instance at:"
echo "  Frontend: http://localhost:3000"
echo "  API Gateway: http://localhost:8000"
echo "  AI Core: http://localhost:8001"
echo ""
echo "To stop Helios: docker-compose down"
echo "To view logs: docker-compose logs -f"
echo ""
echo "Happy analyzing! 🔍" 