import os
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import asyncpg
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class ChatMessage(BaseModel):
    logs: Optional[str] = None
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    model_used: str
    timestamp: datetime

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]

class HealthStatus(BaseModel):
    status: str
    services: Dict[str, str]

# Database models
class DatabaseManager:
    """Manages PostgreSQL connections and operations"""
    
    def __init__(self):
        self.pool = None
        self.db_url = self._build_db_url()
    
    def _build_db_url(self) -> str:
        """Build database connection URL from environment variables"""
        host = os.getenv("POSTGRES_HOST", "postgres")
        port = os.getenv("POSTGRES_PORT", "5432")
        user = os.getenv("POSTGRES_USER", "helios")
        password = os.getenv("POSTGRES_PASSWORD", "helios_password")
        database = os.getenv("POSTGRES_DB", "helios")
        
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    async def initialize(self):
        """Initialize database connection pool and create tables"""
        try:
            self.pool = await asyncpg.create_pool(self.db_url, min_size=5, max_size=20)
            await self._create_tables()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
    
    async def _create_tables(self):
        """Create necessary database tables"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    sender VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    model_used VARCHAR(100),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    logs TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);
            """)
    
    async def save_message(self, session_id: str, sender: str, content: str, 
                          model_used: str = None, logs: str = None):
        """Save a message to the conversation history"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO conversations (session_id, sender, content, model_used, logs)
                VALUES ($1, $2, $3, $4, $5)
            """, session_id, sender, content, model_used, logs)
    
    async def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a session"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT sender, content, model_used, timestamp, logs
                FROM conversations
                WHERE session_id = $1
                ORDER BY timestamp ASC
                LIMIT $2
            """, session_id, limit)
            
            return [
                {
                    "sender": row["sender"],
                    "content": row["content"],
                    "model_used": row["model_used"],
                    "timestamp": row["timestamp"],
                    "logs": row["logs"]
                }
                for row in rows
            ]

class HeliosAPIGateway:
    """Main API Gateway service that orchestrates requests"""
    
    def __init__(self):
        self.ai_core_url = os.getenv("AI_CORE_URL", "http://helios-ai-core:8001")
        self.client = httpx.AsyncClient(timeout=300.0)
        self.db = DatabaseManager()
    
    async def initialize(self):
        """Initialize the gateway and its dependencies"""
        await self.db.initialize()
    
    async def shutdown(self):
        """Clean shutdown of the gateway"""
        await self.client.aclose()
        await self.db.close()
    
    async def check_service_health(self, url: str, service_name: str) -> str:
        """Check health of a service"""
        try:
            response = await self.client.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                return "healthy"
            else:
                return f"unhealthy (HTTP {response.status_code})"
        except Exception as e:
            return f"unreachable ({str(e)})"
    
    async def process_chat_message(self, message: ChatMessage) -> ChatResponse:
        """Process a chat message and return the AI response"""
        try:
            # Generate session ID if not provided
            session_id = message.session_id or str(uuid.uuid4())
            
            # Save user message to database
            await self.db.save_message(
                session_id=session_id,
                sender="user",
                content=message.question,
                logs=message.logs
            )
            
            # Prepare request for AI Core
            ai_request = {
                "logs": message.logs,
                "question": message.question,
                "session_id": session_id
            }
            
            # Call AI Core service
            logger.info(f"Sending request to AI Core for session {session_id}")
            response = await self.client.post(
                f"{self.ai_core_url}/analyze",
                json=ai_request
            )
            response.raise_for_status()
            
            ai_response = response.json()
            
            # Save AI response to database
            await self.db.save_message(
                session_id=session_id,
                sender="assistant",
                content=ai_response["response"],
                model_used=ai_response["model_used"]
            )
            
            return ChatResponse(
                response=ai_response["response"],
                session_id=session_id,
                model_used=ai_response["model_used"],
                timestamp=datetime.now()
            )
            
        except httpx.RequestError as e:
            logger.error(f"Request error to AI Core: {e}")
            raise HTTPException(status_code=503, detail=f"AI service unavailable: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from AI Core: {e}")
            raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in chat processing: {e}")
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    async def get_conversation_history(self, session_id: str) -> ConversationHistory:
        """Retrieve conversation history for a session"""
        try:
            messages = await self.db.get_conversation_history(session_id)
            return ConversationHistory(session_id=session_id, messages=messages)
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Initialize gateway
gateway = HeliosAPIGateway()

# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await gateway.initialize()
    yield
    # Shutdown
    await gateway.shutdown()

app = FastAPI(
    title="Helios API Gateway",
    description="Central orchestration service for the Helios observability platform",
    lifespan=lifespan
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Comprehensive health check of all services"""
    ai_core_health = await gateway.check_service_health(gateway.ai_core_url, "ai-core")
    
    return HealthStatus(
        status="healthy" if ai_core_health == "healthy" else "degraded",
        services={
            "api-gateway": "healthy",
            "ai-core": ai_core_health,
            "database": "healthy"  # Simplified check - could be enhanced
        }
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Main chat endpoint for user interactions"""
    return await gateway.process_chat_message(message)

@app.get("/conversation/{session_id}", response_model=ConversationHistory)
async def get_conversation(session_id: str):
    """Retrieve conversation history for a specific session"""
    return await gateway.get_conversation_history(session_id)

@app.get("/sessions")
async def list_sessions():
    """List all conversation sessions"""
    # This could be enhanced to return session metadata
    return {"message": "Session listing not yet implemented"}

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Helios API Gateway",
        "version": "1.0.0",
        "description": "Central orchestration service for the Helios observability platform"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 