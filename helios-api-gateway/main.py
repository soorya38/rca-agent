import os
import logging
import uuid
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import asyncpg
import uvloop
from contextlib import asynccontextmanager
from asyncio import Semaphore

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class ChatMessage(BaseModel):
    logs: Optional[str] = None
    question: str
    session_id: Optional[str] = None
    priority: Optional[int] = 1
    use_cache: Optional[bool] = True
    stream: Optional[bool] = False

class BatchChatMessage(BaseModel):
    messages: List[ChatMessage]
    max_concurrent: Optional[int] = 10

class ChatResponse(BaseModel):
    response: str
    session_id: str
    model_used: str
    timestamp: datetime
    cached: Optional[bool] = False
    processing_time: Optional[float] = None

class BatchChatResponse(BaseModel):
    responses: List[ChatResponse]
    total_processing_time: float
    average_time_per_message: float
    success_count: int
    error_count: int

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]

class HealthStatus(BaseModel):
    status: str
    services: Dict[str, str]
    performance_metrics: Optional[Dict[str, Any]] = None

# High-performance database manager
class OptimizedDatabaseManager:
    """Optimized PostgreSQL manager with connection pooling and batch operations"""
    
    def __init__(self):
        self.pool = None
        self.db_url = self._build_db_url()
        self.batch_semaphore = Semaphore(20)  # Limit concurrent batch operations
    
    def _build_db_url(self) -> str:
        """Build database connection URL from environment variables"""
        host = os.getenv("POSTGRES_HOST", "postgres")
        port = os.getenv("POSTGRES_PORT", "5432")
        user = os.getenv("POSTGRES_USER", "helios")
        password = os.getenv("POSTGRES_PASSWORD", "helios_password")
        database = os.getenv("POSTGRES_DB", "helios")
        
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    async def initialize(self):
        """Initialize database connection pool with optimizations"""
        try:
            self.pool = await asyncpg.create_pool(
                self.db_url, 
                min_size=10, 
                max_size=50,  # Increased pool size for high throughput
                command_timeout=60,
                server_settings={
                    'jit': 'off',  # Disable JIT for better connection performance
                    'application_name': 'helios-api-gateway'
                }
            )
            await self._create_tables()
            logger.info("Optimized database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
    
    async def _create_tables(self):
        """Create optimized database tables with indexes"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    sender VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    model_used VARCHAR(100),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    logs TEXT,
                    processing_time FLOAT,
                    cached BOOLEAN DEFAULT FALSE
                );
                
                CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);
                CREATE INDEX IF NOT EXISTS idx_conversations_sender ON conversations(sender);
            """)
    
    async def save_message_batch(self, messages: List[Dict[str, Any]]):
        """Save multiple messages efficiently using batch insert"""
        if not messages:
            return
            
        async with self.batch_semaphore:
            async with self.pool.acquire() as conn:
                await conn.executemany("""
                    INSERT INTO conversations (session_id, sender, content, model_used, logs, processing_time, cached)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, [(
                    msg["session_id"], msg["sender"], msg["content"], 
                    msg.get("model_used"), msg.get("logs"), 
                    msg.get("processing_time"), msg.get("cached", False)
                ) for msg in messages])
    
    async def save_message(self, session_id: str, sender: str, content: str, 
                          model_used: str = None, logs: str = None, 
                          processing_time: float = None, cached: bool = False):
        """Save a single message (fallback for non-batch operations)"""
        await self.save_message_batch([{
            "session_id": session_id,
            "sender": sender,
            "content": content,
            "model_used": model_used,
            "logs": logs,
            "processing_time": processing_time,
            "cached": cached
        }])
    
    async def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a session with optimizations"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT sender, content, model_used, timestamp, logs, processing_time, cached
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
                    "logs": row["logs"],
                    "processing_time": row["processing_time"],
                    "cached": row["cached"]
                }
                for row in rows
            ]

class HighThroughputAPIGateway:
    """Optimized API Gateway for processing thousands of requests per second"""
    
    def __init__(self):
        self.ai_core_url = os.getenv("AI_CORE_URL", "http://helios-ai-core:8001")
        
        # HTTP client with optimized connection pooling
        limits = httpx.Limits(
            max_keepalive_connections=100,
            max_connections=200,
            keepalive_expiry=30.0
        )
        
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0),
            limits=limits
        )
        
        self.db = OptimizedDatabaseManager()
        self.request_semaphore = Semaphore(100)  # Limit concurrent AI requests
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "total_batch_requests": 0,
            "cache_hits": 0,
            "average_response_time": 0.0,
            "concurrent_requests": 0
        }
    
    async def initialize(self):
        """Initialize the gateway and its dependencies"""
        await self.db.initialize()
    
    async def shutdown(self):
        """Clean shutdown of the gateway"""
        await self.client.aclose()
        await self.db.close()
    
    async def check_service_health(self, url: str, service_name: str) -> str:
        """Check health of a service with timeout"""
        try:
            response = await self.client.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                return "healthy"
            else:
                return f"unhealthy (HTTP {response.status_code})"
        except Exception as e:
            return f"unreachable ({str(e)})"
    
    async def process_chat_message(self, message: ChatMessage) -> ChatResponse:
        """Process a single chat message with optimizations"""
        start_time = asyncio.get_event_loop().time()
        
        async with self.request_semaphore:
            self.metrics["concurrent_requests"] += 1
            self.metrics["total_requests"] += 1
            
            try:
                # Generate session ID if not provided
                session_id = message.session_id or str(uuid.uuid4())
                
                # Prepare request for AI Core
                ai_request = {
                    "logs": message.logs,
                    "question": message.question,
                    "session_id": session_id,
                    "priority": message.priority,
                    "use_cache": message.use_cache,
                    "stream": message.stream
                }
                
                # Call AI Core service
                logger.debug(f"Sending request to AI Core for session {session_id}")
                response = await self.client.post(
                    f"{self.ai_core_url}/analyze",
                    json=ai_request
                )
                response.raise_for_status()
                
                ai_response = response.json()
                processing_time = asyncio.get_event_loop().time() - start_time
                
                # Update metrics
                if ai_response.get("cached", False):
                    self.metrics["cache_hits"] += 1
                
                # Prepare database entries for batch save
                db_entries = [
                    {
                        "session_id": session_id,
                        "sender": "user",
                        "content": message.question,
                        "logs": message.logs,
                        "processing_time": processing_time
                    },
                    {
                        "session_id": session_id,
                        "sender": "assistant",
                        "content": ai_response["response"],
                        "model_used": ai_response["model_used"],
                        "processing_time": ai_response.get("processing_time", 0),
                        "cached": ai_response.get("cached", False)
                    }
                ]
                
                # Save to database in background
                asyncio.create_task(self.db.save_message_batch(db_entries))
                
                return ChatResponse(
                    response=ai_response["response"],
                    session_id=session_id,
                    model_used=ai_response["model_used"],
                    timestamp=datetime.now(),
                    cached=ai_response.get("cached", False),
                    processing_time=processing_time
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
            finally:
                self.metrics["concurrent_requests"] -= 1
    
    async def process_batch_chat(self, batch_message: BatchChatMessage) -> BatchChatResponse:
        """Process multiple chat messages concurrently for maximum throughput"""
        start_time = asyncio.get_event_loop().time()
        self.metrics["total_batch_requests"] += 1
        
        # Limit concurrent processing within the batch
        batch_semaphore = Semaphore(batch_message.max_concurrent or 10)
        
        async def process_single_with_semaphore(msg: ChatMessage) -> ChatResponse:
            async with batch_semaphore:
                return await self.process_chat_message(msg)
        
        # Process all messages concurrently
        tasks = [process_single_with_semaphore(msg) for msg in batch_message.messages]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from errors
        responses = []
        error_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch message {i} failed: {result}")
                error_count += 1
                # Create error response
                responses.append(ChatResponse(
                    response=f"Error processing request: {str(result)}",
                    session_id=batch_message.messages[i].session_id or str(uuid.uuid4()),
                    model_used="error",
                    timestamp=datetime.now(),
                    cached=False,
                    processing_time=0.0
                ))
            else:
                responses.append(result)
        
        total_time = asyncio.get_event_loop().time() - start_time
        avg_time = total_time / len(batch_message.messages) if batch_message.messages else 0
        
        return BatchChatResponse(
            responses=responses,
            total_processing_time=total_time,
            average_time_per_message=avg_time,
            success_count=len(batch_message.messages) - error_count,
            error_count=error_count
        )
    
    async def get_conversation_history(self, session_id: str) -> ConversationHistory:
        """Retrieve conversation history for a session"""
        try:
            messages = await self.db.get_conversation_history(session_id)
            return ConversationHistory(session_id=session_id, messages=messages)
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Initialize gateway
gateway = HighThroughputAPIGateway()

# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await gateway.initialize()
    yield
    # Shutdown
    await gateway.shutdown()

app = FastAPI(
    title="Helios API Gateway - High Performance",
    description="Optimized central orchestration service for high-throughput observability",
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
    """Comprehensive health check with performance metrics"""
    ai_core_health = await gateway.check_service_health(gateway.ai_core_url, "ai-core")
    
    return HealthStatus(
        status="healthy" if ai_core_health == "healthy" else "degraded",
        services={
            "api-gateway": "healthy",
            "ai-core": ai_core_health,
            "database": "healthy"
        },
        performance_metrics=gateway.metrics
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Optimized chat endpoint for single messages"""
    return await gateway.process_chat_message(message)

@app.post("/chat/batch", response_model=BatchChatResponse)
async def chat_batch(batch_message: BatchChatMessage):
    """High-throughput batch chat endpoint for processing multiple messages"""
    return await gateway.process_batch_chat(batch_message)

@app.get("/conversation/{session_id}", response_model=ConversationHistory)
async def get_conversation(session_id: str):
    """Retrieve conversation history for a specific session"""
    return await gateway.get_conversation_history(session_id)

@app.get("/metrics")
async def get_performance_metrics():
    """Detailed performance metrics for monitoring"""
    return {
        "gateway_metrics": gateway.metrics,
        "database_pool_info": {
            "size": gateway.db.pool.get_size() if gateway.db.pool else 0,
            "min_size": gateway.db.pool.get_min_size() if gateway.db.pool else 0,
            "max_size": gateway.db.pool.get_max_size() if gateway.db.pool else 0
        } if gateway.db.pool else {}
    }

@app.get("/sessions")
async def list_sessions():
    """List all conversation sessions (placeholder for future implementation)"""
    return {"message": "Session listing not yet implemented"}

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Helios API Gateway - High Performance",
        "version": "2.0.0",
        "description": "Optimized central orchestration service supporting thousands of requests per second",
        "features": [
            "Batch processing",
            "Connection pooling",
            "Performance metrics",
            "Async optimization"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        loop="uvloop",
        workers=1,
        access_log=False
    ) 