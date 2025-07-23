import os
import asyncio
import logging
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import yaml
import redis
import json
from pathlib import Path
from asyncio import Semaphore
from collections import defaultdict
import uvloop

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class AnalysisRequest(BaseModel):
    logs: Optional[str] = None
    question: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    priority: Optional[int] = 1  # 1-10, higher = higher priority
    use_cache: Optional[bool] = True
    stream: Optional[bool] = False

class AnalysisResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    model_used: str
    cached: Optional[bool] = False
    processing_time: Optional[float] = None

class BatchAnalysisRequest(BaseModel):
    requests: List[AnalysisRequest]
    max_concurrent: Optional[int] = 10

class BatchAnalysisResponse(BaseModel):
    results: List[AnalysisResponse]
    total_processing_time: float
    average_time_per_request: float

# High-performance cache manager
class CacheManager:
    """Redis-based cache for common log patterns and responses"""
    
    def __init__(self):
        self.redis_client = None
        self.cache_ttl = 3600  # 1 hour
        self._initialize_redis()
    
    def _initialize_redis(self):
        try:
            redis_host = os.getenv("REDIS_HOST", "redis")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            self.redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Cache manager initialized successfully")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory cache: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, logs: str, question: str) -> str:
        """Generate a consistent cache key for log/question combinations"""
        content = f"{logs[:1000]}{question}"  # Use first 1000 chars of logs + question
        return f"helios:analysis:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def get_cached_response(self, logs: str, question: str) -> Optional[str]:
        """Retrieve cached response if available"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key(logs, question)
            cached = self.redis_client.get(cache_key)
            if cached:
                logger.info(f"Cache hit for key: {cache_key[:20]}...")
                return cached
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def cache_response(self, logs: str, question: str, response: str):
        """Cache a response for future use"""
        if not self.redis_client:
            return
        
        try:
            cache_key = self._generate_cache_key(logs, question)
            self.redis_client.setex(cache_key, self.cache_ttl, response)
            logger.info(f"Cached response for key: {cache_key[:20]}...")
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

# Connection pool manager
class ConnectionPoolManager:
    """Manages HTTP connection pools for better performance"""
    
    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self.clients = {}
        self.semaphore = Semaphore(max_connections)
    
    def get_client(self, base_url: str) -> httpx.AsyncClient:
        """Get or create an HTTP client with connection pooling"""
        if base_url not in self.clients:
            limits = httpx.Limits(
                max_keepalive_connections=20,
                max_connections=self.max_connections,
                keepalive_expiry=30.0
            )
            
            self.clients[base_url] = httpx.AsyncClient(
                base_url=base_url,
                timeout=httpx.Timeout(300.0),
                limits=limits
            )
        
        return self.clients[base_url]
    
    async def close_all(self):
        """Close all HTTP clients"""
        for client in self.clients.values():
            await client.aclose()

# Enhanced LLM Provider with performance optimizations
class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, context: Dict[str, Any] = None, stream: bool = False) -> Any:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, context: Dict[str, Any] = None) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name/identifier of the model being used"""
        pass

class OptimizedOllamaProvider(LLMProvider):
    """High-performance Ollama LLM Provider with connection pooling and streaming"""
    
    def __init__(self, base_url: str = "http://ollama:11434", model: str = "llama3.2:latest"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.pool_manager = ConnectionPoolManager(max_connections=50)
        self.request_semaphore = Semaphore(20)  # Limit concurrent requests to Ollama
    
    async def generate(self, prompt: str, context: Dict[str, Any] = None, stream: bool = False) -> str:
        """Generate response using Ollama API with connection pooling"""
        async with self.request_semaphore:
            try:
                client = self.pool_manager.get_client(self.base_url)
                
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": stream,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 2048,  # Reduced for faster responses
                        "num_ctx": 4096,
                        "repeat_penalty": 1.1,
                        "num_thread": 8  # Optimize for multi-threading
                    }
                }
                
                response = await client.post("/api/generate", json=payload)
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "")
                
            except httpx.RequestError as e:
                logger.error(f"Request error to Ollama: {e}")
                raise HTTPException(status_code=503, detail=f"LLM service unavailable: {str(e)}")
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error from Ollama: {e}")
                raise HTTPException(status_code=502, detail=f"LLM service error: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    async def generate_stream(self, prompt: str, context: Dict[str, Any] = None) -> AsyncGenerator[str, None]:
        """Generate streaming response using Ollama API"""
        async with self.request_semaphore:
            try:
                client = self.pool_manager.get_client(self.base_url)
                
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 2048,
                        "num_ctx": 4096,
                        "repeat_penalty": 1.1,
                        "num_thread": 8
                    }
                }
                
                async with client.stream("POST", "/api/generate", json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"Error: {str(e)}"
    
    def get_model_name(self) -> str:
        return f"ollama/{self.model}"

# Queue manager for handling high-throughput requests
class RequestQueueManager:
    """Priority queue manager for handling thousands of requests efficiently"""
    
    def __init__(self, max_concurrent: int = 50):
        self.queues = {
            "high": asyncio.Queue(),
            "medium": asyncio.Queue(),
            "low": asyncio.Queue()
        }
        self.processing_semaphore = Semaphore(max_concurrent)
        self.stats = defaultdict(int)
    
    async def add_request(self, request: AnalysisRequest, callback):
        """Add request to appropriate priority queue"""
        priority = "high" if request.priority >= 7 else "medium" if request.priority >= 4 else "low"
        await self.queues[priority].put((request, callback))
        self.stats["queued"] += 1
    
    async def process_queues(self, ai_core):
        """Process requests from priority queues"""
        while True:
            try:
                # Process high priority first, then medium, then low
                for priority in ["high", "medium", "low"]:
                    try:
                        request, callback = self.queues[priority].get_nowait()
                        asyncio.create_task(self._process_request(ai_core, request, callback))
                    except asyncio.QueueEmpty:
                        continue
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_request(self, ai_core, request: AnalysisRequest, callback):
        """Process individual request with semaphore control"""
        async with self.processing_semaphore:
            try:
                self.stats["processing"] += 1
                result = await ai_core.analyze(request)
                await callback(result)
                self.stats["completed"] += 1
            except Exception as e:
                logger.error(f"Request processing error: {e}")
                self.stats["failed"] += 1
            finally:
                self.stats["processing"] -= 1

class HeliosAICore:
    """High-performance AI Core service with caching, pooling, and async processing"""
    
    def __init__(self):
        self.config = self._load_config()
        self.llm_provider = self._initialize_llm_provider()
        self.cache_manager = CacheManager()
        self.queue_manager = RequestQueueManager(max_concurrent=50)
        
        # Start background queue processor
        asyncio.create_task(self.queue_manager.process_queues(self))
        
        # Optimized Helios persona for faster processing
        self.helios_persona = """You are Helios, an expert AI system for rapid root cause analysis (RCA). Provide concise, structured analysis following this format:

**Root Cause**: [Main issue]
**Impact**: [Severity and affected systems]  
**Fix**: [Immediate action needed]
**Prevention**: [How to prevent recurrence]

Be direct and actionable. Focus on the most critical issues first."""

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with performance optimizations"""
        config = {
            "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
            "max_concurrent_requests": int(os.getenv("MAX_CONCURRENT_REQUESTS", "50")),
            "cache_enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
            "redis_host": os.getenv("REDIS_HOST", "redis"),
            "redis_port": int(os.getenv("REDIS_PORT", "6379"))
        }
        
        # Try to load from config.yaml if it exists
        config_path = Path("/app/config.yaml")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config.update(file_config)
            except Exception as e:
                logger.warning(f"Could not load config.yaml: {e}")
        
        return config
    
    def _initialize_llm_provider(self) -> LLMProvider:
        """Initialize the configured LLM provider with optimizations"""
        provider_type = self.config["llm_provider"].lower()
        
        if provider_type == "ollama":
            return OptimizedOllamaProvider(
                base_url=self.config["ollama_base_url"],
                model=self.config["ollama_model"]
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")
    
    def _build_optimized_prompt(self, request: AnalysisRequest) -> str:
        """Build optimized prompt for faster processing"""
        prompt_parts = [self.helios_persona]
        
        if request.logs:
            # Truncate very long logs to first and last parts for faster processing
            logs = request.logs
            if len(logs) > 5000:
                logs = logs[:2500] + "\n...[truncated]...\n" + logs[-2500:]
            
            prompt_parts.append(f"""
## Logs to Analyze
```
{logs}
```
""")
        
        prompt_parts.append(f"""
## Question: {request.question}

Provide rapid RCA analysis focusing on the most critical issues.""")
        
        return "\n".join(prompt_parts)
    
    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """High-performance analysis with caching and optimizations"""
        start_time = time.time()
        cached_response = None
        
        try:
            # Check cache first if enabled
            if request.use_cache and self.config["cache_enabled"]:
                cached_response = await self.cache_manager.get_cached_response(
                    request.logs or "", request.question
                )
                
                if cached_response:
                    return AnalysisResponse(
                        response=cached_response,
                        session_id=request.session_id,
                        model_used=self.llm_provider.get_model_name(),
                        cached=True,
                        processing_time=time.time() - start_time
                    )
            
            # Build optimized prompt
            prompt = self._build_optimized_prompt(request)
            
            # Generate response
            logger.info(f"Generating response using {self.llm_provider.get_model_name()}")
            response_text = await self.llm_provider.generate(prompt, stream=request.stream)
            
            # Cache the response if enabled
            if request.use_cache and self.config["cache_enabled"]:
                await self.cache_manager.cache_response(
                    request.logs or "", request.question, response_text
                )
            
            processing_time = time.time() - start_time
            
            return AnalysisResponse(
                response=response_text,
                session_id=request.session_id,
                model_used=self.llm_provider.get_model_name(),
                cached=False,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    async def analyze_batch(self, batch_request: BatchAnalysisRequest) -> BatchAnalysisResponse:
        """Process multiple requests concurrently"""
        start_time = time.time()
        
        # Limit concurrent requests
        semaphore = Semaphore(batch_request.max_concurrent or 10)
        
        async def process_single(request: AnalysisRequest) -> AnalysisResponse:
            async with semaphore:
                return await self.analyze(request)
        
        # Process all requests concurrently
        tasks = [process_single(req) for req in batch_request.requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {result}")
                # Create error response
                valid_results.append(AnalysisResponse(
                    response=f"Error processing request: {str(result)}",
                    session_id=batch_request.requests[i].session_id,
                    model_used=self.llm_provider.get_model_name(),
                    cached=False,
                    processing_time=0.0
                ))
            else:
                valid_results.append(result)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(batch_request.requests) if batch_request.requests else 0
        
        return BatchAnalysisResponse(
            results=valid_results,
            total_processing_time=total_time,
            average_time_per_request=avg_time
        )
    
    async def stream_analysis(self, request: AnalysisRequest) -> AsyncGenerator[str, None]:
        """Stream analysis results for faster perceived performance"""
        try:
            prompt = self._build_optimized_prompt(request)
            
            async for chunk in self.llm_provider.generate_stream(prompt):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

# FastAPI application with performance optimizations
app = FastAPI(
    title="Helios AI Core - High Performance",
    description="Optimized AI-powered root cause analysis service for high-throughput scenarios"
)

ai_core = HeliosAICore()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "helios-ai-core-optimized",
        "queue_stats": dict(ai_core.queue_manager.stats),
        "model": ai_core.llm_provider.get_model_name()
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_logs(request: AnalysisRequest):
    """Optimized analysis endpoint"""
    return await ai_core.analyze(request)

@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(batch_request: BatchAnalysisRequest):
    """Batch analysis for processing multiple requests efficiently"""
    return await ai_core.analyze_batch(batch_request)

@app.post("/analyze/stream")
async def analyze_stream(request: AnalysisRequest):
    """Streaming analysis for faster perceived performance"""
    return StreamingResponse(
        ai_core.stream_analysis(request),
        media_type="text/plain"
    )

@app.get("/stats")
async def get_stats():
    """Performance statistics endpoint"""
    return {
        "queue_stats": dict(ai_core.queue_manager.stats),
        "cache_enabled": ai_core.config["cache_enabled"],
        "max_concurrent": ai_core.config["max_concurrent_requests"],
        "model": ai_core.llm_provider.get_model_name()
    }

@app.get("/config")
async def get_config():
    """Get current configuration (excluding sensitive data)"""
    safe_config = ai_core.config.copy()
    return {
        "llm_provider": safe_config["llm_provider"],
        "model": ai_core.llm_provider.get_model_name(),
        "cache_enabled": safe_config["cache_enabled"],
        "max_concurrent_requests": safe_config["max_concurrent_requests"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        loop="uvloop",  # Use uvloop for better performance
        workers=1,      # Single worker with async handling
        access_log=False  # Disable access logs for better performance
    ) 