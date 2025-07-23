import os
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class AnalysisRequest(BaseModel):
    logs: Optional[str] = None
    question: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

class AnalysisResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    model_used: str

# Abstract LLM Provider Interface
class LLMProvider(ABC):
    """Abstract base class for LLM providers to ensure extensibility"""
    
    @abstractmethod
    async def generate(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate a response from the LLM given a prompt and context"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name/identifier of the model being used"""
        pass

class OllamaProvider(LLMProvider):
    """Ollama LLM Provider implementation"""
    
    def __init__(self, base_url: str = "http://ollama:11434", model: str = "llama3.2:latest"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout for LLM responses
    
    async def generate(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate response using Ollama API"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 4000
                }
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
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
    
    def get_model_name(self) -> str:
        return f"ollama/{self.model}"

class OpenAIProvider(LLMProvider):
    """OpenAI LLM Provider implementation (placeholder for extensibility)"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        # TODO: Initialize OpenAI client when needed
    
    async def generate(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate response using OpenAI API"""
        # Placeholder implementation
        raise NotImplementedError("OpenAI provider not yet implemented")
    
    def get_model_name(self) -> str:
        return f"openai/{self.model}"

class HeliosAICore:
    """Main AI Core service that orchestrates LLM interactions"""
    
    def __init__(self):
        self.config = self._load_config()
        self.llm_provider = self._initialize_llm_provider()
        
        # Helios persona and RCA framework
        self.helios_persona = """You are Helios, an expert AI system specializing in root cause analysis (RCA) for complex systems. You are methodical, thorough, and always provide structured analysis. Your responses should be clear, actionable, and follow engineering best practices.

When analyzing logs or system issues, you follow this systematic approach:
1. **Initial Assessment** - Quickly identify the scope and severity
2. **Timeline Reconstruction** - Establish sequence of events
3. **Error Pattern Analysis** - Identify recurring themes and anomalies
4. **Root Cause Identification** - Determine the underlying cause(s)
5. **Impact Assessment** - Evaluate consequences and affected systems
6. **Recommendations** - Provide specific, actionable solutions
7. **Prevention Strategies** - Suggest measures to prevent recurrence

Always structure your responses with clear headings, bullet points, and actionable insights."""

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables and config file"""
        config = {
            "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4")
        }
        
        # Try to load from config.yaml if it exists
        config_path = Path("/app/config.yaml")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    config.update(file_config)
            except Exception as e:
                logger.warning(f"Could not load config.yaml: {e}")
        
        return config
    
    def _initialize_llm_provider(self) -> LLMProvider:
        """Initialize the configured LLM provider"""
        provider_type = self.config["llm_provider"].lower()
        
        if provider_type == "ollama":
            return OllamaProvider(
                base_url=self.config["ollama_base_url"],
                model=self.config["ollama_model"]
            )
        elif provider_type == "openai":
            if not self.config["openai_api_key"]:
                raise ValueError("OpenAI API key required for OpenAI provider")
            return OpenAIProvider(
                api_key=self.config["openai_api_key"],
                model=self.config["openai_model"]
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")
    
    def _build_analysis_prompt(self, request: AnalysisRequest) -> str:
        """Construct the complete prompt for the LLM"""
        prompt_parts = [self.helios_persona]
        
        if request.logs:
            prompt_parts.append(f"""
## System Logs for Analysis
The following logs need to be analyzed:

```
{request.logs}
```
""")
        
        if request.context:
            prompt_parts.append(f"""
## Additional Context
{request.context}
""")
        
        prompt_parts.append(f"""
## User Question
{request.question}

Please analyze the provided information and respond with a structured analysis following your systematic RCA approach. Format your response in clear markdown with appropriate headings and bullet points.""")
        
        return "\n".join(prompt_parts)
    
    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Main analysis method that processes user requests"""
        try:
            # Build the prompt
            prompt = self._build_analysis_prompt(request)
            
            # Generate response using configured LLM
            logger.info(f"Generating response using {self.llm_provider.get_model_name()}")
            response_text = await self.llm_provider.generate(prompt)
            
            # Return structured response
            return AnalysisResponse(
                response=response_text,
                session_id=request.session_id,
                model_used=self.llm_provider.get_model_name()
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# FastAPI application
app = FastAPI(title="Helios AI Core", description="AI-powered root cause analysis service")
ai_core = HeliosAICore()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "helios-ai-core"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_logs(request: AnalysisRequest):
    """Main analysis endpoint"""
    return await ai_core.analyze(request)

@app.get("/config")
async def get_config():
    """Get current configuration (excluding sensitive data)"""
    safe_config = ai_core.config.copy()
    safe_config.pop("openai_api_key", None)  # Remove sensitive data
    return {
        "llm_provider": safe_config["llm_provider"],
        "model": ai_core.llm_provider.get_model_name()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 