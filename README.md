# Project Helios 🌟

**The Extensible, Self-Hosted Observability Platform for AI-Powered Root Cause Analysis**

Helios is a modular, plug-and-play AIOps platform designed to revolutionize how you approach system troubleshooting and root cause analysis. Built with extensibility at its core, Helios allows you to easily swap out components, integrate new data sources, and leverage different AI models for intelligent log analysis.

## ⭐ Key Features

- **🧠 AI-Powered Analysis**: Intelligent root cause analysis using Large Language Models
- **🔧 Plug-and-Play Architecture**: Easily swap LLM providers (Ollama, OpenAI, etc.)
- **🏠 Self-Hosted**: Complete control over your data and infrastructure
- **💬 Chat Interface**: Intuitive conversation-based interaction with your logs
- **📊 Structured RCA**: Systematic, engineering-focused analysis framework
- **🐳 Containerized**: Full Docker Compose deployment for easy setup
- **🔍 Syntax Highlighting**: Beautiful code and log rendering in the UI
- **💾 Persistent Memory**: Conversation history with PostgreSQL storage

## 🏗️ Architecture

Helios follows a microservices architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│  helios-frontend│────│ helios-api-     │────│ helios-ai-core  │
│   (React/Vite)  │    │   gateway       │    │  (FastAPI +     │
│                 │    │  (FastAPI)      │    │   LLM Provider) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                │                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │                 │    │                 │
                       │   PostgreSQL    │    │     Ollama      │
                       │   (Memory       │    │   (LLM Model)   │
                       │    Store)       │    │                 │
                       └─────────────────┘    └─────────────────┘
```

### 🔄 Data Flow

1. **User Input**: User pastes logs or asks questions via the React frontend
2. **API Gateway**: Receives request, manages session, stores in database
3. **AI Core**: Processes request using configured LLM provider
4. **LLM Provider**: Generates structured root cause analysis
5. **Response**: Formatted response returned through the chain to user

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- 8GB+ RAM (recommended for local LLM)
- GPU support (optional but recommended for better performance)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd helios
   ```

2. **Run the setup script:**
   ```bash
   ./scripts/setup.sh
   ```
   
   This script will:
   - Create configuration files
   - Build all services
   - Pull the default Ollama model (llama3:latest)
   - Verify service health

3. **Access Helios:**
   - **Frontend**: http://localhost:3000
   - **API Gateway**: http://localhost:8000
   - **AI Core**: http://localhost:8001

### Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create environment file
cp env.example .env

# Start all services
docker-compose up -d

# Pull the default model
docker exec helios-ollama ollama pull llama3:latest
```

## 🎮 Usage

### Basic Log Analysis

1. Open http://localhost:3000 in your browser
2. Paste your log content in the text area
3. Add a question like "What went wrong?" at the end
4. Click Send or press Ctrl+Enter

### Example Interaction

```
Paste this into Helios:

2024-01-15 10:30:15 ERROR [UserService] Database connection failed: Connection timeout
2024-01-15 10:30:15 WARN [ConnectionPool] Pool exhausted, max connections: 20
2024-01-15 10:30:16 ERROR [UserService] Failed to authenticate user: Database unavailable
2024-01-15 10:30:20 INFO [HealthCheck] Database health check failed
2024-01-15 10:30:25 ERROR [UserService] Database connection failed: Connection timeout

What caused this issue and how can I fix it?
```

Helios will provide a structured analysis with:
- Initial assessment
- Timeline reconstruction  
- Root cause identification
- Impact assessment
- Specific recommendations
- Prevention strategies

## ⚙️ Configuration

### Environment Variables

Key configuration options in your `.env` file:

```bash
# LLM Provider (ollama or openai)
LLM_PROVIDER=ollama

# Ollama settings
OLLAMA_MODEL=llama3:latest
OLLAMA_BASE_URL=http://ollama:11434

# OpenAI settings (if using OpenAI)
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4

# Database settings
POSTGRES_PASSWORD=your_secure_password
```

### Advanced Configuration

Edit `config.yaml` for advanced settings:

```yaml
llm:
  provider: "ollama"
  ollama:
    model: "llama3:latest"
    options:
      temperature: 0.7
      max_tokens: 4000
```

### Switching LLM Providers

**To use OpenAI instead of Ollama:**

1. Edit `.env`:
   ```bash
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_api_key_here
   ```

2. Restart services:
   ```bash
   docker-compose restart helios-ai-core
   ```

**To use different Ollama models:**

1. Pull a new model:
   ```bash
   docker exec helios-ollama ollama pull mistral:latest
   ```

2. Update `.env`:
   ```bash
   OLLAMA_MODEL=mistral:latest
   ```

3. Restart AI Core:
   ```bash
   docker-compose restart helios-ai-core
   ```

## 🔧 Development

### Project Structure

```
helios/
├── helios-frontend/          # React frontend application
│   ├── src/
│   ├── package.json
│   └── Dockerfile
├── helios-api-gateway/       # FastAPI gateway service
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── helios-ai-core/          # AI/LLM service
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── scripts/
│   └── setup.sh            # Automated setup script
├── docker-compose.yml      # Main orchestration file
├── config.yaml            # Advanced configuration
└── env.example            # Environment template
```

### Running in Development Mode

**Frontend development:**
```bash
cd helios-frontend
npm install
npm run dev
```

**Backend development:**
```bash
cd helios-api-gateway
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Adding New LLM Providers

1. **Create a new provider class** in `helios-ai-core/main.py`:
   ```python
   class CustomProvider(LLMProvider):
       async def generate(self, prompt: str, context: Dict[str, Any] = None) -> str:
           # Your implementation
           pass
   
       def get_model_name(self) -> str:
           return "custom/model-name"
   ```

2. **Register the provider** in the `_initialize_llm_provider` method

3. **Add configuration** in `config.yaml` and environment variables

## 🎯 Use Cases

### DevOps & SRE
- Analyze application crashes and errors
- Troubleshoot deployment failures
- Investigate performance degradations
- Root cause analysis for outages

### Security Operations
- Analyze security logs for threats
- Investigate authentication failures
- Trace attack patterns
- Incident response support

### Application Development
- Debug complex application issues
- Analyze error patterns
- Performance bottleneck identification
- Code review assistance

## 🛡️ Security Considerations

### Production Deployment

- Change default passwords in `.env`
- Configure proper CORS settings
- Use HTTPS with reverse proxy (nginx/traefik)
- Implement authentication (future feature)
- Regular security updates for base images

### Data Privacy

- All data processed locally (self-hosted)
- No external API calls when using Ollama
- Conversation history stored in local PostgreSQL
- Full control over data retention policies

## 🔮 Roadmap

### Milestone 3: Enhanced Extensibility
- [ ] Loki log connector
- [ ] Prometheus metrics integration  
- [ ] Custom connector framework
- [ ] Plugin marketplace

### Future Features
- [ ] Multi-tenant support
- [ ] Role-based access control
- [ ] Real-time log streaming
- [ ] Advanced analytics dashboard
- [ ] API integrations (PagerDuty, Slack)
- [ ] Custom alert rules
- [ ] ML-based anomaly detection

## 🐛 Troubleshooting

### Common Issues

**Ollama model download fails:**
```bash
# Check internet connection and retry
docker exec helios-ollama ollama pull llama3:latest
```

**Services won't start:**
```bash
# Check logs
docker-compose logs -f

# Restart specific service
docker-compose restart helios-ai-core
```

**Frontend can't connect to API:**
- Verify `VITE_API_URL` in `.env`
- Check API Gateway is running: `curl http://localhost:8000/health`

**Database connection issues:**
```bash
# Reset database
docker-compose down -v
docker-compose up -d postgres
# Wait for PostgreSQL to be ready, then start other services
```

### Health Checks

Check service status:
```bash
# Overall health
curl http://localhost:8000/health

# Individual services  
curl http://localhost:8001/health  # AI Core
curl http://localhost:3000         # Frontend
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:

- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Ollama team for the excellent local LLM platform
- FastAPI for the robust API framework
- Mantine for the beautiful React components
- The open-source community for inspiration and tools

---

**Helios** - Illuminating the path to root cause analysis ✨

For questions, issues, or feature requests, please open an issue on GitHub. 