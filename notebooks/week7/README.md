# Week 7: Agentic RAG with LangGraph + Telegram Bot

## Overview

Week 7 adds TWO major enhancements to the arXiv Paper Curator:

1. **🤖 Agentic RAG with LangGraph** - Intelligent, adaptive retrieval with decision-making
2. **💬 Telegram Bot Integration** - Conversational interface for mobile/desktop access

---

## 🧠 Part 1: Agentic RAG with LangGraph

### What is Agentic RAG?

**Traditional RAG** (Weeks 5-6):
```
Query → Always Retrieve → Generate Answer
```

**Agentic RAG** (Week 7):
```
Query → Agent Decides:
  ├─ Simple question? → Respond directly (faster!)
  └─ Research needed? → Retrieve
       ├─ Relevant docs? → Generate answer
       └─ Not relevant? → Rewrite query → Try again
```

### Key Features

- **🎯 Intelligent Decision Making** - LLM decides when retrieval is actually needed
- **📊 Document Grading** - Validates that retrieved papers are relevant
- **🔄 Query Refinement** - Rewrites vague queries for better results
- **🔍 Reasoning Transparency** - Shows the agent's decision-making steps
- **♻️ Iterative Improvement** - Can retry with better queries if needed

### What We Built

```
src/services/agents/
├── tools.py            # Retriever tool wrapping OpenSearch
├── nodes.py            # 4 graph nodes (query, grade, rewrite, generate)
├── agentic_rag.py      # LangGraph workflow + service
├── prompts.py          # LLM prompt templates
└── factory.py          # Dependency injection

src/routers/
└── agentic_ask.py      # FastAPI endpoint

Total: ~750 LOC following SOLID, KISS, DRY principles
```

### Architecture

```
LangGraph Workflow:

START
  ↓
generate_query_or_respond
  ├─ No retrieval needed → END (direct response)
  └─ Needs retrieval → retrieve (ToolNode)
       ↓
     grade_documents
       ├─ Relevant → generate_answer → END
       └─ Not relevant → rewrite_query → (loop back)
```

### New API Endpoint

**`POST /api/v1/ask-agentic`**

```json
// Request
{
  "query": "What are transformers in ML?",
  "top_k": 3,
  "use_hybrid": true
}

// Response
{
  "query": "What are transformers in ML?",
  "answer": "Transformers are neural network architectures...",
  "sources": ["https://arxiv.org/pdf/1706.03762.pdf"],
  "chunks_used": 3,
  "search_mode": "hybrid",
  "reasoning_steps": [
    "Decided to retrieve relevant papers",
    "Retrieved documents from database",
    "Generated answer from relevant documents"
  ],
  "retrieval_attempts": 1
}
```

### Quick Start: Agentic RAG

**1. Ensure services are running:**
```bash
docker compose up --build -d
```

**2. Test with cURL:**
```bash
# Simple question (should respond directly)
curl -X POST http://localhost:8000/api/v1/ask-agentic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is 2+2?",
    "top_k": 3,
    "use_hybrid": true
  }'

# Research question (should retrieve papers)
curl -X POST http://localhost:8000/api/v1/ask-agentic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are attention mechanisms?",
    "top_k": 3,
    "use_hybrid": true
  }'
```

**3. Interactive Testing:**
```bash
# Open Jupyter notebook
jupyter notebook notebooks/week7/week7_agentic_rag.ipynb
```

### Comparison: Traditional vs. Agentic

| Feature | Traditional RAG | Agentic RAG |
|---------|----------------|-------------|
| **Retrieval** | Always retrieves | Decides when needed |
| **Relevance Check** | None | Grades documents |
| **Query Refinement** | None | Rewrites if needed |
| **Iterations** | Single pass | Multiple attempts |
| **Transparency** | Black box | Shows reasoning |
| **Simple Questions** | ~15-20s | ~2-5s (no retrieval) |
| **Complex Questions** | Single attempt | Iterative refinement |

### Testing Scenarios

**Scenario 1: Direct Response (No Retrieval)**
- Query: "What is 5 + 7?"
- Expected: Agent responds "12" without retrieving papers
- Reasoning: "Responded directly without retrieval"

**Scenario 2: Successful Retrieval**
- Query: "What are transformers in machine learning?"
- Expected: Agent retrieves papers, grades as relevant, generates answer
- Reasoning: "Decided to retrieve" → "Retrieved documents" → "Generated answer"

**Scenario 3: Query Rewriting**
- Query: "Tell me about ML stuff" (vague)
- Expected: Agent retrieves, grades as not relevant, rewrites query, tries again
- Reasoning: "Retrieved" → "Not relevant" → "Rewritten query" → "Retrieved again" → "Generated answer"

### Design Principles Followed

- ✅ **SOLID** - Single responsibility, dependency inversion, composition
- ✅ **KISS** - Simple nodes (<30 lines), clear logic
- ✅ **DRY** - Reused existing services (OpenSearch, Ollama, Jina)
- ✅ **YAGNI** - Only implemented what's needed
- ✅ **Explicit** - Type hints, docstrings, clear names
- ✅ **2025 Best Practices** - MessagesState, ToolNode, tools_condition

### Documentation

- **Implementation Plan**: `docs/AGENTIC_RAG_IMPLEMENTATION_PLAN.md`
- **Testing Plan**: `docs/AGENTIC_RAG_TESTING_PLAN.md`
- **LangGraph 2025 Patterns**: `docs/LANGGRAPH_2025_BEST_PRACTICES.md`
- **Design Principles**: `docs/DESIGN_PRINCIPLES.md`
- **Interactive Notebook**: `notebooks/week7/week7_agentic_rag.ipynb`

---

## 💬 Part 2: Telegram Bot Integration

### What We Built

- **🤖 Full Telegram Bot Integration**: Conversational interface with command support
- **💬 Natural Language Queries**: Ask questions in plain language, get answers with sources
- **⚡ All Week 6 Features**: Redis caching (150-400x speedup) and Langfuse tracing
- **🎯 Interactive Commands**: `/start`, `/help`, `/ask`, `/search`, `/settings`, `/status`
- **👤 User Session Management**: Per-user preferences and conversation history
- **📱 Mobile-First**: Rich message formatting with clickable arXiv links
- **🔐 Optional Access Control**: Whitelist specific users if needed

## Architecture

### Data Flow
```
Telegram User
    ↓
Telegram Bot (Polling/Webhook)
    ↓
TelegramService + Handlers
    ↓ [Langfuse Tracing]
Cache Check (Redis)
    ├─ Hit → Instant Response (~100ms)
    └─ Miss → Full RAG Pipeline
        ↓
Hybrid Search (OpenSearch BM25 + Vector)
        ↓
LLM Generation (Ollama)
        ↓
Cache Store (Redis)
        ↓
Format Response → Send to Telegram
```

### New Components

```
src/services/telegram/
├── client.py           # Main Telegram bot service
├── handlers.py         # Command and message handlers
├── formatters.py       # Message formatting utilities
├── keyboards.py        # Interactive inline keyboards
├── user_manager.py     # User settings and sessions
└── factory.py          # Factory function

src/schemas/telegram/
├── messages.py         # Message validation schemas
├── commands.py         # Command schemas
└── user_settings.py    # User preferences schema
```

## Key Features

### **Conversational Interface**
- Natural language queries: Just send a message
- Contextual conversation with history tracking
- Automatic query routing (commands vs regular messages)

### **Rich Commands**
```
/start    - Welcome message with bot capabilities
/help     - Detailed usage instructions
/ask      - Explicitly ask a research question
/search   - Quick paper search by keywords
/settings - Customize preferences (search mode, results count)
/status   - Check system health and statistics
/clear    - Clear conversation history
```

### **Interactive Settings**
- Search mode: Hybrid (BM25 + semantic) or BM25 only
- Results per query: 3, 5, or 10 papers
- Category filters: All, cs.AI, cs.LG, cs.CL, etc.
- Model selection: Choose LLM model
- Toggles: Streaming, source display, compact mode

### **User Experience**
- **Typing indicators** during processing
- **Rich formatting** with Markdown support
- **Clickable links** to arXiv papers
- **Automatic message splitting** for long responses
- **Source citations** with paper metadata
- **Cache indicators** (⚡) for instant responses

## Quick Start

### Prerequisites

1. **Telegram Account** - Install Telegram on your phone or computer
2. **All Week 1-6 Services Running** - Full RAG stack must be operational

### Step 1: Create Your Telegram Bot

1. **Open Telegram** and search for `@BotFather`
2. **Send** `/newbot` to BotFather
3. **Follow prompts**:
   - Choose a name (e.g., "My arXiv Curator")
   - Choose a username (e.g., "my_arxiv_curator_bot" - must end in "bot")
4. **Copy the bot token** - You'll receive something like:
   ```
   1234567890:ABCdefGHIjklMNOpqrsTUVwxyz-1234567
   ```

### Step 2: Configure Environment

Add these variables to your `.env` file:

```bash
# Enable Telegram bot
TELEGRAM__ENABLED=true
TELEGRAM__BOT_TOKEN=your_token_from_botfather_here

# Optional: Restrict to specific users (comma-separated Telegram user IDs)
# Leave empty to allow all users
TELEGRAM__ALLOWED_USER_IDS=

# Use polling mode for development (webhook requires HTTPS)
TELEGRAM__USE_WEBHOOK=false
```

### Step 3: Install Dependencies

```bash
# Install python-telegram-bot library
uv sync
```

### Step 4: Start Services

```bash
# Start all services (includes Telegram bot)
docker compose up --build -d

# Check logs to verify Telegram bot started
docker compose logs -f api
```

You should see:
```
INFO - Telegram bot started successfully
INFO - Starting Telegram bot in polling mode
INFO - Bot commands set successfully
```

### Step 5: Test Your Bot

1. **Open Telegram** and search for your bot username
2. **Send** `/start` to your bot
3. **Try asking**: "What are transformers in machine learning?"
4. **Verify** you receive an answer with sources!

## Testing Instructions

### Manual Testing Scenarios

#### Scenario 1: Basic Commands

**Test `/start` command:**
```
You: /start
Bot: 👋 Welcome to arXiv Paper Curator!
     [Shows capabilities and quick commands]
```

**Test `/help` command:**
```
You: /help
Bot: 📚 arXiv Paper Curator Help
     [Shows detailed command documentation]
```

**Test `/status` command:**
```
You: /status
Bot: 🔧 System Status
     ✅ OPENSEARCH
     ✅ OLLAMA
     ✅ CACHE
     [Shows system health]
```

#### Scenario 2: RAG Question Answering

**Test simple question:**
```
You: What are attention mechanisms?
Bot: [15-20s first time]
     *Answer:*
     Attention mechanisms allow models to dynamically focus on...

     📚 *Sources:*
     [1] *Attention Is All You Need*
         🔗 Read on arXiv - 1706.03762
         📊 Score: 12.456

     [2] *Neural Machine Translation...*
         🔗 Read on arXiv - 1409.0473
         📊 Score: 11.234

     ⚙️ Mode: hybrid
```

**Test cached query:**
```
You: What are attention mechanisms?
Bot: [~100ms second time ⚡]
     [Same answer as above]
     ⚙️ Mode: hybrid ⚡ Cached
```

#### Scenario 3: Search Command

**Test paper search:**
```
You: /search transformer neural networks
Bot: 📖 Found 145 papers (showing top 10)

     1. *Attention Is All You Need*
        🔗 Read on arXiv - 1706.03762
        📊 Score: 12.456

     2. *BERT: Pre-training of Deep Bidirectional...*
        🔗 Read on arXiv - 1810.04805
        📊 Score: 11.234
     [...]
```

#### Scenario 4: Settings Customization

**Test settings command:**
```
You: /settings
Bot: ⚙️ Your Settings

     *Search Mode:* HYBRID
     *Results per query:* 3 papers
     *Model:* llama3.2:1b
     *Categories:* All

     [Interactive buttons appear]:
     [🔍 Hybrid Search] [⚡ BM25 Only]
     [3 Results] [5 Results] [10 Results]
     [All Categories] [cs.AI] [cs.LG]
```

**Test changing settings:**
```
You: [Click "5 Results" button]
Bot: ✅ Results per query: 5

You: [Click "cs.AI" button]
Bot: ✅ Category filter: cs.AI
```

#### Scenario 5: Natural Conversation

**Test multi-turn conversation:**
```
You: Tell me about BERT
Bot: [Provides answer about BERT with sources]

You: How does it differ from GPT?
Bot: [Answers about BERT vs GPT differences]

You: /clear
Bot: 🗑️ Conversation history cleared!
     Your settings have been preserved.
```

#### Scenario 6: Error Handling

**Test invalid query:**
```
You: asdfghjkl
Bot: ❌ No relevant papers found.
     Try different keywords or check your category filters.
```

**Test when services are down:**
```
You: test query
Bot: ❌ Error processing your question:
     [User-friendly error message]
```

### Verification Checklist

- [ ] **Bot responds to `/start`** - Shows welcome message
- [ ] **Bot responds to all commands** - `/help`, `/ask`, `/search`, `/settings`, `/status`, `/clear`
- [ ] **Natural language queries work** - Can ask questions without commands
- [ ] **RAG answers include sources** - Papers cited with arXiv links
- [ ] **Cache works** - Repeated queries return instantly (⚡ indicator)
- [ ] **Settings persist** - Changes remain across conversation
- [ ] **Interactive buttons work** - Can click inline keyboard buttons
- [ ] **Long responses split correctly** - Messages don't exceed Telegram limit
- [ ] **Markdown formatting works** - Bold, italics, links render correctly
- [ ] **Langfuse shows traces** - Check http://localhost:3000 for Telegram events
- [ ] **Typing indicator shows** - "Bot is typing..." appears during processing
- [ ] **Error messages are friendly** - No stack traces exposed to user

### Performance Testing

**Cache Performance:**
```bash
# First query (cache miss)
You: What is machine learning?
Bot: [Response in ~15-20 seconds]

# Identical query (cache hit)
You: What is machine learning?
Bot: [Response in ~100ms ⚡]

# Verify 150-400x speedup!
```

**Concurrent Users:**
```bash
# Test with multiple Telegram accounts
# Each user should have independent settings and sessions
```

**Session Management:**
```bash
# Stop using bot for 30 minutes (default timeout)
# Verify session cleanup happens automatically
```

## Configuration

### Environment Variables

```bash
# Enable/Disable Bot
TELEGRAM__ENABLED=true  # Set to false to disable

# Bot Token (Required)
TELEGRAM__BOT_TOKEN=your_token_here

# Deployment Mode
TELEGRAM__USE_WEBHOOK=false  # true for production
TELEGRAM__WEBHOOK_URL=https://your-domain.com
TELEGRAM__WEBHOOK_PATH=/telegram/webhook

# Access Control (Optional)
TELEGRAM__ALLOWED_USER_IDS=123456789,987654321  # Empty = allow all

# Behavior Settings
TELEGRAM__MAX_MESSAGE_LENGTH=4000  # Telegram limit is 4096
TELEGRAM__ENABLE_STREAMING=true
TELEGRAM__SESSION_TIMEOUT_MINUTES=30
TELEGRAM__RATE_LIMIT_MESSAGES_PER_MINUTE=20

# Default User Preferences
TELEGRAM__DEFAULT_TOP_K=3
TELEGRAM__DEFAULT_USE_HYBRID=true
TELEGRAM__DEFAULT_MODEL=llama3.2:1b
```

### User Settings (Customizable per User)

Each user can customize:
- **Search mode**: Hybrid (BM25 + semantic) or BM25 only
- **Results count**: 3, 5, or 10 papers per query
- **Categories**: All, cs.AI, cs.LG, cs.CL, cs.CV, cs.NE
- **Model**: LLM model for answer generation
- **Display**: Show sources, compact mode, streaming

Settings persist across sessions and are stored in-memory (for production, consider Redis/PostgreSQL storage).

## Architecture Details

### Service Integration

The Telegram bot integrates with all existing Week 1-6 services:

1. **OpenSearch** - Hybrid search for relevant papers
2. **Jina Embeddings** - Semantic search capabilities
3. **Ollama LLM** - Answer generation
4. **Redis Cache** - 150-400x speedup for repeated queries
5. **Langfuse** - Complete tracing of Telegram interactions
6. **PostgreSQL** - Paper metadata (via OpenSearch)

### Message Flow

```python
# Simplified message flow
async def handle_message(update, context):
    1. Extract user_id, chat_id, text
    2. Check rate limits
    3. Get/create user settings
    4. Show "typing..." indicator
    5. Check cache (Redis)
    6. If cache hit → format and send
    7. If cache miss:
        a. Generate embedding (Jina)
        b. Search papers (OpenSearch)
        c. Build prompt with context
        d. Generate answer (Ollama)
        e. Cache result (Redis)
        f. Trace interaction (Langfuse)
    8. Format response (Markdown)
    9. Split if too long
    10. Send to Telegram
    11. Update conversation history
```

### Error Handling

The bot implements graceful degradation:

- **Markdown formatting fails** → Falls back to plain text
- **Embedding generation fails** → Falls back to BM25 search
- **Cache unavailable** → Continues without caching
- **Langfuse tracing fails** → Logs warning, continues
- **Service errors** → User-friendly error messages

### Rate Limiting

Built-in rate limiting protects against abuse:
- **Per-user limits**: 20 messages/minute (configurable)
- **Automatic throttling**: Enforced by Telegram library
- **Session timeouts**: 30 minutes of inactivity

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Bot doesn't respond** | Check `TELEGRAM__ENABLED=true` and valid `BOT_TOKEN` |
| **"Unauthorized" error** | Bot token is invalid, regenerate with @BotFather |
| **Bot responds but no answers** | Check OpenSearch, Ollama, and embeddings services |
| **Slow responses** | First query always slow, subsequent queries use cache |
| **Markdown formatting broken** | Bot automatically falls back to plain text |
| **Can't find bot** | Ensure bot username is correct, ends with "bot" |
| **"Forbidden" errors** | Check user_id restrictions in `ALLOWED_USER_IDS` |
| **Memory issues** | User sessions stored in-memory, monitor RAM usage |

### Debug Mode

Enable detailed logging:

```bash
# In .env
DEBUG=true

# Check bot logs
docker compose logs -f api | grep telegram
```

### Health Check

```bash
# Check if bot is running
curl http://localhost:8000/api/v1/health

# Should show telegram_service status
```

### Common Fixes

**Bot not starting:**
```bash
# 1. Check token is valid
# 2. Restart API service
docker compose restart api

# 3. Check logs for errors
docker compose logs api
```

**Responses too slow:**
```bash
# 1. Check cache is working
docker exec rag-redis redis-cli ping  # Should return PONG

# 2. Check cache hit rate
# Look for ⚡ indicator in bot responses

# 3. Verify Langfuse tracing not blocking
LANGFUSE__ENABLED=false  # Temporarily disable
```

## Performance Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| **First Query** | 15-20s | Full RAG pipeline execution |
| **Cached Query** | 50-100ms | **150-400x faster** |
| **Typing Indicator** | <500ms | Shows immediately |
| **Search Only** | 2-3s | `/search` command |
| **Status Check** | <1s | `/status` command |
| **Settings Update** | <100ms | Instant button response |
| **Concurrent Users** | 10+ | Tested simultaneously |

### Cache Hit Rates (Expected)

- **Repeated exact queries**: 100% hit rate
- **Popular questions**: 60-80% hit rate
- **Unique queries**: 0% hit rate (first time)

### Resource Usage

- **Memory**: ~50MB per active user session
- **CPU**: Minimal (<5% idle, spikes during generation)
- **Network**: ~10KB per message (excluding LLM generation)

## Production Deployment

### Webhook Mode (Recommended for Production)

```bash
# .env for production
TELEGRAM__USE_WEBHOOK=true
TELEGRAM__WEBHOOK_URL=https://your-domain.com
TELEGRAM__WEBHOOK_PATH=/telegram/webhook

# Requires:
# - HTTPS domain with valid certificate
# - Nginx/Caddy for TLS termination
# - Public IP or reverse proxy
```

### Security Best Practices

1. **Restrict Users**: Use `ALLOWED_USER_IDS` for private bots
2. **Rate Limiting**: Keep default limits (20 msgs/min)
3. **Environment Variables**: Never commit `.env` with real tokens
4. **HTTPS Only**: Use webhook mode in production
5. **Monitoring**: Track usage via Langfuse dashboard

### Scaling Considerations

For high-traffic deployments:

1. **User Storage**: Migrate from in-memory to Redis/PostgreSQL
2. **Cache**: Increase Redis memory allocation
3. **Load Balancing**: Multiple API instances (webhook mode)
4. **Rate Limits**: Adjust per your requirements
5. **Monitoring**: Set up alerts for errors and latency

## Next Steps

### Optional Enhancements (Week 7.1+)

- **📸 Image Support**: Upload paper PDFs, get summaries
- **🗣️ Voice Messages**: Ask questions via voice (speech-to-text)
- **📊 User Analytics**: Dashboard for usage patterns
- **🤝 Group Chat**: Multi-user discussions
- **🌍 Multi-Language**: Internationalization support
- **🔔 Notifications**: Push alerts for new papers in categories
- **📈 Personalization**: ML-based paper recommendations
- **🔗 Semantic Cache**: Fuzzy matching for similar queries

### Integration Ideas

- **Slack Bot**: Port to Slack with same architecture
- **Discord Bot**: Extend to Discord communities
- **WhatsApp Bot**: Use WhatsApp Business API
- **Web Widget**: Embed in websites
- **API Access**: Expose RESTful API for integrations

## Resources

- **Telegram Bot API**: https://core.telegram.org/bots/api
- **python-telegram-bot**: https://python-telegram-bot.org
- **@BotFather**: https://t.me/botfather
- **Langfuse Docs**: https://langfuse.com/docs
- **Redis Docs**: https://redis.io/docs

## Code Structure

```python
# Entry point: src/main.py
telegram_service = make_telegram_service(...)
await telegram_service.start()

# Service: src/services/telegram/client.py
class TelegramService:
    async def start() -> Start bot in polling/webhook mode
    async def stop() -> Stop bot gracefully
    async def health_check() -> Check bot status

# Handlers: src/services/telegram/handlers.py
class TelegramHandlers:
    async def start_command() -> /start
    async def help_command() -> /help
    async def ask_command() -> /ask
    async def search_command() -> /search
    async def settings_command() -> /settings
    async def handle_message() -> Regular text messages

# Formatters: src/services/telegram/formatters.py
format_rag_response() -> Rich Markdown formatting
format_search_results() -> Search result display
format_welcome_message() -> /start message
escape_markdown_v2() -> Telegram MarkdownV2 escaping
split_long_message() -> Auto-split >4000 chars
```

## Success Criteria

Week 7 is complete when:

- ✅ Bot responds to all commands
- ✅ Natural language queries return RAG answers
- ✅ Cache provides 150-400x speedup
- ✅ Settings persist across sessions
- ✅ Interactive keyboards work
- ✅ Langfuse shows Telegram traces
- ✅ Error handling is graceful
- ✅ Markdown formatting renders correctly
- ✅ Long messages split automatically
- ✅ Multiple users can use concurrently

---

**Week 7 transforms your RAG system into a mobile-first, conversational research assistant accessible anywhere via Telegram!** 🚀

---

## FAQ

**Q: Do I need a public IP for the Telegram bot?**
A: No! Polling mode works perfectly for development and low-traffic deployments. Webhook requires HTTPS.

**Q: Can multiple users use the bot simultaneously?**
A: Yes! Each user has independent settings and sessions.

**Q: How much does it cost to run the bot?**
A: Free! Telegram bot API is completely free with no limits.

**Q: Can I restrict the bot to specific users?**
A: Yes! Set `TELEGRAM__ALLOWED_USER_IDS=123456,789012` with comma-separated Telegram user IDs.

**Q: How do I get my Telegram user ID?**
A: Message `@userinfobot` on Telegram or check bot logs after sending a message.

**Q: Does the bot store conversation history?**
A: Yes, in-memory for the last 10 messages per user. For production, migrate to Redis/PostgreSQL.

**Q: Can I deploy this on a server without Docker?**
A: Yes! Just run `python src/main.py` after installing dependencies with `uv sync`.

**Q: How do I update the bot token?**
A: Update `TELEGRAM__BOT_TOKEN` in `.env` and restart: `docker compose restart api`

**Q: Can I customize the bot's personality?**
A: Yes! Edit prompts in `src/services/ollama/prompts/` and message templates in `src/services/telegram/formatters.py`.

**Q: Does this work with other LLM models?**
A: Yes! Change `OLLAMA_MODEL` or use `/settings` command to select different Ollama models.

---

Enjoy your new conversational RAG interface! 🎉
