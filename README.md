# SIGGRAPH 2025 RAG Search Application

A full-stack Retrieval-Augmented Generation (RAG) application for searching and querying SIGGRAPH 2025 research papers. This project combines a Next.js frontend with a FastAPI backend to provide an intelligent search interface powered by AI.

## 🎯 Project Overview

Build a production-ready RAG application that allows users to:
- Search through 11,000+ SIGGRAPH 2025 paper chunks
- Get AI-generated answers with inline citations
- View source papers with links to PDFs, GitHub repos, and videos
- Experience real-time streaming responses

## 🏗️ Architecture

This is a **full-stack application** with separate frontend and backend:

-   **Frontend**: Next.js app running on `http://localhost:3000`
    - Modern React UI with Tailwind CSS
    - Real-time streaming via Server-Sent Events (SSE)
    - Responsive design with progress indicators

-   **Backend**: FastAPI server running on `http://localhost:8082`
    - RESTful API with SSE streaming support
    - Hybrid search (semantic + keyword)
    - AI-powered answer generation with citations

The frontend communicates with the backend via HTTP API calls.

## Dataset 
Chunked dataset in json form: https://drive.google.com/drive/folders/1-NaRLrjlMMW56ATTwXB5FYV84EjTOALT

## Prerequisites

-   Python 3.8+
-   Node.js 18+ and npm
-   Git
-   GitHub account

## 🚀 Getting Started

### Step 0: Fork This Repository

**IMPORTANT**: You must fork this repository to your own GitHub account to work on the assignment.

1. **Click the "Fork" button** at the top right of this repository
2. **Select your GitHub account** as the destination
3. **Clone your forked repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/siggraph-rag.git
   cd siggraph-rag
   ```

You will be working in **your fork**, not the original repository. This allows you to:
- ✅ Push your code changes
- ✅ Track your progress with commits
- ✅ Deploy your own version
- ✅ Submit your work via your repository link

---

## 1. Setup Instructions

Follow these steps to set up your local environment.

### a. Navigate to the Backend Directory

```bash
cd backend
```

### b. Create and Activate a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate
```

### c. Install Dependencies

Install the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### d. Set Up Environment Variables

You will need to provide API keys for the services used by the RAG pipeline. **All models (embeddings and LLMs) are accessed via OpenRouter API** for simplicity and cost-effectiveness.

Create a `.env` file in the `backend` directory:

```bash
cp .env.example .env
```

Now, edit the `.env` file and add your API keys:

```bash
# Required: OpenRouter API (for all models)
OPENROUTER_API_KEY="sk-or-v1-..."

# Required: Qdrant Cloud
QDRANT_URL="https://your-cluster.qdrant.io"
QDRANT_API_KEY="your-qdrant-api-key"

# Model Configuration
EMBEDDING_MODEL="text-embedding-3-large"
LLM_MODEL="gpt-4-turbo-preview"

# Reranker Configuration (choose one)
RERANKER_TYPE="cohere"  
COHERE_API_KEY="your-cohere-key"
```

**Get Your API Keys:**
- **Required**: OpenRouter: https://openrouter.ai/keys
- **Required**: Qdrant Cloud: https://cloud.qdrant.io
- **Optional**: Cohere (only if using Cohere reranker): https://dashboard.cohere.com/api-keys

**Reranker Options:**
- `cross-encoder` (default): Local model, no API key needed, ~400MB download on first use
- `cohere`: Cloud API, requires API key, faster for large batches
- `none`: Disable reranking

## 2. Running the Backend Server

Once the setup is complete, you can run the FastAPI server.

```bash
python api_server.py
```

The server will start on `http://localhost:8082`.

```
╔════════════════════════════════════════════════════════════════╗
║           SIGGRAPH 2025 RAG API Server                        ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Starting server on http://localhost:8082                      ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

## 3. API Endpoints

This server exposes the following key endpoints for the frontend:

-   `GET /health`: Health check to verify the server is running.
-   `GET /api/info`: Provides information about the API.
-   `GET /api/stream`: The main endpoint for handling RAG queries via Server-Sent Events (SSE).
-   `POST /api/query`: An optional non-streaming endpoint for queries.

API documentation is automatically generated by FastAPI and can be viewed at `http://localhost:8082/docs`.

## 4. Integrating with the Next.js Frontend

For the frontend to communicate with this backend, you must configure its environment.

1.  **Navigate** to the root of the frontend project .
2.  **Create** a file named `.env.local` if it doesn't exist.
3.  **Add** the following line to specify the backend API URL:

    ```
    # week3_assignment/.env.local
    NEXT_PUBLIC_API_URL=http://localhost:8082
    ```

4.  **Run** the frontend development server from the `week3_assignment` directory:

    ```bash
    npm install
    npm run dev
    ```

Now, when you access the frontend at `http://localhost:3000`, it will make API calls to your running backend at `http://localhost:8082`.

## 5. Development and Testing

This project includes a mock RAG implementation (`test_backend_integration.py`) that allows you to test the API integration without needing a fully functional RAG pipeline. The `api_server.py` is currently configured to use this mock.

To implement the actual RAG logic, you will need to:

1.  Create `rag_generate.py` and `retrieval_pipeline.py`.
2.  Update `api_server.py` to import `RAGGenerator` from `rag_generate` instead of `test_backend_integration`.

---

## 6. Building the RAG Pipeline (Student Implementation)

This section describes what you need to implement to make the RAG application work properly. The `api_server.py` is already complete and handles all API routing and SSE streaming. **Your focus is on building the RAG logic.**

### Required Files

You need to create two main files in the `backend/` directory:

1.  **`rag_generate.py`** - Main RAG orchestration and answer generation
2.  **`retrieval_pipeline.py`** - Document retrieval and ranking

### File 1: `retrieval_pipeline.py`

This file handles retrieving relevant document chunks from your knowledge base.

**Required Classes and Methods:**

```python
class RetrievalPipeline:
    """
    Handles hybrid retrieval: semantic search (via OpenRouter + Qdrant Cloud) + keyword search (BM25).
    """
    
    def __init__(self, config: Optional[RetrievalPipelineConfig] = None):
        """
        Initialize the retrieval pipeline.
        
        Args:
            config: Optional configuration object. If None, loads from environment variables.
        
        Environment variables required (if config is None):
            - OPENROUTER_API_KEY: OpenRouter API key for embeddings
            - QDRANT_URL: Qdrant Cloud URL
            - QDRANT_API_KEY: Qdrant Cloud API key
            - CHUNKS_PATH: Path to chunks.json
            - RERANKER_TYPE: "cross-encoder", "cohere", or "none"
            - CROSS_ENCODER_MODEL: Model name for local reranking (if using cross-encoder)
            - COHERE_API_KEY: Cohere API key (if using cohere reranker)
        """
        pass
    
    def retrieve(self, query: str, top_k: int = 8) -> List[SearchResult]:
        """
        Retrieve the most relevant document chunks for a query.
        
        Args:
            query: User's search query
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects with paper_id, chunk_text, score, and metadata
        """
        pass

class SearchResult:
    """
    Represents a single search result.
    """
    paper_id: str          # Unique paper identifier
    chunk_text: str        # The actual text content
    score: float           # Relevance score
    metadata: dict         # Paper metadata (title, authors, pdf_url, etc.)
```

**What You Need to Implement:**

-   **Semantic Search**: Use OpenRouter API for embeddings + Qdrant Cloud for vector search
-   **Keyword Search**: Implement BM25 for exact keyword matching (runs locally)
-   **Hybrid Fusion**: Combine semantic and keyword results using weighted scoring
-   **Reranking**: Use local cross-encoder model (default) or Cohere API (optional)


**Architecture:**
- **Embeddings**: OpenRouter API -> bge-large-en-v1.5
- **Vector DB**: Qdrant Cloud (no local storage needed)
- **BM25**: Local in-memory index
- **Reranking**: Cohere API

---

### File 2: `rag_generate.py`

This file orchestrates the RAG pipeline and generates answers using an LLM.

**What You Need to Implement:**

1.  **LLM Integration**: Connect to OpenAI or Anthropic API
2.  **Query Refinement** (Optional): Use LLM to improve search queries
3.  **Context Formatting**: Format retrieved chunks into a prompt
4.  **Answer Generation**: Call LLM with context and query
5.  **Source Extraction**: Build metadata for citations
6.  **Streaming Support**: The `api_server.py` handles streaming by calling your methods

**Key Dependencies:**

```python
# All models accessed via OpenRouter API
import requests  # For OpenRouter API calls
from retrieval_pipeline import RetrievalPipeline, SearchResult
```

**OpenRouter Integration:**

The RAG pipeline uses OpenRouter API for both embeddings and LLM generation. 

Example LLM call via OpenRouter:
```python
import requests

headers = {
    "Authorization": f"Bearer {openrouter_api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "openai/gpt-4-turbo-preview",  # or "anthropic/claude-3-opus"
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Your question here"}
    ],
    "stream": True  # For streaming responses
}

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers=headers,
    json=payload,
    stream=True
)
```

---

### Data Files You'll Need

Your RAG pipeline will need access to:

1.  **`chunks.json`**: Contains paper metadata and text chunks
    -   Format: List of objects with `paper_id`, `chunk_text`, `metadata` (title, authors, urls)
    
2.  **Vector Database**: Pre-computed embeddings (e.g., Qdrant, FAISS)
    -   Should contain embeddings for all chunks in `chunks.json`
    -   Indexed by `paper_id` or chunk ID

---

### Integration with `api_server.py`

Once you've implemented `rag_generate.py` and `retrieval_pipeline.py`, update the import in `api_server.py`:

**Change line 36 from:**
```python
from test_backend_integration import RAGGenerator, GenerationConfig, SYSTEM_PROMPT
```

**To:**
```python
from rag_generate import RAGGenerator, GenerationConfig, SYSTEM_PROMPT
```

The `api_server.py` will automatically:
-   Initialize your `RAGGenerator` on startup
-   Call `refine_query()` for query refinement
-   Call `retrieval.retrieve()` for document search
-   Stream LLM responses via `llm_client.chat.completions.create(stream=True)`
-   Format SSE events for the frontend

---

### Testing Your Implementation

1.  **Test Retrieval First**:
    ```python
    from retrieval_pipeline import RetrievalPipeline
    
    # Loads configuration from environment variables (.env file)
    pipeline = RetrievalPipeline()
    results = pipeline.retrieve("What is 3D Gaussian Splatting?", top_k=5)
    print(results)
    ```

2.  **Test RAG Generation**:
    ```python
    from rag_generate import RAGGenerator, GenerationConfig
    
    config = GenerationConfig()
    rag = RAGGenerator(config)
    response = rag.generate("What is 3D Gaussian Splatting?")
    print(response)
    ```

3.  **Test Full API**:
    ```bash
    python api_server.py
    curl -N "http://localhost:8082/api/stream?query=test&top_k=5"
    ```

---

### Implementation Checklist

- [ ] Set up environment variables in `.env` file
- [ ] Get OpenRouter API key for embeddings and LLM
- [ ] Set up Qdrant Cloud cluster and get credentials
- [ ] Create `retrieval_pipeline.py` with `RetrievalPipeline` class
- [ ] Implement OpenRouter API integration for embeddings
- [ ] Implement Qdrant Cloud connection for vector search
- [ ] Implement keyword search (BM25) - runs locally
- [ ] Implement hybrid result fusion and ranking
- [ ] Set up reranker (cross-encoder local or Cohere API)
- [ ] Create `rag_generate.py` with `RAGGenerator` class
- [ ] Integrate OpenRouter API for LLM generation
- [ ] Implement context formatting
- [ ] Implement streaming answer generation
- [ ] Extract and format source metadata
- [ ] Update `api_server.py` import statement
- [ ] Test with `curl` commands
- [ ] Test with frontend UI

---

### Tips for Success

-   **Start Simple**: Get basic retrieval working first, then add reranking
-   **Test Incrementally**: Test each component independently before integration
-   **Use Mock Data**: Create a small test dataset to iterate quickly
-   **Check API Contract**: Ensure your return formats match what `api_server.py` expects
-   **Monitor Costs**: LLM API calls cost money - use smaller models for testing
-   **Handle Errors**: Add try-except blocks and meaningful error messages

---

## 7. Deployment Guide

This section covers deploying your full-stack RAG application to production using free hosting platforms.

### Deployment Architecture

- **Frontend**: Deploy to Vercel (free tier)
- **Backend**: Deploy to Render (free tier)
- **Vector Database**: Qdrant Cloud (free tier)
- **APIs**: OpenRouter (pay-as-you-go)

### Prerequisites for Deployment

Before deploying, ensure you have:

1. ✅ Working local development setup
2. ✅ All API keys configured in `.env`
3. ✅ GitHub account for code hosting
4. ✅ Vercel account (sign up at https://vercel.com)
5. ✅ Render account (sign up at https://render.com)

---

### Step 1: Prepare Your Code for Deployment

#### a. Create `.gitignore` File

Create a `.gitignore` file in your project root to exclude sensitive files:

```gitignore
# Dependencies
node_modules/
.next/
venv/
__pycache__/

# Environment variables
.env
.env.local
.env*.local

# Build outputs
/build
/dist
/.next/
/out/

# OS files
.DS_Store
*.log

# IDE
.vscode/
.idea/
```

#### b. Update Backend for Production

Ensure your `api_server.py` reads the port from environment variables (already configured):

```python
# This is already in your api_server.py
port = int(os.getenv("PORT", 8082))
```
---

### Step 2: Push Code to GitHub

1. **Initialize Git Repository**:
   ```bash
   cd /path/to/your/project
   git init
   git add .
   git commit -m "Initial commit: SIGGRAPH RAG application"
   ```

2. **Create GitHub Repository**:
   - Go to https://github.com/new
   - Create a new repository (e.g., `siggraph-rag`)
   - Don't initialize with README (you already have one)

3. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/siggraph-rag.git
   git branch -M main
   git push -u origin main
   ```

---

### Step 3: Deploy Backend to Render

#### a. Create New Web Service

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure the service:

   - **Name**: `siggraph-rag-backend`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`

#### b. Configure Environment Variables

In the Render dashboard, add these environment variables:

```
OPENROUTER_API_KEY=sk-or-v1-...
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-4-turbo-preview
RERANKER_TYPE=cohere
COHERE_API_KEY=your-cohere-key
```

**Memory Optimization Tips**:
- Use `RERANKER_TYPE=cohere` instead of `cross-encoder` to avoid downloading large models
- If you must use local reranking, use smaller models like `cross-encoder/ms-marco-TinyBERT-L-2-v2`

#### c. Deploy

1. Click **"Create Web Service"**
2. Wait 5-10 minutes for deployment
3. Your backend will be live at: `https://siggraph-rag-backend.onrender.com`

#### d. Verify Deployment

Test your backend:
```bash
curl https://siggraph-rag-backend.onrender.com/health
```

Expected response:
```json
{"status":"healthy","rag_initialized":true,"timestamp":1234567890}
```

**Important**: Render's free tier spins down after 15 minutes of inactivity. First request after sleep takes ~30 seconds to wake up.

---

### Step 4: Deploy Frontend to Vercel

#### a. Install Vercel CLI

```bash
npm install -g vercel
```

#### b. Navigate to Frontend Directory

```bash
cd frontend
```

#### c. Configure Environment Variable

Create `.env.production` file:

```bash
NEXT_PUBLIC_API_URL=https://siggraph-rag-backend.onrender.com
```

#### d. Deploy to Vercel

```bash
vercel login
vercel --prod
```

Follow the prompts

#### e. Add Environment Variable in Vercel Dashboard

1. Go to https://vercel.com/dashboard
2. Select your project
3. Go to **Settings** → **Environment Variables**
4. Add:
   - **Key**: `NEXT_PUBLIC_API_URL`
   - **Value**: `https://siggraph-rag-backend.onrender.com`
   - **Environments**: Production, Preview, Development

5. Redeploy:
   ```bash
   vercel --prod
   ```

Your frontend will be live at: `https://siggraph-rag.vercel.app`

---

### Step 5: Test Full-Stack Deployment

1. **Open your frontend URL** in a browser
2. **Submit a test query**: "What is 3D Gaussian Splatting?"
3. **Verify**:
   - ✅ Connection established
   - ✅ Progress updates appear
   - ✅ Answer streams in real-time
   - ✅ Source papers displayed with links

---

### Troubleshooting Deployment Issues

#### Backend Issues

**Problem**: Backend returns 404
- **Solution**: Check `Start Command` is correct: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`

**Problem**: Backend crashes on startup
- **Solution**: Check logs in Render dashboard, verify all environment variables are set

**Problem**: "No open ports detected"
- **Solution**: Ensure your server binds to `0.0.0.0` and uses `$PORT` environment variable

#### Frontend Issues

**Problem**: "Connection error" in browser
- **Solution**: Verify `NEXT_PUBLIC_API_URL` is set correctly in Vercel environment variables

**Problem**: Frontend shows old version
- **Solution**: Clear Vercel cache and redeploy: `vercel --prod --force`

**Problem**: CORS errors
- **Solution**: Verify backend has CORS middleware configured (already in `api_server.py`)

#### General Issues

**Problem**: Slow first request (30+ seconds)
- **Solution**: Normal for Render free tier - backend is waking up from sleep

**Problem**: API costs too high
- **Solution**: Use cheaper models for testing (e.g., `gpt-3.5-turbo` instead of `gpt-4`)

---

### Cost Optimization

#### Free Tier Limits

- **Vercel**: Unlimited deployments, 100GB bandwidth/month
- **Render**: 750 hours/month (enough for one service), spins down after 15min inactivity
- **Qdrant Cloud**: 1GB storage, 1M vectors
- **OpenRouter**: Pay-as-you-go (no free tier)

#### Reducing API Costs

1. **Use cheaper models for development**:
   ```
   LLM_MODEL=gpt-3.5-turbo  # ~$0.001/1K tokens vs $0.01/1K for GPT-4
   EMBEDDING_MODEL=text-embedding-3-small  # Cheaper than large
   ```

2. **Reduce retrieval size**:
   ```python
   retrieval_top_k=5  # Instead of 8
   ```

3. **Disable query refinement for testing**:
   ```python
   refine_query=False
   ```

4. **Cache common queries** (advanced):
   - Implement Redis caching for frequent questions

---

### Monitoring and Maintenance

#### Check Backend Logs

Render Dashboard → Your Service → Logs

#### Check Frontend Logs

Vercel Dashboard → Your Project → Deployments → View Logs

#### Monitor API Usage

- OpenRouter: https://openrouter.ai/activity
- Cohere: https://dashboard.cohere.com/billing
- Qdrant: https://cloud.qdrant.io

#### Update Deployment

When you make code changes:

```bash
# Push to GitHub
git add .
git commit -m "Your changes"
git push origin main

# Render will auto-deploy backend
# Redeploy frontend
cd frontend
vercel --prod
```

---

### Production Best Practices

1. **Environment Variables**: Never commit API keys to Git
2. **Error Handling**: Add comprehensive error messages for debugging
3. **Rate Limiting**: Implement rate limiting to prevent API abuse
4. **Monitoring**: Set up uptime monitoring (e.g., UptimeRobot)
5. **Backups**: Regularly backup your Qdrant database
6. **Documentation**: Keep README updated with deployment changes
7. **Testing**: Test on staging environment before production deployment

---

## 8. Project Structure

Your final project structure should look like this:

```
siggraph-rag/
├── backend/
│   ├── api_server.py           # FastAPI server (provided)
│   ├── rag_generate.py         # RAG orchestration (you implement)
│   ├── retrieval_pipeline.py   # Retrieval logic (you implement)
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Environment variables (not in git)
│   ├── .env.example            # Example env file
│   └── chunks.json             # Paper data
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx        # Main page
│   │   │   └── layout.tsx      # Root layout
│   │   ├── components/
│   │   │   └── rag/            # RAG UI components
│   │   └── hooks/
│   │       ├── useRAGStream.ts # SSE hook (recommended)
│   │       └── useRAGWebSocket.ts # WebSocket hook
│   ├── package.json
│   ├── .env.local              # Local env (not in git)
│   └── .env.production         # Production env (not in git)
├── .gitignore
└── README.md
```

---

## 9. Additional Resources

### Documentation
- **FastAPI**: https://fastapi.tiangolo.com
- **Next.js**: https://nextjs.org/docs
- **Qdrant**: https://qdrant.tech/documentation
- **OpenRouter**: https://openrouter.ai/docs

### API References
- **OpenAI API**: https://platform.openai.com/docs/api-reference
- **Cohere Rerank**: https://docs.cohere.com/reference/rerank
- **Qdrant Client**: https://qdrant.tech/documentation/interfaces

### Learning Resources
- **RAG Tutorial**: https://www.pinecone.io/learn/retrieval-augmented-generation
- **Vector Search**: https://www.pinecone.io/learn/vector-search
- **Semantic Search**: https://www.sbert.net/examples/applications/semantic-search/README.html

### Community
- **FastAPI Discord**: https://discord.gg/fastapi
- **Next.js Discord**: https://discord.gg/nextjs
- **Qdrant Discord**: https://discord.gg/qdrant

---

## 10. Submission Checklist

Before submitting your project, ensure:

- [ ] Backend runs locally without errors
- [ ] Frontend runs locally and connects to backend
- [ ] All API endpoints return correct responses
- [ ] SSE streaming works properly
- [ ] Source citations appear correctly
- [ ] Code is well-documented with comments
- [ ] Environment variables are documented in `.env.example`
- [ ] `.gitignore` excludes sensitive files
- [ ] README is updated with your implementation notes
- [ ] Backend is deployed to Render (or alternative)
- [ ] Frontend is deployed to Vercel (or alternative)
- [ ] Deployed app is fully functional

---

## 11. Support

If you encounter issues:

1. **Check the logs**: Backend (Render) and Frontend (Vercel) dashboards
2. **Review this README**: Most common issues are covered
3. **Search GitHub Issues**: Check if others had similar problems
4. **Ask for help**: Post in circle or as your TA with:
   - Error message
   - What you tried
   - Relevant code snippets
   - Environment (local/deployed)

---

**Good luck building your RAG application! 🚀**
