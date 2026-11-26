import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import HTMLResponse
from app.config import settings
from app.routers import documents, qa, health

# Configure structured logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("academic_qa")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting up Academic Q&A RAG System...")
    logger.info(f"Configuration: Vector Store={settings.VECTOR_STORE_NAME}, Model={settings.MODEL_NAME}")
    yield
    logger.info("Shutting down Academic Q&A RAG System...")


# Initialize FastAPI application with custom docs endpoints to prevent CDN blocking
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan,
    docs_url=None,   # Handled custom below with cdnjs
    redoc_url=None   # Handled custom below
)

# Cross-Origin Resource Sharing (CORS) Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(qa.router)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    """Swagger UI using reliable Cloudflare CDN (bypasses ISP blocks on jsdelivr)."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{settings.API_TITLE} - Swagger UI",
        swagger_js_url="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui-bundle.min.js",
        swagger_css_url="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui.min.css"
    )


@app.get("/redoc", include_in_schema=False)
async def custom_redoc():
    """ReDoc documentation alternative."""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=f"{settings.API_TITLE} - ReDoc",
        redoc_js_url="https://cdnjs.cloudflare.com/ajax/libs/redoc/2.1.3/redoc.standalone.min.js"
    )


@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
@app.get("/ui", response_class=HTMLResponse, tags=["Web UI"])
async def interactive_web_ui() -> HTMLResponse:
    """Built-in interactive Academic Q&A Web Dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Academic Q&A RAG System</title>
  <style>
    :root {
      --primary: #2563eb;
      --primary-dark: #1d4ed8;
      --bg: #f8fafc;
      --card: #ffffff;
      --text: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --accent: #10b981;
      --accent-badge: #d1fae5;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }
    header { background: #1e293b; color: white; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    header h1 { font-size: 1.25rem; font-weight: 700; display: flex; align-items: center; gap: 0.5rem; }
    .badge { background: #10b981; color: white; padding: 0.25rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
    .header-links a { color: #94a3b8; text-decoration: none; margin-left: 1rem; font-size: 0.875rem; transition: color 0.2s; }
    .header-links a:hover { color: white; }
    .container { display: grid; grid-template-columns: 360px 1fr; gap: 1.5rem; max-width: 1400px; margin: 1.5rem auto; padding: 0 1.5rem; height: calc(100vh - 110px); }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); display: flex; flex-direction: column; overflow: hidden; }
    .card-header { padding: 1rem 1.25rem; border-bottom: 1px solid var(--border); font-weight: 600; font-size: 0.95rem; background: #ffffff; display: flex; justify-content: space-between; align-items: center; }
    .card-body { padding: 1.25rem; overflow-y: auto; flex: 1; }
    
    /* Upload form */
    .form-group { margin-bottom: 1rem; }
    .form-group label { display: block; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.35rem; color: #334155; }
    .form-group input[type="text"], .form-group input[type="file"] { width: 100%; padding: 0.55rem 0.75rem; border: 1px solid var(--border); border-radius: 0.5rem; font-size: 0.875rem; }
    .btn { background: var(--primary); color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 0.5rem; font-size: 0.875rem; font-weight: 600; cursor: pointer; transition: background 0.2s; width: 100%; display: flex; justify-content: center; align-items: center; gap: 0.5rem; }
    .btn:hover { background: var(--primary-dark); }
    .btn:disabled { background: #94a3b8; cursor: not-allowed; }
    .btn-secondary { background: #e2e8f0; color: #334155; width: auto; padding: 0.35rem 0.75rem; }
    .btn-secondary:hover { background: #cbd5e1; }

    /* Document list */
    .doc-list { list-style: none; margin-top: 1rem; }
    .doc-item { padding: 0.75rem; border: 1px solid var(--border); border-radius: 0.5rem; margin-bottom: 0.5rem; font-size: 0.85rem; background: #f8fafc; }
    .doc-item-title { font-weight: 600; color: #1e293b; margin-bottom: 0.25rem; word-break: break-all; }
    .doc-item-meta { font-size: 0.75rem; color: var(--text-muted); display: flex; justify-content: space-between; align-items: center; }
    .btn-delete { background: none; border: none; color: #ef4444; font-size: 0.75rem; cursor: pointer; }
    .btn-delete:hover { text-decoration: underline; }

    /* Chat Section */
    .chat-messages { flex: 1; overflow-y: auto; padding: 1.25rem; display: flex; flex-direction: column; gap: 1rem; }
    .message { max-width: 85%; padding: 1rem; border-radius: 0.75rem; font-size: 0.92rem; line-height: 1.6; }
    .message.user { align-self: flex-end; background: var(--primary); color: white; border-bottom-right-radius: 0.15rem; }
    .message.bot { align-self: flex-start; background: #ffffff; border: 1px solid var(--border); border-bottom-left-radius: 0.15rem; }
    .sources-box { margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px dashed var(--border); font-size: 0.8rem; color: #475569; }
    .source-tag { display: inline-block; background: #f1f5f9; padding: 0.2rem 0.5rem; border-radius: 0.35rem; margin: 0.2rem 0.2rem 0 0; font-weight: 500; }
    .confidence-badge { display: inline-flex; align-items: center; gap: 0.25rem; font-size: 0.75rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 9999px; background: #e0f2fe; color: #0369a1; margin-bottom: 0.5rem; }

    /* Input area */
    .chat-input-area { padding: 1rem 1.25rem; border-top: 1px solid var(--border); background: white; display: flex; gap: 0.75rem; }
    .chat-input { flex: 1; padding: 0.75rem 1rem; border: 1px solid var(--border); border-radius: 0.5rem; font-size: 0.92rem; outline: none; }
    .chat-input:focus { border-color: var(--primary); }
    .chat-send-btn { width: auto; padding: 0 1.5rem; }

    .loading-spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3); border-radius: 50%; border-top-color: white; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

<header>
  <h1>📚 Academic Q&A RAG System <span class="badge">Google Gemini 1.5</span></h1>
  <div class="header-links">
    <a href="/docs" target="_blank">Swagger UI</a>
    <a href="/redoc" target="_blank">ReDoc</a>
    <a href="/health" target="_blank">Health Status</a>
  </div>
</header>

<div class="container">
  <!-- Left Side: Ingestion & Document Management -->
  <div class="card">
    <div class="card-header">
      <span>📄 Ingest Academic Papers</span>
    </div>
    <div class="card-body">
      <form id="uploadForm">
        <div class="form-group">
          <label>Document Title</label>
          <input type="text" id="docTitle" placeholder="e.g. Attention Is All You Need" required />
        </div>
        <div class="form-group">
          <label>File (PDF, DOCX, TXT, CSV, XLSX, XLS)</label>
          <input type="file" id="docFile" accept=".pdf,.txt,.docx,.csv,.xlsx,.xls" required />
        </div>
        <button type="submit" class="btn" id="uploadBtn">
          <span id="uploadBtnText">Upload & Vectorize</span>
        </button>
      </form>
      <div id="uploadStatus" style="font-size:0.8rem; margin-top:0.6rem;"></div>

      <div style="margin-top: 1.5rem; border-top: 1px solid var(--border); padding-top: 1rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
          <strong style="font-size:0.85rem;">Indexed Documents</strong>
          <button class="btn-secondary" onclick="loadDocuments()">↻ Refresh</button>
        </div>
        <ul class="doc-list" id="docList">
          <li style="font-size:0.8rem; color:var(--text-muted);">No documents loaded yet.</li>
        </ul>
      </div>
    </div>
  </div>

  <!-- Right Side: Interactive Q&A Chat -->
  <div class="card">
    <div class="card-header">
      <span>💬 Academic Inquiry & Source Citations</span>
      <button class="btn-secondary" onclick="resetChat()">+ New Session</button>
    </div>
    <div class="chat-messages" id="chatBox">
      <div class="message bot">
        👋 Welcome to the <strong>Academic Q&A RAG System</strong>!
        <br><br>
        1. Upload an academic paper or coursework document on the left.
        <br>
        2. Ask any research question below to get answers cited with exact excerpts and page numbers.
      </div>
    </div>
    <div class="chat-input-area">
      <input type="text" id="queryInput" class="chat-input" placeholder="Type your research question..." onkeydown="if(event.key==='Enter') sendQuestion()" />
      <button class="btn chat-send-btn" id="sendBtn" onclick="sendQuestion()">
        <span id="sendBtnText">Ask</span>
      </button>
    </div>
  </div>
</div>

<script>
  let conversationId = null;

  async function loadDocuments() {
    try {
      const res = await fetch('/api/v1/documents/list');
      const docs = await res.json();
      const listEl = document.getElementById('docList');
      listEl.innerHTML = '';
      if (!docs || docs.length === 0) {
        listEl.innerHTML = '<li style="font-size:0.8rem; color:var(--text-muted);">No documents indexed yet.</li>';
        return;
      }
      docs.forEach(doc => {
        const li = document.createElement('li');
        li.className = 'doc-item';
        li.innerHTML = `
          <div class="doc-item-title">${escapeHtml(doc.title || doc.file_name)}</div>
          <div class="doc-item-meta">
            <span>${doc.chunk_count} chunks | ${(doc.size_bytes / 1024).toFixed(1)} KB</span>
            <button class="btn-delete" onclick="deleteDoc('${doc.doc_id}')">Delete</button>
          </div>
        `;
        listEl.appendChild(li);
      });
    } catch(err) {
      console.error(err);
    }
  }

  async function deleteDoc(docId) {
    if (!confirm('Are you sure you want to delete this document from the vector store?')) return;
    try {
      await fetch(`/api/v1/documents/${docId}`, { method: 'DELETE' });
      loadDocuments();
    } catch(err) {
      alert('Delete failed: ' + err);
    }
  }

  document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = document.getElementById('docFile').files[0];
    const title = document.getElementById('docTitle').value;
    const btn = document.getElementById('uploadBtn');
    const btnText = document.getElementById('uploadBtnText');
    const status = document.getElementById('uploadStatus');

    if (!file) return;

    btn.disabled = true;
    btnText.innerHTML = '<span class="loading-spinner"></span> Ingesting...';
    status.innerHTML = '<span style="color:#0284c7;">Extracting text & generating vector embeddings via Gemini...</span>';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', title);

    try {
      const res = await fetch('/api/v1/documents/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload failed');

      status.innerHTML = `<span style="color:#16a34a;">✓ Successfully indexed "${data.title}" (${data.chunk_count} chunks)!</span>`;
      document.getElementById('docFile').value = '';
      document.getElementById('docTitle').value = '';
      loadDocuments();
    } catch(err) {
      status.innerHTML = `<span style="color:#dc2626;">✗ Error: ${err.message}</span>`;
    } finally {
      btn.disabled = false;
      btnText.innerText = 'Upload & Vectorize';
    }
  });

  async function sendQuestion() {
    const input = document.getElementById('queryInput');
    const query = input.value.trim();
    if (!query) return;

    const chatBox = document.getElementById('chatBox');
    const sendBtn = document.getElementById('sendBtn');
    const sendBtnText = document.getElementById('sendBtnText');

    // Append user message
    const userMsg = document.createElement('div');
    userMsg.className = 'message user';
    userMsg.innerText = query;
    chatBox.appendChild(userMsg);
    input.value = '';
    chatBox.scrollTop = chatBox.scrollHeight;

    // Loading state
    sendBtn.disabled = true;
    sendBtnText.innerHTML = '<span class="loading-spinner"></span>';

    try {
      const res = await fetch('/api/v1/qa/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          conversation_id: conversationId,
          top_k: 4
        })
      });

      const data = await res.json();
      const botMsg = document.createElement('div');
      botMsg.className = 'message bot';

      if (!res.ok) {
        botMsg.innerHTML = `<span style="color:#dc2626;">⚠️ ${escapeHtml(data.detail || 'Error processing inquiry.')}</span>`;
      } else {
        const confPercent = Math.round(data.confidence_score * 100);
        let sourcesHtml = '';
        if (data.sources && data.sources.length > 0) {
          sourcesHtml = '<div class="sources-box"><strong>📚 Source Citations:</strong><br>';
          data.sources.forEach(s => {
            const pageStr = s.page_number ? ` (Page ${s.page_number})` : '';
            sourcesHtml += `<div class="source-tag">📄 ${escapeHtml(s.title)}${pageStr} - ${Math.round(s.relevance_score * 100)}% Match</div>`;
          });
          sourcesHtml += '</div>';
        }

        botMsg.innerHTML = `
          <div class="confidence-badge">🎯 Confidence: ${confPercent}% | ⏱️ ${data.processing_time_ms} ms ${data.cached ? '⚡ (Cached)' : ''}</div>
          <div>${escapeHtml(data.answer).replace(/\\n/g, '<br>')}</div>
          ${sourcesHtml}
        `;
      }

      chatBox.appendChild(botMsg);
      chatBox.scrollTop = chatBox.scrollHeight;
    } catch(err) {
      const errEl = document.createElement('div');
      errEl.className = 'message bot';
      errEl.innerHTML = `<span style="color:#dc2626;">Network error: ${err.message}</span>`;
      chatBox.appendChild(errEl);
    } finally {
      sendBtn.disabled = false;
      sendBtnText.innerText = 'Ask';
    }
  }

  function resetChat() {
    conversationId = 'conv_' + Math.random().toString(36).substr(2, 9);
    const chatBox = document.getElementById('chatBox');
    chatBox.innerHTML = '<div class="message bot">✨ Started a fresh conversation session. Ask your questions below!</div>';
  }

  function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.innerText = text;
    return div.innerHTML;
  }

  // Load document list on start
  loadDocuments();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
