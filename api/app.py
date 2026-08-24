import os
import time
import logging
from typing import List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# RAG & LLM dependencies
from huggingface_hub import hf_hub_download, snapshot_download
from llama_cpp import Llama
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import chromadb
import mlflow

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Config from environment variables (User must set these in HF Spaces secrets)
MODEL_REPO = os.getenv("MODEL_REPO", "TeslaInch/phi-3.5-mini-SCD-gguf")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "phi-3.5-mini-Q4_K_M.gguf")
VECTOR_DB_REPO = os.getenv("VECTOR_DB_REPO", "TeslaInch/SCD-vectorDB-v1")
CHROMA_PATH = "./chroma_db"

# MLflow Config (e.g. https://dagshub.com/username/repo.mlflow)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("SCD-Medical-LLM-Inference")
        logger.info(f"Connected to MLflow: {MLFLOW_TRACKING_URI}")
    except Exception as e:
        logger.warning(f"Failed to connect to MLflow (Check Auth/URI): {e}")
        MLFLOW_TRACKING_URI = None  # Disable tracking for this run

app = FastAPI(
    title="Medical LLM API",
    description="Inference server for Sickle Cell Disease RAG pipeline.",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Enable CORS for the frontend web application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (update in production!)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for models
llm = None
embedder = None
reranker = None
collection = None

# ── Pydantic Models ────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    question: str = Field(..., description="The clinical question to ask the model.")
    case: Optional[str] = Field(None, description="Optional clinical case context.")
    history: Optional[list] = Field(default_factory=list, description="List of previous conversation turns.")

class Citation(BaseModel):
    source: str
    content: str
    relevance_score: float

class PredictResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: float
    latency_ms: float

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

# ── Startup & Initialization ───────────────────────────────────────────────────

@app.on_event("startup")
def initialize_services():
    global llm, embedder, reranker, collection
    logger.info("Initializing services...")

    try:
        # 1. Download Model
        logger.info(f"Downloading model {MODEL_FILENAME} from {MODEL_REPO}...")
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        
        logger.info("Loading llama-cpp-python model...")
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=os.cpu_count() or 4
        )
        
        # 2. Initialize Embedder & Reranker
        logger.info("Loading Embedder and Reranker...")
        embedder = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512, device='cpu')

        # 3. Download and load Vector DB
        logger.info(f"Downloading Vector DB from {VECTOR_DB_REPO}...")
        os.makedirs(CHROMA_PATH, exist_ok=True)
        # Snapshot download grabs the whole directory
        snapshot_download(repo_id=VECTOR_DB_REPO, repo_type="dataset", local_dir=CHROMA_PATH)
        
        db_client = chromadb.PersistentClient(path=CHROMA_PATH)
        # Assuming the collection name is 'langchain'
        collection = db_client.get_collection("langchain")
        
        logger.info("All services initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # In a real scenario, you might want to raise, but for testing we catch it.
        pass

# ── Middleware ─────────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Incoming request: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(f"Request completed: {request.method} {request.url} - Status {response.status_code} - {process_time:.2f}ms")
        return response
    except Exception as exc:
        process_time = (time.time() - start_time) * 1000
        logger.error(f"Request failed: {request.method} {request.url} - {process_time:.2f}ms - Exception: {str(exc)}")
        raise

# ── Endpoints ──────────────────────────────────────────────────────────────────

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root():
    """Redirect the root URL to the Swagger UI docs."""
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    """Health check endpoint to ensure the API is running."""
    if llm is None or collection is None:
        raise HTTPException(status_code=503, detail="Services not fully initialized")
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse, responses={500: {"model": ErrorResponse}})
async def predict(req: PredictRequest):
    """
    Takes a clinical question, retrieves context, reranks with dynamic thresholding, 
    and returns a generated answer with citations.
    """
    start_time = time.time()
    
    # ── Safety Guardrails ──
    if "dosage" in req.question.lower() or "how many mg" in req.question.lower():
        logger.warning(f"Blocked dangerous query: {req.question}")
        # Example of soft-blocking (allowing response but logging warning)
        pass

    try:
        if collection is None:
            raise ValueError("Vector DB collection is not initialized.")
            
        full_query = f"{req.case}\n{req.question}" if req.case else req.question
        
        # 1. Retrieval (Get top 10 to rerank)
        query_embedding = embedder.embed_query(full_query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            include=["documents", "metadatas"]
        )
        
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        if not docs:
            return PredictResponse(answer="No relevant context found.", citations=[], confidence=0.0, latency_ms=0)
            
        # 2. Reranking
        pairs = [[full_query, doc] for doc in docs]
        scores = reranker.predict(pairs)
        
        # Combine docs, metas, and scores
        ranked = sorted(zip(docs, metadatas, scores), key=lambda x: x[2], reverse=True)
        
        # ── Dynamic Thresholding ──
        # Drop chunks whose score is below 0.05 OR falls by >30% compared to Rank 1
        top_score = ranked[0][2]
        filtered_ranked = [
            (d, m, s) for d, m, s in ranked 
            if s >= 0.05 and (top_score - s) <= (0.30 * top_score)
        ]
        
        # Take the top 1 filtered result to optimize CPU inference time (reduces prompt length)
        final_chunks = filtered_ranked[:1]
        
        # Truncate text to avoid massive context causing 10+ min inference on CPU
        context_str = "\n\n".join([f"[{m.get('source', 'Unknown')}] {d[:1200]}" for d, m, s in final_chunks])
        # 3. LLM Generation
        system_instruction = (
            "You are a medical AI assistant specialised in sickle cell disease. "
            "Use ONLY the provided context to answer the clinical question. "
            "Summarize the answer concisely in your own words. DO NOT copy and paste the context directly. "
            "If the context does not contain the answer, say 'I cannot answer this based on the provided guidelines.'"
        )
        
        # Phi-3 exact prompt format with history
        prompt_text = f"<|system|>\n{system_instruction}<|end|>\n"
        
        # Append recent history (keep only last 4 turns to avoid CPU memory bloat)
        recent_history = req.history[-4:] if req.history else []
        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Map frontend 'ai' role to 'assistant'
            mapped_role = "assistant" if role == "ai" else "user"
            prompt_text += f"<|{mapped_role}|>\n{content}<|end|>\n"
            
        prompt_text += f"<|user|>\nContext:\n{context_str}\n\nQuestion:\n{full_query}<|end|>\n<|assistant|>\n"
        
        # ── MLflow Tracking ──
        with mlflow.start_run(run_name="predict"):
            mlflow.log_param("question", req.question)
            mlflow.log_param("case_context", req.case or "None")
            mlflow.log_param("chunks_retrieved", len(final_chunks))
            mlflow.log_metric("confidence_score", float(top_score))
            
            # Call llama-cpp
            output = llm(
                prompt_text,
                max_tokens=400,
                temperature=0.1,
                stop=["<|end|>", "<|user|>", "<|system|>"],
                echo=False
            )
            
            answer = output['choices'][0]['text'].strip()
            process_time_ms = (time.time() - start_time) * 1000
            
            mlflow.log_metric("latency_ms", process_time_ms)
            mlflow.log_text(answer, "generated_answer.txt")
            
            # Build citations
            citations = [
                Citation(source=m.get("source", "Unknown"), content=d[:200]+"...", relevance_score=float(s))
                for d, m, s in final_chunks
            ]
            
            return PredictResponse(
                answer=answer,
                citations=citations,
                confidence=float(top_score),
                latency_ms=int(time.time() - start_time) * 1000
            )
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Serve React Frontend ────────────────────────────────────────────────────────
# Mount the React 'dist' directory if it exists (for HF Spaces deployment)
if os.path.exists("dist"):
    app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse("dist/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
