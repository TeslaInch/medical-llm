import logging
import time
from typing import List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical LLM API",
    description="Inference server for Sickle Cell Disease RAG pipeline.",
    version="1.0.0"
)

# ── Pydantic Models ────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    question: str = Field(..., description="The clinical question to ask the model.")
    case: Optional[str] = Field(None, description="Optional clinical case context.")
    # We can add parameters like temperature, etc. later if needed.

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

@app.get("/health")
async def health_check():
    """Health check endpoint to ensure the API is running."""
    # In a real app, you would check DB connection and model loaded status here
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse, responses={500: {"model": ErrorResponse}})
async def predict(req: PredictRequest):
    """
    Takes a clinical question and optional case, retrieves context from the RAG pipeline, 
    and returns a generated answer with citations.
    """
    start_time = time.time()
    
    # ── Input Validation / Safety Guardrails ──
    # Example guardrail: block dosage calculations if strictly prohibited
    if "dosage" in req.question.lower() or "how many mg" in req.question.lower():
        logger.warning(f"Blocked dangerous query: {req.question}")
        # Depending on policy, we might still answer but append a disclaimer, or block.
        # For now, let's just log it.
        pass

    try:
        # TODO: Hook up the actual RAG pipeline here (VectorDB, Reranker, vLLM)
        # For Week 14 container setup, we mock the response.
        
        # Simulated processing delay
        time.sleep(0.5)
        
        mock_answer = "This is a mocked clinical answer. The RAG pipeline will be integrated here."
        mock_citations = [
            Citation(source="mock_guideline.pdf", content="Mocked context...", relevance_score=0.95)
        ]
        
        process_time_ms = (time.time() - start_time) * 1000
        
        return PredictResponse(
            answer=mock_answer,
            citations=mock_citations,
            confidence=0.85,
            latency_ms=process_time_ms
        )
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal inference error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
