"""
APPLUS3 API - FastAPI Entry Point
Provides REST endpoints for code generation with Knowledge Graph context

Endpoints:
- POST /generate - Run full generation pipeline
- GET /status/{run_id} - Check generation status
- GET /kg - Get Knowledge Graph state
- GET /health - Health check
"""

import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import (
    Orchestrator,
    KnowledgeGraph,
    GenerationRequest,
    GenerationResult
)
from workers import M0Worker, M1Worker, M2Worker


# =============================================================================
# Pydantic Models
# =============================================================================

class GenerateRequest(BaseModel):
    """Request to generate code."""
    idea: str = Field(..., min_length=10, description="The idea or requirement to generate code for")
    components: List[Dict[str, Any]] = Field(default=[], description="Optional explicit component list")
    config: Dict[str, Any] = Field(default={}, description="Generation configuration")


class GenerateResponse(BaseModel):
    """Response from generation."""
    request_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    """Status of a generation run."""
    request_id: str
    status: str
    progress: int
    stages_completed: List[str]
    files_generated: int
    error: Optional[str] = None


class KGStatsResponse(BaseModel):
    """Knowledge Graph statistics."""
    total_nodes: int
    total_edges: int
    node_types: int
    edge_types: int


class KGNodeResponse(BaseModel):
    """A node in the Knowledge Graph."""
    id: str
    type: str
    name: str
    file_path: str
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    app: str
    version: str
    timestamp: str


# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title="APPLUS3 API",
    description="Context-Aware Multi-IA Code Generation with Knowledge Graph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_orchestrator: Optional[Orchestrator] = None
_runs: Dict[str, Dict[str, Any]] = {}


def get_orchestrator() -> Orchestrator:
    """Get or create orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(
            kg_path="runs/knowledge_graph.json",
            runs_dir="runs"
        )
        # Register workers
        _orchestrator.register_worker('m0', lambda idea, kg: M0Worker(kg).analyze(idea).__dict__)
        _orchestrator.register_worker('m1', lambda ctx: M1Worker().generate(ctx['component'], ctx.get('kg_context', ''), ctx.get('previously_generated', {})).__dict__)
        _orchestrator.register_worker('m2', lambda files, cfg: M2Worker().generate(files, cfg).__dict__)
    return _orchestrator


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        app="APPLUS3",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Start code generation pipeline.

    The generation runs in the background. Use /status/{request_id} to check progress.
    """
    request_id = str(uuid.uuid4())[:8]

    # Initialize run state
    _runs[request_id] = {
        'status': 'pending',
        'progress': 0,
        'stages_completed': [],
        'files_generated': 0,
        'result': None,
        'error': None
    }

    # Run generation in background
    background_tasks.add_task(
        run_generation,
        request_id,
        request.idea,
        request.components,
        request.config
    )

    return GenerateResponse(
        request_id=request_id,
        status="started",
        message="Generation started. Check /status/{request_id} for progress."
    )


async def run_generation(
    request_id: str,
    idea: str,
    components: List[Dict[str, Any]],
    config: Dict[str, Any]
):
    """Background task for code generation."""
    try:
        _runs[request_id]['status'] = 'running'

        orchestrator = get_orchestrator()

        gen_request = GenerationRequest(
            id=request_id,
            idea=idea,
            components=components,
            config=config
        )

        result = orchestrator.run_pipeline(gen_request)

        # Update run state
        _runs[request_id]['status'] = 'completed' if result.success else 'failed'
        _runs[request_id]['progress'] = 100
        _runs[request_id]['stages_completed'] = [s.stage.value for s in result.stages]
        _runs[request_id]['files_generated'] = len(result.generated_files)
        _runs[request_id]['result'] = {
            'files': result.generated_files,
            'kg_stats': result.kg_stats,
            'review': result.review_result,
            'duration_ms': result.total_duration_ms
        }

    except Exception as e:
        _runs[request_id]['status'] = 'failed'
        _runs[request_id]['error'] = str(e)


@app.get("/status/{request_id}", response_model=StatusResponse)
async def get_status(request_id: str):
    """Get status of a generation run."""
    if request_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")

    run = _runs[request_id]
    return StatusResponse(
        request_id=request_id,
        status=run['status'],
        progress=run['progress'],
        stages_completed=run['stages_completed'],
        files_generated=run['files_generated'],
        error=run.get('error')
    )


@app.get("/result/{request_id}")
async def get_result(request_id: str):
    """Get full result of a completed generation run."""
    if request_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")

    run = _runs[request_id]
    if run['status'] not in ['completed', 'failed']:
        raise HTTPException(status_code=400, detail="Run not yet completed")

    return {
        'request_id': request_id,
        'status': run['status'],
        'result': run.get('result'),
        'error': run.get('error')
    }


@app.get("/kg/stats", response_model=KGStatsResponse)
async def get_kg_stats():
    """Get Knowledge Graph statistics."""
    orchestrator = get_orchestrator()
    stats = orchestrator.kg.stats()
    return KGStatsResponse(**stats)


@app.get("/kg/nodes", response_model=List[KGNodeResponse])
async def get_kg_nodes(node_type: Optional[str] = None):
    """Get Knowledge Graph nodes, optionally filtered by type."""
    orchestrator = get_orchestrator()

    if node_type:
        nodes = orchestrator.kg.query_by_type(node_type)
    else:
        nodes = orchestrator.kg.get_all_nodes()

    return [
        KGNodeResponse(
            id=n.id,
            type=n.type,
            name=n.name,
            file_path=n.file_path,
            metadata=n.metadata
        )
        for n in nodes
    ]


@app.delete("/kg/clear")
async def clear_kg():
    """Clear the Knowledge Graph."""
    orchestrator = get_orchestrator()
    orchestrator.kg.clear()
    orchestrator.kg.save()
    return {"status": "cleared", "message": "Knowledge Graph cleared"}


@app.get("/runs")
async def list_runs():
    """List all generation runs."""
    return {
        'total': len(_runs),
        'runs': [
            {
                'request_id': rid,
                'status': run['status'],
                'files_generated': run['files_generated']
            }
            for rid, run in _runs.items()
        ]
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
