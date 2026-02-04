"""
APPLUS3 Orchestrator Module
Central brain that sequences M0->M1->M2, passes KG context, validates outputs

The Orchestrator is the CEREBRO that orders M0/M1/M2 workers.
Knowledge Graph is the ÚNICA source of truth.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum

from .knowledge_graph import KnowledgeGraph, Node, Edge, EdgeType
from .retrieval import ContextRetriever, RetrievalResult
from .anchored_review import AnchoredReviewer, AnchoredReviewResult


class PipelineStage(str, Enum):
    """Stages in the generation pipeline."""
    M0_ANALYSIS = "M0_ANALYSIS"
    M1_GENERATION = "M1_GENERATION"
    M2_DEPLOYMENT = "M2_DEPLOYMENT"
    REVIEW = "REVIEW"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass
class PipelineResult:
    """Result from a single pipeline stage."""
    stage: PipelineStage
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: int = 0


@dataclass
class GenerationRequest:
    """Request for code generation."""
    id: str
    idea: str
    components: List[str] = field(default_factory=list)
    output_dir: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Complete result from generation pipeline."""
    request_id: str
    success: bool
    stages: List[PipelineResult] = field(default_factory=list)
    generated_files: Dict[str, str] = field(default_factory=dict)
    kg_stats: Dict[str, int] = field(default_factory=dict)
    total_duration_ms: int = 0
    review_result: Optional[Dict[str, Any]] = None


class Orchestrator:
    """
    Central orchestrator for APPLUS3 code generation pipeline.

    Coordinates:
    - M0 Worker: Idea analysis and component identification
    - M1 Worker: Code generation with Multi-IA consensus
    - M2 Worker: Deployment configuration generation

    Maintains Knowledge Graph as single source of truth.
    """

    def __init__(
        self,
        kg_path: str = "runs/knowledge_graph.json",
        runs_dir: str = "runs"
    ):
        """
        Initialize orchestrator.

        Args:
            kg_path: Path to Knowledge Graph storage
            runs_dir: Directory for run outputs
        """
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.kg = KnowledgeGraph(kg_path)
        self.retriever = ContextRetriever(self.kg)
        self.reviewer = AnchoredReviewer(self.kg)

        # Worker references (set externally or via register_worker)
        self._workers: Dict[str, Callable] = {}

        # Current run state
        self._current_run_id: Optional[str] = None
        self._current_run_dir: Optional[Path] = None

    def register_worker(self, stage: str, worker_fn: Callable) -> None:
        """
        Register a worker function for a pipeline stage.

        Args:
            stage: Stage name (m0, m1, m2)
            worker_fn: Worker function to call
        """
        self._workers[stage.lower()] = worker_fn

    def run_pipeline(self, request: GenerationRequest) -> GenerationResult:
        """
        Execute the full generation pipeline.

        Pipeline stages:
        1. M0: Analyze idea, identify components, populate KG
        2. M1: Generate code for each component with KG context
        3. M2: Generate deployment configs
        4. Review: Validate outputs with anchored review

        Args:
            request: Generation request with idea and config

        Returns:
            GenerationResult with all outputs and stats
        """
        start_time = time.time()
        stages: List[PipelineResult] = []
        generated_files: Dict[str, str] = {}

        # Initialize run
        self._current_run_id = request.id or str(uuid.uuid4())[:8]
        self._current_run_dir = self.runs_dir / f"run_{self._current_run_id}"
        self._current_run_dir.mkdir(parents=True, exist_ok=True)

        # Log request
        self._log_event("pipeline_start", {
            "request_id": self._current_run_id,
            "idea": request.idea[:200],
            "components": request.components
        })

        try:
            # Stage 1: M0 Analysis
            m0_result = self._run_m0(request)
            stages.append(m0_result)

            if not m0_result.success:
                return self._build_result(request, stages, generated_files, start_time, False)

            # Stage 2: M1 Generation
            components = m0_result.data.get('components', request.components)
            m1_result = self._run_m1(request, components)
            stages.append(m1_result)

            if m1_result.success:
                generated_files = m1_result.data.get('files', {})

            # Stage 3: M2 Deployment
            m2_result = self._run_m2(request, generated_files)
            stages.append(m2_result)

            if m2_result.success:
                generated_files.update(m2_result.data.get('files', {}))

            # Stage 4: Review
            review_result = self._run_review(generated_files)
            stages.append(review_result)

            # Save artifacts
            self._save_artifacts(generated_files)
            self.kg.save()

            return self._build_result(
                request, stages, generated_files, start_time,
                success=review_result.data.get('passed', False)
            )

        except Exception as e:
            self._log_event("pipeline_error", {"error": str(e)})
            stages.append(PipelineResult(
                stage=PipelineStage.FAILED,
                success=False,
                error=str(e)
            ))
            return self._build_result(request, stages, generated_files, start_time, False)

    def _run_m0(self, request: GenerationRequest) -> PipelineResult:
        """
        Run M0 analysis stage.

        Args:
            request: Generation request

        Returns:
            PipelineResult with identified components
        """
        start = time.time()
        self._log_event("m0_start", {"idea": request.idea[:100]})

        try:
            # Check for registered worker
            if 'm0' in self._workers:
                result = self._workers['m0'](request.idea, self.kg)
                components = result.get('components', [])
            else:
                # Default: parse components from request or infer
                components = request.components or self._infer_components(request.idea)

            # Populate KG with component nodes
            for comp in components:
                node = Node(
                    id=comp.get('id', comp.get('name', '').lower().replace(' ', '_')),
                    type="Module",
                    name=comp.get('name', comp.get('id', '')),
                    metadata={
                        'description': comp.get('description', ''),
                        'pattern': comp.get('pattern', ''),
                        'priority': comp.get('priority', 99)
                    },
                    file_path=comp.get('file', '')
                )
                self.kg.add_node(node)

            # Add dependency edges
            for comp in components:
                for dep in comp.get('depends_on', []):
                    edge = Edge(
                        source_id=comp.get('id', comp.get('name', '').lower().replace(' ', '_')),
                        target_id=dep,
                        edge_type=EdgeType.DEPENDS_ON.value
                    )
                    try:
                        self.kg.add_edge(edge)
                    except ValueError:
                        pass  # Ignore if dependency doesn't exist yet

            duration = int((time.time() - start) * 1000)
            self._log_event("m0_complete", {"components": len(components), "duration_ms": duration})

            return PipelineResult(
                stage=PipelineStage.M0_ANALYSIS,
                success=True,
                data={'components': components},
                duration_ms=duration
            )

        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.M0_ANALYSIS,
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start) * 1000)
            )

    def _run_m1(
        self,
        request: GenerationRequest,
        components: List[Dict[str, Any]]
    ) -> PipelineResult:
        """
        Run M1 generation stage.

        Args:
            request: Generation request
            components: Components to generate

        Returns:
            PipelineResult with generated files
        """
        start = time.time()
        generated_files: Dict[str, str] = {}

        self._log_event("m1_start", {"components": len(components)})

        try:
            # Sort by priority/dependency order
            sorted_components = sorted(
                components,
                key=lambda c: c.get('priority', 99)
            )

            for comp in sorted_components:
                comp_id = comp.get('id', comp.get('name', '').lower().replace(' ', '_'))

                # Get context from KG
                context = self.retriever.get_context_for_generation(comp_id)

                # Build generation context
                gen_context = {
                    'component': comp,
                    'kg_context': context.context_text,
                    'previously_generated': list(generated_files.keys()),
                    'idea': request.idea
                }

                # Generate code
                if 'm1' in self._workers:
                    result = self._workers['m1'](gen_context)
                    code = result.get('code', '')
                else:
                    code = self._generate_placeholder(comp)

                # Store result
                file_path = comp.get('file', f"{comp_id}.py")
                generated_files[file_path] = code

                # Update KG node with generated code info
                node = self.kg.get_node(comp_id)
                if node:
                    node.metadata['generated'] = True
                    node.metadata['code_lines'] = len(code.split('\n'))
                    self.kg.add_node(node)

                self._log_event("m1_component_generated", {
                    "component": comp_id,
                    "file": file_path,
                    "lines": len(code.split('\n'))
                })

            duration = int((time.time() - start) * 1000)
            self._log_event("m1_complete", {
                "files": len(generated_files),
                "duration_ms": duration
            })

            return PipelineResult(
                stage=PipelineStage.M1_GENERATION,
                success=True,
                data={'files': generated_files},
                duration_ms=duration
            )

        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.M1_GENERATION,
                success=False,
                data={'files': generated_files},
                error=str(e),
                duration_ms=int((time.time() - start) * 1000)
            )

    def _run_m2(
        self,
        request: GenerationRequest,
        generated_files: Dict[str, str]
    ) -> PipelineResult:
        """
        Run M2 deployment stage.

        Args:
            request: Generation request
            generated_files: Files generated by M1

        Returns:
            PipelineResult with deployment configs
        """
        start = time.time()
        deploy_files: Dict[str, str] = {}

        self._log_event("m2_start", {"input_files": len(generated_files)})

        try:
            if 'm2' in self._workers:
                result = self._workers['m2'](generated_files, request.config)
                deploy_files = result.get('files', {})
            else:
                # Default deployment files
                deploy_files = self._generate_default_deployment(generated_files)

            duration = int((time.time() - start) * 1000)
            self._log_event("m2_complete", {
                "files": len(deploy_files),
                "duration_ms": duration
            })

            return PipelineResult(
                stage=PipelineStage.M2_DEPLOYMENT,
                success=True,
                data={'files': deploy_files},
                duration_ms=duration
            )

        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.M2_DEPLOYMENT,
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start) * 1000)
            )

    def _run_review(self, generated_files: Dict[str, str]) -> PipelineResult:
        """
        Run anchored review on generated code.

        Args:
            generated_files: All generated files

        Returns:
            PipelineResult with review outcome
        """
        start = time.time()

        try:
            # Combine all code for review
            all_code = "\n\n".join([
                f"# === {path} ===\n{code}"
                for path, code in generated_files.items()
                if path.endswith('.py')
            ])

            review_result = self.reviewer.review(
                generated_code=all_code,
                context={'file_count': len(generated_files)}
            )

            duration = int((time.time() - start) * 1000)
            self._log_event("review_complete", {
                "passed": review_result.passed,
                "avg_score": review_result.avg_score,
                "duration_ms": duration
            })

            return PipelineResult(
                stage=PipelineStage.REVIEW,
                success=True,
                data={
                    'passed': review_result.passed,
                    'avg_score': review_result.avg_score,
                    'reviews': [
                        {'dimension': r.dimension.value, 'score': r.score}
                        for r in review_result.reviews
                    ],
                    'suggestions': review_result.suggestions
                },
                duration_ms=duration
            )

        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.REVIEW,
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start) * 1000)
            )

    def _build_result(
        self,
        request: GenerationRequest,
        stages: List[PipelineResult],
        generated_files: Dict[str, str],
        start_time: float,
        success: bool
    ) -> GenerationResult:
        """Build final generation result."""
        review_stage = next(
            (s for s in stages if s.stage == PipelineStage.REVIEW),
            None
        )

        return GenerationResult(
            request_id=self._current_run_id or request.id,
            success=success,
            stages=stages,
            generated_files=generated_files,
            kg_stats=self.kg.stats(),
            total_duration_ms=int((time.time() - start_time) * 1000),
            review_result=review_stage.data if review_stage else None
        )

    def _infer_components(self, idea: str) -> List[Dict[str, Any]]:
        """Infer components from idea text."""
        # Simple extraction (in production: use LLM)
        return [
            {'id': 'main', 'name': 'Main Module', 'file': 'main.py', 'priority': 1}
        ]

    def _generate_placeholder(self, component: Dict[str, Any]) -> str:
        """Generate placeholder code for a component."""
        name = component.get('name', 'Module')
        desc = component.get('description', '')
        return f'''"""
{name}
{desc}

Generated by APPLUS3 Orchestrator
"""

# TODO: Implement {name}

def main():
    """Entry point."""
    print("{name} initialized")

if __name__ == "__main__":
    main()
'''

    def _generate_default_deployment(
        self,
        generated_files: Dict[str, str]
    ) -> Dict[str, str]:
        """Generate default deployment configuration."""
        return {
            'requirements.txt': 'fastapi>=0.104.0\nuvicorn>=0.24.0\nrequests>=2.31.0\npydantic>=2.5.0\n',
            'Dockerfile': f'''FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        }

    def _save_artifacts(self, generated_files: Dict[str, str]) -> None:
        """Save generated files to run directory."""
        if not self._current_run_dir:
            return

        for file_path, content in generated_files.items():
            full_path = self._current_run_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log pipeline event."""
        if not self._current_run_dir:
            return

        log_path = self._current_run_dir / "events.jsonl"
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': event_type,
            **data
        }

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event) + '\n')


def run_pipeline(idea: str, **kwargs) -> GenerationResult:
    """
    Convenience function to run the full pipeline.

    Args:
        idea: The idea/requirement to generate code for
        **kwargs: Additional configuration

    Returns:
        GenerationResult
    """
    orchestrator = Orchestrator(
        kg_path=kwargs.get('kg_path', 'runs/knowledge_graph.json'),
        runs_dir=kwargs.get('runs_dir', 'runs')
    )

    request = GenerationRequest(
        id=kwargs.get('request_id', str(uuid.uuid4())[:8]),
        idea=idea,
        components=kwargs.get('components', []),
        output_dir=kwargs.get('output_dir', 'output'),
        config=kwargs.get('config', {})
    )

    return orchestrator.run_pipeline(request)
