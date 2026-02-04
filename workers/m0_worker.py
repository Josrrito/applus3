"""
APPLUS3 M0 Worker - Analysis Module
Parses user request, identifies components needed, creates initial KG nodes

M0 is the Discovery phase: understanding what needs to be built.
"""

import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import sys
sys.path.insert(0, '..')

from orchestrator.knowledge_graph import KnowledgeGraph, Node, Edge, EdgeType


@dataclass
class ComponentSpec:
    """Specification for a component to generate."""
    id: str
    name: str
    description: str
    file_path: str
    pattern: str = ""
    depends_on: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    priority: int = 99


@dataclass
class AnalysisResult:
    """Result from M0 analysis."""
    success: bool
    components: List[ComponentSpec] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    patterns_identified: List[str] = field(default_factory=list)
    error: Optional[str] = None


class M0Worker:
    """
    M0 Analysis Worker.

    Responsibilities:
    - Parse user idea/requirements
    - Identify components needed
    - Determine dependencies between components
    - Create initial Knowledge Graph nodes
    - Identify applicable patterns
    """

    # Common component patterns to detect
    COMPONENT_PATTERNS = {
        'api': r'\b(api|endpoint|route|fastapi|flask)\b',
        'database': r'\b(database|db|storage|persist|sql|mongo)\b',
        'auth': r'\b(auth|login|jwt|token|session)\b',
        'worker': r'\b(worker|job|queue|task|celery)\b',
        'model': r'\b(model|schema|entity|dataclass)\b',
        'service': r'\b(service|business|logic|handler)\b',
        'config': r'\b(config|settings|env|environment)\b',
        'utils': r'\b(util|helper|common|shared)\b',
    }

    def __init__(self, kg: Optional[KnowledgeGraph] = None):
        """
        Initialize M0 worker.

        Args:
            kg: Optional KnowledgeGraph to populate
        """
        self.kg = kg

    def analyze(self, idea: str, context: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """
        Analyze idea and identify components.

        Args:
            idea: User's idea or requirements text
            context: Optional additional context

        Returns:
            AnalysisResult with identified components
        """
        context = context or {}

        try:
            # Step 1: Extract components from idea
            components = self._extract_components(idea)

            # Step 2: Infer dependencies
            dependencies = self._infer_dependencies(components)

            # Step 3: Identify patterns
            patterns = self._identify_patterns(idea, components)

            # Step 4: Assign priorities (topological order)
            self._assign_priorities(components, dependencies)

            # Step 5: Populate KG if available
            if self.kg:
                self._populate_kg(components, dependencies)

            return AnalysisResult(
                success=True,
                components=components,
                dependencies=dependencies,
                patterns_identified=patterns
            )

        except Exception as e:
            return AnalysisResult(
                success=False,
                error=str(e)
            )

    def _extract_components(self, idea: str) -> List[ComponentSpec]:
        """
        Extract components from idea text.

        Uses pattern matching and heuristics.
        In production: would use LLM for semantic extraction.
        """
        components: List[ComponentSpec] = []
        idea_lower = idea.lower()

        # Check for explicit component mentions
        for comp_type, pattern in self.COMPONENT_PATTERNS.items():
            if re.search(pattern, idea_lower, re.IGNORECASE):
                components.append(self._create_component(comp_type, idea))

        # Always include main entry point
        if not any(c.id == 'main' for c in components):
            components.append(ComponentSpec(
                id='main',
                name='Main Entry Point',
                description='Application entry point',
                file_path='main.py',
                priority=99
            ))

        # If no specific components found, create generic structure
        if len(components) <= 1:
            components = self._create_default_structure(idea)

        return components

    def _create_component(self, comp_type: str, idea: str) -> ComponentSpec:
        """Create a ComponentSpec for a given type."""
        templates = {
            'api': ComponentSpec(
                id='api',
                name='API Layer',
                description='REST API endpoints',
                file_path='api/main.py',
                pattern='api_pattern',
                depends_on=['service'],
                exports=['app', 'router']
            ),
            'database': ComponentSpec(
                id='database',
                name='Database Layer',
                description='Data persistence and storage',
                file_path='db/database.py',
                pattern='repository_pattern',
                exports=['Database', 'get_db']
            ),
            'auth': ComponentSpec(
                id='auth',
                name='Authentication',
                description='User authentication and authorization',
                file_path='auth/auth.py',
                pattern='auth_pattern',
                depends_on=['database'],
                exports=['authenticate', 'create_token']
            ),
            'worker': ComponentSpec(
                id='worker',
                name='Background Worker',
                description='Async job processing',
                file_path='workers/worker.py',
                pattern='worker_pattern',
                depends_on=['database'],
                exports=['Worker', 'process_job']
            ),
            'model': ComponentSpec(
                id='model',
                name='Data Models',
                description='Data structures and schemas',
                file_path='models/models.py',
                pattern='model_pattern',
                exports=['Model', 'Schema']
            ),
            'service': ComponentSpec(
                id='service',
                name='Business Logic',
                description='Core business logic layer',
                file_path='services/service.py',
                pattern='service_pattern',
                depends_on=['database', 'model'],
                exports=['Service']
            ),
            'config': ComponentSpec(
                id='config',
                name='Configuration',
                description='Application settings',
                file_path='config/settings.py',
                pattern='config_pattern',
                exports=['Settings', 'get_settings']
            ),
            'utils': ComponentSpec(
                id='utils',
                name='Utilities',
                description='Helper functions and common utilities',
                file_path='utils/helpers.py',
                exports=['helpers']
            ),
        }
        return templates.get(comp_type, templates['utils'])

    def _create_default_structure(self, idea: str) -> List[ComponentSpec]:
        """Create default project structure when no specific components detected."""
        return [
            ComponentSpec(
                id='core',
                name='Core Module',
                description=f'Core logic for: {idea[:100]}',
                file_path='core/core.py',
                priority=1
            ),
            ComponentSpec(
                id='main',
                name='Main Entry Point',
                description='Application entry point',
                file_path='main.py',
                depends_on=['core'],
                priority=2
            ),
        ]

    def _infer_dependencies(
        self,
        components: List[ComponentSpec]
    ) -> Dict[str, List[str]]:
        """
        Infer dependencies between components.

        Returns map of component_id -> [dependency_ids]
        """
        dependencies: Dict[str, List[str]] = {}
        component_ids = {c.id for c in components}

        for comp in components:
            # Filter to only existing components
            valid_deps = [d for d in comp.depends_on if d in component_ids]
            dependencies[comp.id] = valid_deps

        return dependencies

    def _identify_patterns(
        self,
        idea: str,
        components: List[ComponentSpec]
    ) -> List[str]:
        """Identify design patterns applicable to the project."""
        patterns = set()

        # Collect patterns from components
        for comp in components:
            if comp.pattern:
                patterns.add(comp.pattern)

        # Detect additional patterns from idea
        if 'orchestrat' in idea.lower():
            patterns.add('orchestrator_pattern')
        if 'pipeline' in idea.lower():
            patterns.add('pipeline_pattern')
        if 'event' in idea.lower():
            patterns.add('event_driven_pattern')

        return list(patterns)

    def _assign_priorities(
        self,
        components: List[ComponentSpec],
        dependencies: Dict[str, List[str]]
    ) -> None:
        """
        Assign generation priorities based on dependency order.

        Components with no dependencies get priority 1.
        Others get priority = 1 + max(priority of dependencies)
        """
        comp_map = {c.id: c for c in components}
        visited = set()
        priorities: Dict[str, int] = {}

        def get_priority(comp_id: str) -> int:
            if comp_id in priorities:
                return priorities[comp_id]
            if comp_id in visited:
                return 99  # Cycle detected
            visited.add(comp_id)

            deps = dependencies.get(comp_id, [])
            if not deps:
                priorities[comp_id] = 1
            else:
                max_dep_priority = max(get_priority(d) for d in deps if d in comp_map)
                priorities[comp_id] = max_dep_priority + 1

            return priorities[comp_id]

        for comp in components:
            comp.priority = get_priority(comp.id)

    def _populate_kg(
        self,
        components: List[ComponentSpec],
        dependencies: Dict[str, List[str]]
    ) -> None:
        """Populate Knowledge Graph with component nodes and edges."""
        # Add nodes
        for comp in components:
            node = Node(
                id=comp.id,
                type='Module',
                name=comp.name,
                metadata={
                    'description': comp.description,
                    'pattern': comp.pattern,
                    'priority': comp.priority,
                    'exports': comp.exports
                },
                file_path=comp.file_path
            )
            self.kg.add_node(node)

        # Add edges
        for comp_id, deps in dependencies.items():
            for dep_id in deps:
                edge = Edge(
                    source_id=comp_id,
                    target_id=dep_id,
                    edge_type=EdgeType.DEPENDS_ON.value
                )
                try:
                    self.kg.add_edge(edge)
                except ValueError:
                    pass  # Skip if nodes don't exist


def analyze_idea(idea: str, kg: Optional[KnowledgeGraph] = None) -> AnalysisResult:
    """
    Convenience function to analyze an idea.

    Args:
        idea: User's idea text
        kg: Optional KnowledgeGraph

    Returns:
        AnalysisResult
    """
    worker = M0Worker(kg)
    return worker.analyze(idea)
