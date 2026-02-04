"""
APPLUS3 Context Retrieval Module
Generated following Knowledge Graph patterns from Idea2Paper

Retrieves relevant context from KG for generation prompts.
Uses BFS traversal for dependency chains.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Set, Any, Optional

from .knowledge_graph import KnowledgeGraph, Node, Edge, EdgeType


@dataclass
class RetrievalResult:
    """Result of a context retrieval operation."""
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    context_text: str = ""
    relevance_score: float = 0.0

    def is_empty(self) -> bool:
        """Check if result contains any data."""
        return len(self.nodes) == 0 and len(self.edges) == 0


class ContextRetriever:
    """
    Retrieves relevant context from Knowledge Graph for code generation.

    Implements multiple retrieval strategies:
    - Dependency chain traversal (BFS)
    - Pattern-based retrieval
    - Combined context for generation
    """

    def __init__(self, kg: KnowledgeGraph):
        """
        Initialize retriever with a knowledge graph.

        Args:
            kg: KnowledgeGraph instance to query
        """
        self.kg = kg

    def get_dependency_chain(self, node_id: str, max_depth: int = 3) -> List[Node]:
        """
        Get all dependencies of a node using BFS traversal.

        Args:
            node_id: Starting node ID
            max_depth: Maximum traversal depth

        Returns:
            List of dependent nodes in BFS order
        """
        result: List[Node] = []
        visited: Set[str] = set()
        queue: deque = deque([(node_id, 0)])

        while queue:
            current_id, depth = queue.popleft()

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)
            node = self.kg.get_node(current_id)

            if node and current_id != node_id:
                result.append(node)

            if depth < max_depth:
                dependencies = self.kg.get_dependencies(current_id)
                for dep in dependencies:
                    if dep.id not in visited:
                        queue.append((dep.id, depth + 1))

        return result

    def get_related_by_pattern(self, pattern_name: str) -> List[Node]:
        """
        Get all nodes that use a specific pattern.

        Args:
            pattern_name: Name of the pattern to search for

        Returns:
            List of nodes using the pattern
        """
        result: List[Node] = []
        all_edges = self.kg.get_all_edges()

        for edge in all_edges:
            if edge.edge_type == EdgeType.USES_PATTERN.value:
                target_node = self.kg.get_node(edge.target_id)
                if target_node and pattern_name.lower() in target_node.name.lower():
                    source_node = self.kg.get_node(edge.source_id)
                    if source_node:
                        result.append(source_node)

        return result

    def get_context_for_generation(self, target_node_id: str) -> RetrievalResult:
        """
        Get comprehensive context for generating a target node.

        Retrieves:
        - Direct dependencies
        - Dependency chain (transitive)
        - Related patterns
        - Relevant edges

        Args:
            target_node_id: ID of the node to generate

        Returns:
            RetrievalResult with all relevant context
        """
        result = RetrievalResult()
        target_node = self.kg.get_node(target_node_id)

        if not target_node:
            return result

        # Get dependency chain
        chain = self.get_dependency_chain(target_node_id, max_depth=3)
        result.nodes.extend(chain)

        # Get direct dependencies
        direct_deps = self.kg.get_dependencies(target_node_id)
        for dep in direct_deps:
            if dep not in result.nodes:
                result.nodes.append(dep)

        # Get pattern if target uses one
        if target_node.metadata.get('pattern'):
            pattern_related = self.get_related_by_pattern(target_node.metadata['pattern'])
            for node in pattern_related[:3]:  # Limit to 3 examples
                if node not in result.nodes and node.id != target_node_id:
                    result.nodes.append(node)

        # Collect relevant edges
        all_edges = self.kg.get_all_edges()
        node_ids = {n.id for n in result.nodes}
        node_ids.add(target_node_id)

        for edge in all_edges:
            if edge.source_id in node_ids or edge.target_id in node_ids:
                result.edges.append(edge)

        # Calculate relevance score
        result.relevance_score = self._calculate_relevance(target_node, result)

        # Build context text
        result.context_text = self.build_prompt_context(result)

        return result

    def _calculate_relevance(self, target: Node, result: RetrievalResult) -> float:
        """
        Calculate relevance score for retrieved context.

        Args:
            target: Target node
            result: Retrieved result

        Returns:
            Relevance score between 0 and 1
        """
        if not result.nodes:
            return 0.0

        score = 0.0

        # Direct dependencies are highly relevant
        direct_deps = self.kg.get_dependencies(target.id)
        direct_dep_ids = {d.id for d in direct_deps}

        for node in result.nodes:
            if node.id in direct_dep_ids:
                score += 0.3
            elif node.type == target.type:
                score += 0.1
            else:
                score += 0.05

        # Normalize to [0, 1]
        return min(1.0, score)

    def build_prompt_context(
        self,
        result: RetrievalResult,
        max_chars: int = 8000
    ) -> str:
        """
        Build LLM-ready context string from retrieval result.

        Args:
            result: RetrievalResult to format
            max_chars: Maximum characters in output

        Returns:
            Formatted context string for LLM prompt
        """
        lines: List[str] = []
        char_count = 0

        # Header
        header = "=== KNOWLEDGE GRAPH CONTEXT ===\n"
        lines.append(header)
        char_count += len(header)

        # Nodes section
        if result.nodes:
            lines.append("\n## RELATED COMPONENTS:\n")
            char_count += 25

            for node in result.nodes:
                if char_count >= max_chars:
                    break

                node_text = self._format_node(node)
                if char_count + len(node_text) < max_chars:
                    lines.append(node_text)
                    char_count += len(node_text)

        # Edges section (relationships)
        if result.edges and char_count < max_chars - 500:
            lines.append("\n## RELATIONSHIPS:\n")
            char_count += 20

            for edge in result.edges[:10]:  # Limit edges
                if char_count >= max_chars:
                    break

                edge_text = self._format_edge(edge)
                if char_count + len(edge_text) < max_chars:
                    lines.append(edge_text)
                    char_count += len(edge_text)

        # Footer
        footer = f"\n=== END CONTEXT (relevance: {result.relevance_score:.2f}) ===\n"
        lines.append(footer)

        return "".join(lines)

    def _format_node(self, node: Node) -> str:
        """Format a node for prompt context."""
        lines = [
            f"\n### {node.name} ({node.type})",
            f"- ID: {node.id}",
            f"- File: {node.file_path or 'N/A'}",
        ]

        if node.metadata:
            for key, value in list(node.metadata.items())[:5]:
                if isinstance(value, str) and len(value) < 200:
                    lines.append(f"- {key}: {value}")

        return "\n".join(lines) + "\n"

    def _format_edge(self, edge: Edge) -> str:
        """Format an edge for prompt context."""
        return f"- {edge.source_id} --[{edge.edge_type}]--> {edge.target_id}\n"

    def get_generation_order(self) -> List[str]:
        """
        Determine optimal generation order based on dependencies.

        Uses topological sort to ensure dependencies are generated first.

        Returns:
            List of node IDs in generation order
        """
        all_nodes = self.kg.get_all_nodes()

        # Build in-degree map
        in_degree: Dict[str, int] = {n.id: 0 for n in all_nodes}

        for edge in self.kg.get_all_edges():
            if edge.edge_type == EdgeType.DEPENDS_ON.value:
                if edge.source_id in in_degree:
                    in_degree[edge.source_id] += 1

        # Topological sort using Kahn's algorithm
        queue = deque([n_id for n_id, deg in in_degree.items() if deg == 0])
        result: List[str] = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            # Decrease in-degree of dependents
            dependents = self.kg.get_dependents(node_id)
            for dep in dependents:
                if dep.id in in_degree:
                    in_degree[dep.id] -= 1
                    if in_degree[dep.id] == 0:
                        queue.append(dep.id)

        return result
