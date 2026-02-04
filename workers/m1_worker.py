"""
APPLUS3 M1 Worker - Code Generation Module
AgentCoder per-function, uses Multi-IA consensus, receives KG context

M1 is the Generation phase: creating actual code with context awareness.
"""

import os
import json
import time
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class GenerationResult:
    """Result from code generation."""
    success: bool
    code: str = ""
    consensus_type: str = ""  # CONSENSUS, RSA_SYNTHESIZED, NO_CONSENSUS
    cost: float = 0.0
    duration_ms: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class M1Worker:
    """
    M1 Code Generation Worker.

    Implements AgentCoder pattern:
    - Receives component specification with KG context
    - Calls Multi-IA (GPT + Claude + DeepSeek) for consensus
    - Validates generated code
    - Returns complete, executable code

    Uses APPLUS-WEBAPP backend for Multi-IA consensus.
    """

    # Default backend URL (can be overridden)
    DEFAULT_BACKEND_URL = "https://applus-webapp-backend.onrender.com"

    # Prompt template for code generation
    CODE_PROMPT_TEMPLATE = """Generate a COMPLETE, EXECUTABLE Python module.

=== COMPONENT SPECIFICATION ===
Name: {name}
Description: {description}
File: {file_path}
Exports: {exports}
Dependencies: {depends_on}
Pattern: {pattern}

=== KNOWLEDGE GRAPH CONTEXT ===
{kg_context}

=== PREVIOUSLY GENERATED FILES ===
{previous_files}

=== REQUIREMENTS ===
1. Generate complete Python code - NO placeholders, NO TODOs
2. Include all imports at the top
3. Include docstrings for all classes and functions
4. Use type hints throughout
5. Handle errors appropriately
6. Make it production-ready

Generate ONLY the Python code. No explanations, no markdown."""

    def __init__(
        self,
        backend_url: Optional[str] = None,
        timeout: int = 120
    ):
        """
        Initialize M1 worker.

        Args:
            backend_url: URL of APPLUS-WEBAPP backend (for Multi-IA)
            timeout: Request timeout in seconds
        """
        self.backend_url = backend_url or os.getenv(
            'APPLUS_BACKEND_URL',
            self.DEFAULT_BACKEND_URL
        )
        self.timeout = timeout
        self._session = requests.Session()

    def generate(
        self,
        component: Dict[str, Any],
        kg_context: str = "",
        previous_files: Dict[str, str] = None
    ) -> GenerationResult:
        """
        Generate code for a component using Multi-IA consensus.

        Args:
            component: Component specification dict
            kg_context: Context from Knowledge Graph
            previous_files: Previously generated files for context

        Returns:
            GenerationResult with generated code
        """
        start_time = time.time()
        previous_files = previous_files or {}

        try:
            # Build the prompt
            prompt = self._build_prompt(component, kg_context, previous_files)

            # Call Multi-IA consensus
            result = self._call_multi_ia(prompt)

            if not result.get('success'):
                return GenerationResult(
                    success=False,
                    error=result.get('error', 'Multi-IA call failed'),
                    duration_ms=int((time.time() - start_time) * 1000)
                )

            # Extract code from response
            code = self._extract_code(result.get('majority_response', ''))

            # Validate code
            if not self._validate_code(code):
                return GenerationResult(
                    success=False,
                    code=code,
                    error="Generated code failed validation",
                    duration_ms=int((time.time() - start_time) * 1000)
                )

            return GenerationResult(
                success=True,
                code=code,
                consensus_type=result.get('consensus_result', 'UNKNOWN'),
                cost=result.get('total_cost', 0.0),
                duration_ms=int((time.time() - start_time) * 1000),
                metadata={
                    'model_responses': len(result.get('individual_responses', [])),
                    'avg_confidence': result.get('avg_confidence', 0.0)
                }
            )

        except Exception as e:
            return GenerationResult(
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000)
            )

    def _build_prompt(
        self,
        component: Dict[str, Any],
        kg_context: str,
        previous_files: Dict[str, str]
    ) -> str:
        """Build generation prompt with context."""
        # Format previous files (truncated for context limit)
        prev_files_text = ""
        char_count = 0
        max_chars = 4000

        for path, code in previous_files.items():
            if char_count > max_chars:
                prev_files_text += f"\n... and {len(previous_files) - len(prev_files_text.split('==='))} more files"
                break
            snippet = f"\n### {path}\n```python\n{code[:1000]}...\n```\n"
            prev_files_text += snippet
            char_count += len(snippet)

        if not prev_files_text:
            prev_files_text = "No previous files generated yet."

        return self.CODE_PROMPT_TEMPLATE.format(
            name=component.get('name', 'Module'),
            description=component.get('description', ''),
            file_path=component.get('file_path', component.get('file', 'module.py')),
            exports=', '.join(component.get('exports', [])),
            depends_on=', '.join(component.get('depends_on', [])),
            pattern=component.get('pattern', 'N/A'),
            kg_context=kg_context[:3000] if kg_context else "No KG context available.",
            previous_files=prev_files_text
        )

    def _call_multi_ia(self, prompt: str) -> Dict[str, Any]:
        """
        Call APPLUS-WEBAPP Multi-IA consensus endpoint.

        Args:
            prompt: The generation prompt

        Returns:
            Response dict from Multi-IA
        """
        url = f"{self.backend_url}/api/ai/test-consensus"

        try:
            response = self._session.post(
                url,
                json={"prompt": prompt},
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timed out - backend may be sleeping (Render free tier)'
            }
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': 'Could not connect to backend'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _extract_code(self, response: str) -> str:
        """
        Extract Python code from LLM response.

        Handles various response formats:
        - Plain code
        - Markdown code blocks
        - Mixed text and code
        """
        if not response:
            return ""

        # Try to extract from markdown code block
        if '```python' in response:
            start = response.find('```python') + 9
            end = response.find('```', start)
            if end > start:
                return response[start:end].strip()

        if '```' in response:
            start = response.find('```') + 3
            # Skip language identifier if present
            newline = response.find('\n', start)
            if newline > start and newline - start < 20:
                start = newline + 1
            end = response.find('```', start)
            if end > start:
                return response[start:end].strip()

        # If no code blocks, assume entire response is code
        # but clean up any leading/trailing text
        lines = response.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            # Detect start of Python code
            if (line.strip().startswith(('import ', 'from ', 'def ', 'class ', '#', '"""', "'''")) or
                    (in_code and line.strip())):
                in_code = True
                code_lines.append(line)
            elif in_code and not line.strip():
                code_lines.append(line)  # Keep empty lines within code

        return '\n'.join(code_lines).strip() if code_lines else response.strip()

    def _validate_code(self, code: str) -> bool:
        """
        Basic validation of generated code.

        Args:
            code: Code to validate

        Returns:
            True if code appears valid
        """
        if not code or len(code) < 50:
            return False

        # Check for Python syntax indicators
        has_python_elements = any([
            'def ' in code,
            'class ' in code,
            'import ' in code,
            'from ' in code,
        ])

        # Check it's not HTML/JS
        not_web_code = all([
            '<html>' not in code.lower(),
            '<script>' not in code.lower(),
            'function(' not in code,
        ])

        return has_python_elements and not_web_code

    def generate_batch(
        self,
        components: List[Dict[str, Any]],
        kg_context: str = "",
        max_parallel: int = 3
    ) -> Dict[str, GenerationResult]:
        """
        Generate code for multiple components.

        Args:
            components: List of component specifications
            kg_context: Shared KG context
            max_parallel: Maximum parallel generations

        Returns:
            Dict mapping component_id to GenerationResult
        """
        results: Dict[str, GenerationResult] = {}
        previous_files: Dict[str, str] = {}

        # Sort by priority to respect dependencies
        sorted_components = sorted(
            components,
            key=lambda c: c.get('priority', 99)
        )

        # Generate sequentially to maintain context chain
        for component in sorted_components:
            comp_id = component.get('id', component.get('name', 'unknown'))

            result = self.generate(
                component=component,
                kg_context=kg_context,
                previous_files=previous_files
            )

            results[comp_id] = result

            # Add to previous files for next component's context
            if result.success:
                file_path = component.get('file_path', component.get('file', f'{comp_id}.py'))
                previous_files[file_path] = result.code

        return results


def generate_code(
    component: Dict[str, Any],
    kg_context: str = "",
    previous_files: Dict[str, str] = None
) -> GenerationResult:
    """
    Convenience function for single component generation.

    Args:
        component: Component specification
        kg_context: KG context string
        previous_files: Previously generated files

    Returns:
        GenerationResult
    """
    worker = M1Worker()
    return worker.generate(component, kg_context, previous_files)
