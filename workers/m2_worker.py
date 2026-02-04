"""
APPLUS3 M2 Worker - Deployment Module
Generates configs, Dockerfiles, requirements, deployment scripts

M2 is the Export phase: preparing code for deployment.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
import re


@dataclass
class DeploymentResult:
    """Result from deployment configuration generation."""
    success: bool
    files: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


class M2Worker:
    """
    M2 Deployment Worker.

    Generates:
    - requirements.txt (extracted from imports)
    - Dockerfile (production-ready)
    - docker-compose.yml
    - .env.example
    - Deployment scripts
    """

    # Known package mappings (import name -> pip package name)
    PACKAGE_MAPPINGS = {
        'fastapi': 'fastapi>=0.104.0',
        'uvicorn': 'uvicorn>=0.24.0',
        'pydantic': 'pydantic>=2.5.0',
        'requests': 'requests>=2.31.0',
        'httpx': 'httpx>=0.25.0',
        'sqlalchemy': 'sqlalchemy>=2.0.0',
        'alembic': 'alembic>=1.12.0',
        'redis': 'redis>=5.0.0',
        'celery': 'celery>=5.3.0',
        'pytest': 'pytest>=7.4.0',
        'python-dotenv': 'python-dotenv>=1.0.0',
        'pyjwt': 'pyjwt>=2.8.0',
        'bcrypt': 'bcrypt>=4.0.0',
        'passlib': 'passlib>=1.7.0',
        'aiohttp': 'aiohttp>=3.9.0',
        'numpy': 'numpy>=1.26.0',
        'pandas': 'pandas>=2.1.0',
    }

    # Standard library modules (don't include in requirements)
    STDLIB_MODULES = {
        'os', 'sys', 'json', 'time', 'datetime', 'typing', 'dataclasses',
        'pathlib', 'enum', 'collections', 'functools', 'itertools',
        'threading', 'asyncio', 'logging', 'uuid', 're', 'math',
        'hashlib', 'base64', 'urllib', 'http', 'io', 'abc',
        'contextlib', 'copy', 'traceback', 'inspect', 'importlib'
    }

    def __init__(self, project_name: str = "applus3"):
        """
        Initialize M2 worker.

        Args:
            project_name: Name of the project
        """
        self.project_name = project_name

    def generate(
        self,
        generated_files: Dict[str, str],
        config: Optional[Dict[str, Any]] = None
    ) -> DeploymentResult:
        """
        Generate deployment configuration files.

        Args:
            generated_files: Dict of file_path -> code
            config: Optional deployment configuration

        Returns:
            DeploymentResult with generated config files
        """
        config = config or {}
        files: Dict[str, str] = {}
        warnings: List[str] = []

        try:
            # Extract imports from all Python files
            imports = self._extract_imports(generated_files)

            # Generate requirements.txt
            files['requirements.txt'] = self._generate_requirements(imports)

            # Generate Dockerfile
            files['Dockerfile'] = self._generate_dockerfile(config)

            # Generate docker-compose.yml
            files['docker-compose.yml'] = self._generate_docker_compose(config)

            # Generate .env.example
            files['.env.example'] = self._generate_env_example(generated_files)

            # Generate .gitignore
            files['.gitignore'] = self._generate_gitignore()

            # Generate README.md
            files['README.md'] = self._generate_readme(generated_files, config)

            # Check for potential issues
            warnings = self._check_deployment_issues(generated_files, imports)

            return DeploymentResult(
                success=True,
                files=files,
                warnings=warnings
            )

        except Exception as e:
            return DeploymentResult(
                success=False,
                error=str(e)
            )

    def _extract_imports(self, files: Dict[str, str]) -> Set[str]:
        """Extract all import statements from Python files."""
        imports: Set[str] = set()

        for file_path, code in files.items():
            if not file_path.endswith('.py'):
                continue

            # Match import statements
            import_pattern = r'^(?:from\s+(\w+)|import\s+(\w+))'

            for line in code.split('\n'):
                match = re.match(import_pattern, line.strip())
                if match:
                    module = match.group(1) or match.group(2)
                    # Get top-level package
                    top_level = module.split('.')[0]
                    if top_level and top_level not in self.STDLIB_MODULES:
                        imports.add(top_level)

        return imports

    def _generate_requirements(self, imports: Set[str]) -> str:
        """Generate requirements.txt from imports."""
        requirements = []

        # Always include core packages
        requirements.append('# Core dependencies')
        requirements.append('fastapi>=0.104.0')
        requirements.append('uvicorn>=0.24.0')
        requirements.append('pydantic>=2.5.0')
        requirements.append('requests>=2.31.0')
        requirements.append('')

        # Add detected packages
        requirements.append('# Detected dependencies')
        for imp in sorted(imports):
            if imp in self.PACKAGE_MAPPINGS:
                pkg = self.PACKAGE_MAPPINGS[imp]
                if pkg not in requirements:
                    requirements.append(pkg)
            elif imp not in {'fastapi', 'uvicorn', 'pydantic', 'requests'}:
                # Unknown package - add with warning
                requirements.append(f'{imp}  # TODO: verify version')

        requirements.append('')
        requirements.append('# Development dependencies')
        requirements.append('pytest>=7.4.0')
        requirements.append('black>=23.0.0')
        requirements.append('ruff>=0.1.0')

        return '\n'.join(requirements)

    def _generate_dockerfile(self, config: Dict[str, Any]) -> str:
        """Generate production Dockerfile."""
        python_version = config.get('python_version', '3.11')
        port = config.get('port', 8000)

        return f'''# APPLUS3 Production Dockerfile
# Generated by M2 Worker

FROM python:{python_version}-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT={port}

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \\
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{port}/health || exit 1

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "{port}"]
'''

    def _generate_docker_compose(self, config: Dict[str, Any]) -> str:
        """Generate docker-compose.yml."""
        port = config.get('port', 8000)

        return f'''# APPLUS3 Docker Compose Configuration
# Generated by M2 Worker

version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "{port}:{port}"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Uncomment if using Redis
  # redis:
  #   image: redis:7-alpine
  #   ports:
  #     - "6379:6379"
  #   restart: unless-stopped

networks:
  default:
    driver: bridge
'''

    def _generate_env_example(self, files: Dict[str, str]) -> str:
        """Generate .env.example from code analysis."""
        env_vars = set()

        # Common patterns for environment variable access
        patterns = [
            r'os\.environ\[[\'"]([\w_]+)[\'\"]\]',
            r'os\.environ\.get\([\'"]([\w_]+)[\'\"]\)',
            r'os\.getenv\([\'"]([\w_]+)[\'\"]\)',
            r'settings\.(\w+)',
        ]

        for code in files.values():
            for pattern in patterns:
                matches = re.findall(pattern, code)
                env_vars.update(matches)

        # Generate .env.example
        lines = [
            '# APPLUS3 Environment Variables',
            '# Copy this file to .env and fill in the values',
            '',
            '# Application',
            'ENVIRONMENT=development',
            'LOG_LEVEL=INFO',
            'PORT=8000',
            '',
            '# API Keys (required for Multi-IA)',
            'OPENAI_API_KEY=sk-...',
            'ANTHROPIC_API_KEY=sk-ant-...',
            'DEEPSEEK_API_KEY=sk-...',
            '',
        ]

        # Add detected environment variables
        if env_vars:
            lines.append('# Detected in code')
            for var in sorted(env_vars):
                if var not in {'ENVIRONMENT', 'LOG_LEVEL', 'PORT',
                               'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPSEEK_API_KEY'}:
                    lines.append(f'{var}=')

        return '\n'.join(lines)

    def _generate_gitignore(self) -> str:
        """Generate .gitignore file."""
        return '''# APPLUS3 .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/

# IDEs
.idea/
.vscode/
*.swp
*.swo

# Environment
.env
.env.local
*.env

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Build
build/
dist/
*.egg-info/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Project specific
runs/
*.json.bak
knowledge_graph.json
'''

    def _generate_readme(
        self,
        files: Dict[str, str],
        config: Dict[str, Any]
    ) -> str:
        """Generate README.md."""
        file_list = '\n'.join([f'- `{f}`' for f in sorted(files.keys()) if f.endswith('.py')])

        return f'''# {self.project_name.upper()}

Generated by APPLUS3 - Context-Aware Multi-IA Code Generation

## Overview

This project was generated using:
- **Knowledge Graph**: Maintains context between components
- **Multi-IA Consensus**: GPT-4o + Claude + DeepSeek
- **Anchored Review**: Quality validation

## Structure

{file_list}

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn api.main:app --reload

# Run with Docker
docker-compose up --build
```

## Configuration

Copy `.env.example` to `.env` and configure:
- API keys for Multi-IA providers
- Application settings

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

MIT
'''

    def _check_deployment_issues(
        self,
        files: Dict[str, str],
        imports: Set[str]
    ) -> List[str]:
        """Check for potential deployment issues."""
        warnings = []

        # Check for hardcoded secrets
        secret_patterns = [
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
        ]

        for file_path, code in files.items():
            for pattern in secret_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    warnings.append(f"Potential hardcoded secret in {file_path}")

        # Check for missing common packages
        if 'fastapi' not in imports and any('api' in f for f in files.keys()):
            warnings.append("API files detected but fastapi not imported")

        return warnings


def generate_deployment(
    generated_files: Dict[str, str],
    config: Optional[Dict[str, Any]] = None
) -> DeploymentResult:
    """
    Convenience function for deployment generation.

    Args:
        generated_files: Generated Python files
        config: Deployment configuration

    Returns:
        DeploymentResult
    """
    worker = M2Worker()
    return worker.generate(generated_files, config)
