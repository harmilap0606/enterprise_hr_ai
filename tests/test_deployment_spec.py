"""
tests/test_deployment_spec.py
==============================
Tests for container deployment specification for Hugging Face Spaces:
1. Verifies Dockerfile exists and configures port 7860 and multi-service setup.
2. Verifies entrypoint.sh startup script exists and configures both FastAPI and Streamlit on port 7860.
3. Verifies README.md frontmatter metadata matches Hugging Face Spaces Docker spec.
4. Verifies frontend/dashboard.py points to internal FastAPI (http://127.0.0.1:8000).
5. Verifies .dockerignore properly ignores bloat while retaining runtime assets.
6. Verifies requirements.txt includes all runtime packages.
"""

from pathlib import Path


def test_dockerfile_specification():
    """Verify Dockerfile presence and essential directives for Hugging Face Spaces."""
    dockerfile = Path("Dockerfile")
    assert dockerfile.exists(), "Dockerfile is missing!"

    content = dockerfile.read_text(encoding="utf-8")
    assert "FROM python:" in content
    assert "WORKDIR /app" in content
    assert "COPY requirements.txt" in content
    assert "pip install" in content
    assert "EXPOSE 7860" in content
    assert "EXPOSE" in content and "8000" in content
    assert "entrypoint.sh" in content
    assert "ENTRYPOINT" in content
    assert "7860" in content


def test_entrypoint_script_specification():
    """Verify entrypoint.sh presence and dual-service orchestration on port 7860."""
    entrypoint = Path("entrypoint.sh")
    assert entrypoint.exists(), "entrypoint.sh is missing!"

    content = entrypoint.read_text(encoding="utf-8")
    assert "uvicorn app.main:app" in content
    assert "127.0.0.1:8000" in content
    assert "streamlit run frontend/dashboard.py" in content
    assert "7860" in content


def test_huggingface_readme_metadata():
    """Verify README.md contains required Hugging Face Spaces Docker frontmatter."""
    readme = Path("README.md")
    assert readme.exists(), "README.md is missing!"

    content = readme.read_text(encoding="utf-8")
    assert "sdk: docker" in content
    assert "app_port: 7860" in content
    assert "title: Enterprise HR AI" in content


def test_dashboard_points_to_internal_fastapi():
    """Verify frontend/dashboard.py connects to internal FastAPI on http://127.0.0.1:8000."""
    dashboard = Path("frontend/dashboard.py")
    assert dashboard.exists(), "frontend/dashboard.py is missing!"

    content = dashboard.read_text(encoding="utf-8")
    assert 'API_BASE = "http://127.0.0.1:8000"' in content
    assert "localhost:8501" not in content


def test_dockerignore_specification():
    """Verify .dockerignore excludes source control and caches."""
    dockerignore = Path(".dockerignore")
    assert dockerignore.exists(), ".dockerignore is missing!"

    content = dockerignore.read_text(encoding="utf-8")
    assert ".git" in content
    assert "__pycache__" in content
    assert ".pytest_cache" in content


def test_requirements_runtime_completeness():
    """Verify requirements.txt covers all production runtime dependencies."""
    req_file = Path("requirements.txt")
    assert req_file.exists(), "requirements.txt is missing!"

    content = req_file.read_text(encoding="utf-8")
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "streamlit",
        "plotly",
        "requests",
        "rank-bm25",
        "chromadb",
        "transformers",
        "torch",
        "langgraph"
    ]
    for pkg in required_packages:
        assert pkg in content.lower(), f"Package '{pkg}' missing from requirements.txt!"
