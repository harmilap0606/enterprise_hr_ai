"""
streamlit_app.py
================
Streamlit Community Cloud root entrypoint.
Launches the frontend dashboard located at frontend/dashboard.py.

NOTE: This file is the Streamlit UI root entrypoint.
The FastAPI application remains in app/main.py.
"""

import sys
import runpy
from pathlib import Path

repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

dashboard_path = repo_root / "frontend" / "dashboard.py"
runpy.run_path(str(dashboard_path), run_name="__main__")
