"""
FastAPI Application Entry Point.
Registers routers for Attrition Prediction, Executive Dashboard, and Skills Capability.
Includes request logging middleware and application lifecycle management.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from app.api import attrition, dashboard, skills, rag, agents
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Application startup complete - Enterprise HR AI Platform online")
    yield
    logger.info("Application shutdown complete")


# Core application instance
app = FastAPI(
    title="Enterprise HR AI Platform",
    description=(
        "Production AI Platform for Workforce Retention, Engagement Intelligence, "
        "and Capability Development."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs each request on receipt and upon completion."""
    method = request.method
    path = request.url.path
    
    logger.info(f"Incoming request: {method} {path}")
    response = await call_next(request)
    logger.info(f"Request completed: {method} {path} with status {response.status_code}")
    return response


# Mount API Routers
app.include_router(attrition.router)
app.include_router(dashboard.router)
app.include_router(skills.router)
app.include_router(rag.router)
app.include_router(agents.router)


@app.get("/", tags=["System"])
def root():
    """Health check and platform status."""
    return {
        "status": "online",
        "service": "Enterprise HR AI Platform",
        "version": "1.0.0",
        "endpoints": [
            "POST /predict/attrition",
            "GET  /dashboard/summary",
            "GET  /dashboard/attrition-by-department",
            "GET  /dashboard/skill-gaps",
            "GET  /dashboard/recommendations?department=&risk_level=",
            "GET  /employees/{employee_id}",
            "GET  /skills/recommendations/{employee_id}"
        ],
        "documentation": "/docs"
    }


@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe."""
    return {"status": "healthy"}
