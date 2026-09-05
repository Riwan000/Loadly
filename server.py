"""
⚡ Loadly - FastAPI Server
Stage 1.5: Forecast + Explain REST API

Usage:
    python server.py
    # or
    uvicorn server:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

from api.routes import router as api_router
from models import load_model, load_model_metadata

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    print("\n" + "=" * 60)
    print("⚡ Starting Loadly API Server")
    print("=" * 60)

    # Preload model and metadata into cache at startup
    model = load_model()
    metadata = load_model_metadata()
    if model is not None:
        print("✅ ML Model loaded successfully")
    else:
        print("⚠️ Warning: Model could not be loaded at startup")

    if metadata is not None:
        print("✅ Model metadata loaded successfully")
    else:
        print("⚠️ Warning: Model metadata not found")

    print(f"📖 Interactive API Documentation: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    yield
    print("\n⚡ Shutting down Loadly API Server...")


# Initialize FastAPI App
app = FastAPI(
    title="⚡ Loadly API — Intelligent Power Demand Forecasting",
    description="REST API for multi-scale electricity demand forecasting, weather impact analytics, and AI energy assistant across all Indian states.",
    version="1.5.0",
    lifespan=lifespan
)

# Determine allowed origins for CORS
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    allowed_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]

# Enable CORS for React and frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)

from fastapi import Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend_assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str, request: Request):
        if full_path.startswith("api/") or full_path in ("docs", "redoc", "openapi.json"):
            raise HTTPException(status_code=404, detail="Not Found")
        
        # If root is requested without text/html (API clients/TestClient), return API info
        accept_header = request.headers.get("accept", "")
        if full_path == "" and "text/html" not in accept_header:
            return {
                "message": "⚡ Loadly API is running.",
                "version": "1.5.0",
                "documentation": "/docs",
                "redoc": "/redoc",
                "health_check": "/api/health"
            }

        file_path = os.path.join(frontend_dist, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    @app.get("/", tags=["General"])
    def root():
        """Root landing endpoint (when frontend is not built)."""
        return {
            "message": "⚡ Loadly API is running. Build frontend/ for web UI.",
            "version": "1.5.0",
            "documentation": "/docs",
            "redoc": "/redoc",
            "health_check": "/api/health"
        }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
