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

from api.auth import (
    ADMIN_KEY_ENV, KEY_STATUS_MISSING, KEY_STATUS_TOO_SHORT, MIN_ADMIN_KEY_LENGTH, admin_key_status
)
from api.routes import router as api_router
from models import load_model, load_model_metadata

load_dotenv()


def _start_update_scheduler():
    """Start the opt-in daily data-update thread (LOADLY_AUTO_UPDATE=1); never raises."""
    try:
        from utils.update_scheduler import DailyUpdateScheduler, scheduler_settings, STARTUP_DELAY_ENV, DEFAULT_STARTUP_DELAY_S
        from utils.daily_update import run_daily_update, refresh_caches_if_data_changed
        enabled, time_str = scheduler_settings()
        if not enabled:
            return None
        delay = float(os.getenv(STARTUP_DELAY_ENV, DEFAULT_STARTUP_DELAY_S))
        scheduler = DailyUpdateScheduler(run_daily_update, time_str, delay, refresh_caches_if_data_changed)
        scheduler.start()
        print(f"🔄 Automatic data update enabled: catch-up in {delay:.0f}s, then daily at {time_str} IST")
        return scheduler
    except Exception as exc:
        print(f"⚠️ Could not start the automatic data update scheduler: {exc}")
        return None


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

    key_status = admin_key_status()
    if key_status == KEY_STATUS_MISSING:
        print(f"🔒 Admin endpoints disabled: {ADMIN_KEY_ENV} is not set (they will return 503)")
    elif key_status == KEY_STATUS_TOO_SHORT:
        print(f"⚠️ Warning: {ADMIN_KEY_ENV} is shorter than {MIN_ADMIN_KEY_LENGTH} characters; "
              "admin endpoints are disabled until a stronger key is configured")
    else:
        print("✅ Admin key configured for operator endpoints")

    print(f"📖 Interactive API Documentation: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    scheduler = _start_update_scheduler()
    yield
    if scheduler is not None:
        scheduler.stop()
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
    # Auth is header-based (X-Admin-Key), not cookie-based, so credentialed CORS is not needed.
    allow_credentials=False,
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

    frontend_dist_real = os.path.realpath(frontend_dist)

    def _resolve_dist_file(full_path: str):
        """Return the real path of a file inside frontend/dist, or None.

        full_path arrives percent-decoded, so it may contain '..', backslashes,
        drive letters or absolute paths; anything resolving outside dist is refused.
        """
        if not full_path or "\\" in full_path or "\x00" in full_path:
            return None
        if os.path.isabs(full_path) or os.path.splitdrive(full_path)[0]:
            return None
        try:
            candidate = os.path.realpath(os.path.join(frontend_dist_real, full_path))
            root = os.path.normcase(frontend_dist_real)
            if os.path.commonpath([root, os.path.normcase(candidate)]) != root:
                return None
        except ValueError:  # e.g. different drives on Windows
            return None
        return candidate if os.path.isfile(candidate) else None

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

        file_path = _resolve_dist_file(full_path)
        if file_path is not None:
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist_real, "index.html"))
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
    # Auto-reload is a development convenience only; opt in with LOADLY_DEV=1.
    dev_mode = os.getenv("LOADLY_DEV", "").strip().lower() in ("1", "true", "yes")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=dev_mode)
