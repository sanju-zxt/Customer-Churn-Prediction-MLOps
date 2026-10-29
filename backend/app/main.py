"""
Backend FastAPI application for real-time landmark streaming.
Sets up binary WebSocket protocol (MessagePack) for sub-100ms inference pipeline.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from backend.app.api.recognition import router as recognition_router

PROJECT_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    yield
    app.state.model = None


app = FastAPI(
    title="Real-Time Landmark Recognition API",
    description="Binary MessagePack WebSocket pipeline for ultra-low-latency landmark inference.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(recognition_router, prefix="/api")
