from fastapi import APIRouter

api_router = APIRouter(prefix="/api")

from . import tts

api_router.include_router(tts.router, tags=["tts"])
