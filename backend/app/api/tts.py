import io
import wave
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter()

SUPPORTED_VOICES = {
    "en_US-less-medium": "en_US/less-medium.onnx",
    "en_US-amy-medium": "en_US/amy/medium.onnx",
    "en_US-libritts-medium": "en_US/libritts/high.onnx",
}

PIPER_VOICE_CACHE: dict[str, "PiperVoice"] = {}


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default="en_US-less-medium", description="Voice identifier")
    ssml: Optional[bool] = Field(default=False, description="Text is SSML")


def generate_silent_wav(duration_s: float = 0.1, sample_rate: int = 22050) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        n_samples = int(duration_s * sample_rate)
        wf.writeframes(int(0).to_bytes(2, "little") * n_samples)
    return buf.getvalue()


def generate_pipertts_wav(text: str, voice: str) -> bytes:
    from piper import PiperVoice, SynthesisConfig

    model_path = SUPPORTED_VOICES.get(voice)
    if model_path is None:
        raise ValueError(
            f"Voice '{voice}' not found. Available: {list(SUPPORTED_VOICES.keys())}"
        )

    if model_path not in PIPER_VOICE_CACHE:
        piper_voice = PiperVoice.load(model_path)
        PIPER_VOICE_CACHE[model_path] = piper_voice
    else:
        piper_voice = PIPER_VOICE_CACHE[model_path]

    syn_config = SynthesisConfig(
        noise_scale=0.8,
        noise_w_scale=0.8,
        length_scale=1.0,
    )

    buf = io.BytesIO()
    wav_file = wave.open(buf, "wb")
    piper_voice.synthesize_wav(text, wav_file=wav_file, syn_config=syn_config)
    wav_file.close()

    return buf.getvalue()


@router.post("/tts")
async def text_to_speech(req: TTSRequest):
    try:
        voice_key = req.voice
        if voice_key not in SUPPORTED_VOICES:
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{voice_key}' not supported. Available: {list(SUPPORTED_VOICES.keys())}",
            )

        try:
            audio_data = generate_pipertts_wav(req.text, voice_key)
        except Exception:
            audio_data = generate_silent_wav(duration_s=0.5)

        def iter_wav():
            yield audio_data

        return StreamingResponse(
            iter_wav(),
            media_type="audio/wav",
            headers={"Content-Disposition": "inline", "X-Voice": voice_key},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
