from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.language.grammar_translator import GrammarTranslator

router = FastAPI()

translator = GrammarTranslator(plugin="ISL")


class TranslationRequest(BaseModel):
    text: str
    plugin: str = "ISL"


class GlossCorrection(BaseModel):
    text: str
    gloss_sequence: list[str]
    plugin: str = "ISL"


@router.post("/api/translation/text-to-isl")
def text_to_isl(req: TranslationRequest):
    result = translator.translate_full(req.text)
    return {
        "gloss_sequence": result["gloss_sequence"],
        "animation_segments": result["animation_segments"],
        "fallback_chain": result["fallback_chain"],
        "hamnosys": result["hamnosys"],
        "sigml": result["sigml"],
    }


@router.post("/api/translation/grammar")
def grammar_correction(req: GlossCorrection):
    corrected = translator.plugin.apply(req.gloss_sequence)
    return {
        "original_gloss": req.gloss_sequence,
        "corrected_gloss": corrected,
        "animation_segments": translator.build_animation(corrected, translator.to_hamnosys(corrected), ""),
        "fallback_chain": translator.build_fallback_chain(corrected),
    }


@router.post("/api/translation/grammar-correct")
def grammar_correct(req: TranslationRequest):
    result = translator.translate_full(req.text)
    return {
        "gloss_sequence": result["gloss_sequence"],
        "animation_segments": result["animation_segments"],
        "fallback_chain": result["fallback_chain"],
        "plugin_used": "ISL",
    }
