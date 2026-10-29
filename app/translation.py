"""Text-to-sign translation module using GrammarTranslator."""
from pathlib import Path
from app.language.grammar_translator import GrammarTranslator

translator = GrammarTranslator(plugin="ISL")


def translate_text(text: str) -> dict:
    return translator.translate_full(text)


def correct_gloss(gloss_sequence: list[str], plugin: str = "ISL") -> dict:
    grammar = GrammarTranslator(plugin=plugin)
    corrected = grammar.plugin.apply(gloss_sequence)
    return {
        "original": gloss_sequence,
        "corrected": corrected,
        "animation_segments": grammar.build_animation(
            corrected,
            grammar.to_hamnosys(corrected),
            "",
        ),
        "fallback_chain": grammar.build_fallback_chain(corrected),
    }
