import os
import logging
import json
from typing import Optional
from logging_config import setup_logging

logger = setup_logging("GPTAnalyzer")

class NoModelError(Exception):
    """Raised when neither OpenAI API or Hugging Face model is available"""
    pass

def analyze_notes(notes: str) -> dict:
    """Analyze notes using a simple rule-based approach when transformers is not available"""
    try:
        from transformers import pipeline
        # If transformers is available, use it
        analyzer = pipeline("text-classification")
        result = analyzer(notes)
        return result[0] if isinstance(result, list) and result else {"error": "Unexpected result format"}
    except ImportError:
        logger.warning("Transformers package not available, using simple analysis")
        # Simple fallback analysis
        return {
            "summary": notes,
            "analysis": "Basic analysis (transformers not available)",
            "sentiment": "neutral"
        }

def analyze_block(text: str) -> dict:
    """Analysera ett block av text (3–5 meningar) för kontext, behov, hinder, nästa steg."""
    # TODO: Lägg till dialektanalys och triggers för svenska säljsamtal
    # Här kan du byta till OpenAI/GPT-anrop om du vill
    return {
        "summary": f"Sammanfattning av block: {text[:100]}...",
        "needs": "(dummy) Identifierade behov",
        "obstacles": "(dummy) Identifierade hinder",
        "next_action": "(dummy) Föreslagen åtgärd"
    }

def analyze_context(full_text: str) -> dict:
    """Analysera hela samtalssektionen/blocket för djupare kontext, hinder, beslutspunkt."""
    # TODO: Integrera Hugging Face-modeller för svenska (t.ex. KB-BERT)
    # Dummy, byt mot GPT-anrop vid behov
    return {
        "summary": f"Sektionssammanfattning: {full_text[:120]}...",
        "decision_point": "(dummy) Beslutspunkt identifierad",
        "customer_goal": "(dummy) Kundens mål",
        "uncertainties": "(dummy) Osäkerheter"
    }

class GPTAnalyzer:
    pass