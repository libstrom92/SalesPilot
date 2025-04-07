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

class GPTAnalyzer: