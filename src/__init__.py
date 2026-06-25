"""
X-OCR source package initialiser.
Imports key classes for convenient top-level access.
"""
from src.preprocessing import Preprocessor
from src.layout import LayoutAnalyser
from src.ocr import TrOCREngine
from src.candidates import OCRCandidate, CandidateSet, build_candidate_sets
from src.context import ContextReasoner
from src.xai import XAIGenerator
from src.explanation import ExplanationAgent
from src.pipeline import OCRPipeline

__all__ = [
    "Preprocessor",
    "LayoutAnalyser",
    "TrOCREngine",
    "OCRCandidate",
    "CandidateSet",
    "build_candidate_sets",
    "ContextReasoner",
    "XAIGenerator",
    "ExplanationAgent",
    "OCRPipeline",
]
