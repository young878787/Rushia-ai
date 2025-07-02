#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
語義分析系統模組
提供情感分析、意圖識別、親密度計算等功能
"""

from .emotion_analyzer import EmotionAnalyzer
from .intent_recognizer import IntentRecognizer
from .intimacy_calculator import IntimacyCalculator
from .context_analyzer import ContextAnalyzer
from .semantic_manager import SemanticAnalysisManager
from .keyword_config import keyword_config

__all__ = [
    'EmotionAnalyzer',
    'IntentRecognizer', 
    'IntimacyCalculator',
    'ContextAnalyzer',
    'SemanticAnalysisManager',
    'keyword_config'
]

__version__ = "1.0.0"
__author__ = "RushiaMode Team"
__description__ = "露西亞專用語義分析模組系統"
