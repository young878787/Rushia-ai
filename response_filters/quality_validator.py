#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
品質驗證過濾器
最終驗證回應品質，確保符合露西亞的角色設定
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple, List
from .base_filter import BaseResponseFilter

logger = logging.getLogger(__name__)

class QualityValidatorFilter(BaseResponseFilter):
    """品質驗證過濾器 - 最終品質檢查和驗證"""
    
    def __init__(self, chat_instance=None):
        super().__init__(chat_instance)
        
        # 初始化驗證規則
        self._initialize_validation_rules()
        
        logger.info("品質驗證過濾器初始化完成")
    
    def _initialize_validation_rules(self):
        """初始化所有驗證規則"""
        
        # 不當內容關鍵詞（僅保留真正有問題的詞彙）
        self.bad_words = [
            'AI助手', '小智', '露西安', '頻道', '系統維護', 
            '奇怪', '妥當', '回答過', '燻肉', '肉味', '對話歷史', 
            '被發現', '艾瑞克', 'Erik', '他向前走', '搞什麼鬼', 
            '別以為', '不知道你在', '威脅', '搞鬼', 
            '繼續對話', '當前的情境', '回覆說.*然後'
        ]
        
        # 分析性語言指標（僅保留真正的分析性語言）
        self.analysis_indicators = [
            '根據設定', '現在需要', '我要以.*身份', '按照.*設定', '作為.*角色', 
            '首先.*分析', '回覆".*"', '處理.*請求', '給出回應',
            '明白了.*那我們', '繼續對話', '當前.*情境', '回覆說.*然後'
        ]
        
        # 角色混亂指標
        self.character_confusion_indicators = [
            '艾瑞克', 'Erik', '他向前走', '搞什麼鬼', '別以為', 
            '不知道你在', '威脅', '搞鬼', '角色扮演'
        ]
        
        # 商業/系統用語
        self.business_terms = [
            '時段已經關閉', '關閉', '營業時間', '服務暫停', '系統維護',
            '請稍後再試', '暫不提供', '功能暫停', '無法處理'
        ]
        
        # 不完整詞彙結尾（只檢查真正不完整的結尾）
        self.incomplete_endings = [
            '然後', '接著', '所以', '因此', '而且', '或者', '但是',
            '不過', '只是', '也就', '比如'
        ]
        
        # 溫暖指標（正向）
        self.warm_indicators = [
            '呢', '哦', '啊', '嗯', '♪', '♡', '～', '想', '喜歡', '開心',
            '溫暖', '陪伴', '一起', '愛', '關心', '照顧', '幸福'
        ]
        
        logger.debug(f"初始化驗證規則完成: {len(self.bad_words)} 個不當詞彙, {len(self.warm_indicators)} 個溫暖指標")
    
    def filter(self, response: str, user_input: str = "", context: Dict = None) -> str:
        """
        品質驗證和最終修正
        
        Args:
            response: 原始回應
            user_input: 用戶輸入
            context: 對話上下文
            
        Returns:
            str: 驗證後的回應，如果不合格則返回空字符串
        """
        if not response:
            return response or ""
        
        # 進行全面的品質檢查
        is_valid, reason = self.validate(response, user_input, context)
        
        if not is_valid:
            logger.warning(f"回應未通過品質驗證: {reason}, 內容: {response[:50]}...")
            # 返回空字符串表示需要重新生成
            return ""
        
        # 如果通過驗證，進行最終優化
        response = self._final_optimization(response, user_input, context)
        
        return response
    
    def validate(self, response: str, user_input: str = "", context: Dict = None) -> Tuple[bool, str]:
        """
        綜合品質驗證（從主程式 _validate_response_quality 遷移並增強）
        
        Args:
            response: 要驗證的回應
            user_input: 用戶輸入
            context: 對話上下文
            
        Returns:
            Tuple[bool, str]: (是否通過驗證, 失敗原因)
        """
        # 強化 None 檢查
        if response is None:
            return False, "response_is_none"
        
        response_str = str(response)  # 確保是字符串
        
        if not response_str or len(response_str.strip()) < 2:
            return False, "too_short_or_empty"
        
        # 檢查 1: 長度合理性
        if len(response_str.strip()) > 200:
            return False, "too_long"
        
        # 檢查 2: 不當詞彙
        for bad_word in self.bad_words:
            if bad_word in response_str:
                return False, f"contains_bad_word:{bad_word}"
        
        # 檢查 3: 模型思考過程洩露
        for indicator in self.analysis_indicators:
            if indicator in response_str:
                return False, f"analysis_language_leak:{indicator}"
        
        # 檢查 4: 角色混亂
        for indicator in self.character_confusion_indicators:
            if indicator in response_str:
                return False, f"character_confusion:{indicator}"
        
        # 檢查 5: 不當的對話格式（引號+他）
        if re.search(r'"[^"]*".*?他', response_str):
            return False, "inappropriate_dialogue_format"
        
        # 檢查 6: 商業/系統用語
        for term in self.business_terms:
            if term in response_str:
                return False, f"business_language:{term}"
        
        # 檢查 7: 重複的句段或詞語
        if re.search(r'(.{4,})\1{2,}', response_str):
            return False, "excessive_repetition"
        
        # 檢查 8: 不完整的詞彙結尾
        for ending in self.incomplete_endings:
            if response_str.strip().endswith(ending):
                return False, f"incomplete_ending:{ending}"
        
        # 檢查 9: 過多的重複字符
        if re.search(r'(.)\1{4,}', response_str):
            return False, "excessive_character_repetition"
        
        # 檢查 10: 內容相關性（簡單檢查）
        if not self._check_content_relevance(response_str, user_input):
            return False, "low_relevance"
        
        return True, "passed"
    
    def _check_content_relevance(self, response: str, user_input: str) -> bool:
        """
        檢查內容相關性
        
        Args:
            response: 回應內容
            user_input: 用戶輸入
            
        Returns:
            bool: 是否相關
        """
        # 如果用戶輸入為空，總是認為相關
        if not user_input.strip():
            return True
        
        # 檢查是否包含溫暖指標（露西亞的基本特徵）
        warm_count = sum(1 for indicator in self.warm_indicators if indicator in response)
        
        # 至少要有一個溫暖指標
        return warm_count >= 1
    
    def _final_optimization(self, response: str, user_input: str, context: Dict = None) -> str:
        """
        最終優化處理
        
        Args:
            response: 回應內容
            user_input: 用戶輸入
            context: 對話上下文
            
        Returns:
            str: 優化後的回應
        """
        if not response:
            return response or ""
        
        # 確保有適當的表情符號
        if not any(symbol in response for symbol in ['♪', '♡', '～']):
            # 在適當位置添加表情符號
            if response.endswith('。') or response.endswith('！') or response.endswith('？'):
                response = response[:-1] + '♪'
            else:
                response += '♪'
        
        # 確保語調溫暖
        if not any(indicator in response for indicator in ['呢', '哦', '啊', '嗯']):
            # 如果缺乏溫暖語調，可以在末尾添加
            if len(response) < 20:  # 短回應
                response += '呢♡'
        
        return response
    
    def get_quality_score(self, response: str, user_input: str = "", context: Dict = None) -> Dict[str, Any]:
        """
        獲取回應品質評分
        
        Args:
            response: 回應內容
            user_input: 用戶輸入
            context: 對話上下文
            
        Returns:
            Dict: 品質評分詳情
        """
        # 強化 None 檢查
        if response is None:
            return {
                'total_score': 0,
                'details': {'error': 'response_is_none'},
                'passed': False
            }
        
        response_str = str(response)  # 確保是字符串
        
        score = 0
        max_score = 100
        details = {}
        
        # 長度評分 (20分)
        length = len(response_str.strip())
        if 10 <= length <= 50:
            length_score = 20
        elif 5 <= length <= 80:
            length_score = 15
        else:
            length_score = 5
        
        score += length_score
        details['length_score'] = length_score
        
        # 表情符號評分 (15分)
        emoji_count = sum(1 for symbol in ['♪', '♡', '～'] if symbol in response_str)
        emoji_score = min(emoji_count * 5, 15)
        score += emoji_score
        details['emoji_score'] = emoji_score
        
        # 溫暖度評分 (25分)
        warm_count = sum(1 for indicator in self.warm_indicators if indicator in response_str)
        warm_score = min(warm_count * 5, 25)
        score += warm_score
        details['warm_score'] = warm_score
        
        # 清潔度評分 (20分) - 不包含不當內容
        has_bad_content = any(bad_word in response_str for bad_word in self.bad_words)
        clean_score = 0 if has_bad_content else 20
        score += clean_score
        details['clean_score'] = clean_score
        
        # 角色一致性評分 (20分) - 不包含角色混亂
        has_confusion = any(indicator in response_str for indicator in self.character_confusion_indicators)
        consistency_score = 0 if has_confusion else 20
        score += consistency_score
        details['consistency_score'] = consistency_score
        
        details['total_score'] = score
        details['max_score'] = max_score
        details['grade'] = self._get_grade(score, max_score)
        
        return details
    
    def _get_grade(self, score: int, max_score: int) -> str:
        """根據分數獲取等級"""
        percentage = (score / max_score) * 100
        
        if percentage >= 90:
            return "A+"
        elif percentage >= 80:
            return "A"
        elif percentage >= 70:
            return "B"
        elif percentage >= 60:
            return "C"
        else:
            return "D"
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """獲取驗證統計"""
        return {
            'bad_words_count': len(self.bad_words),
            'analysis_indicators_count': len(self.analysis_indicators),
            'character_confusion_indicators_count': len(self.character_confusion_indicators),
            'business_terms_count': len(self.business_terms),
            'incomplete_endings_count': len(self.incomplete_endings),
            'warm_indicators_count': len(self.warm_indicators),
            'total_validation_rules': (len(self.bad_words) + len(self.analysis_indicators) + 
                                     len(self.character_confusion_indicators) + len(self.business_terms) + 
                                     len(self.incomplete_endings))
        }
