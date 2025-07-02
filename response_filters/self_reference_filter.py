#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自稱優化過濾器
優化露西亞的自稱方式，讓語調更自然可愛
"""

import re
import logging
import random
from typing import Dict, Any, Optional, Tuple
from .base_filter import BaseResponseFilter

logger = logging.getLogger(__name__)

class SelfReferenceFilter(BaseResponseFilter):
    """自稱優化過濾器 - 優化露西亞的自稱方式"""
    
    def __init__(self, chat_instance=None):
        super().__init__(chat_instance)
        
        # 預編譯自稱優化相關的正則表達式
        self._compile_patterns()
        
        logger.info("自稱優化過濾器初始化完成")
    
    def _compile_patterns(self):
        """預編譯自稱優化相關的正則表達式模式"""
        
        # 基礎系統詞替換模式
        system_replacement_patterns = [
            (r'\b(?:AI|ai|助手|助理|機器人|机器人|程式|程序|系統|系统)\b', '露西亞'),
        ]
        
        self.system_patterns = [(re.compile(pattern, re.IGNORECASE), replacement) 
                               for pattern, replacement in system_replacement_patterns]
        
        # 動作相關自稱模式（從主程式遷移並增強）
        action_patterns = [
            (r'\b露西[亞](會|能|可以|想要|希望|覺得|觉得|認為|认为|喜歡|喜欢|愛|爱|想|在|正在)', r'我\1'),
        ]
        
        self.action_patterns = [(re.compile(pattern), replacement) 
                               for pattern, replacement in action_patterns]
        
        # 狀態相關自稱模式
        state_patterns = [
            (r'\b露西[亞](的|也|還|还|就|都|只|才|很|真的|總是|总是|今天|現在|现在|昨天|明天)', 
             lambda m: random.choice(['露醬我', '露醬', '我']) + m.group(1)),
        ]
        
        self.state_patterns = [(re.compile(pattern), replacement) 
                              for pattern, replacement in state_patterns]
        
        # 情感表達自稱模式
        emotion_patterns = [
            (r'\b露西[亞](心情|感覺|感觉|覺得.*開心|开心|高興|高兴|難過|难过|快樂|快乐)', 
             lambda m: random.choice(['露醬我', '我']) + m.group(1)),
        ]
        
        self.emotion_patterns = [(re.compile(pattern), replacement) 
                                for pattern, replacement in emotion_patterns]
        
        # 對話開頭自稱模式
        beginning_patterns = [
            (r'^露西[亞](說|说|話|话|想|覺得|觉得|會|会|能|可以)', 
             lambda m: random.choice(['露醬我', '露醬', '我']) + m.group(1)),
        ]
        
        self.beginning_patterns = [(re.compile(pattern), replacement) 
                                  for pattern, replacement in beginning_patterns]
        
        # 時間相關自稱模式
        time_patterns = [
            (r'\b露西[亞](今天|現在|现在|昨天|明天|最近|剛才|刚才)', 
             lambda m: random.choice(['露醬我', '我']) + m.group(1)),
        ]
        
        self.time_patterns = [(re.compile(pattern), replacement) 
                             for pattern, replacement in time_patterns]
        
        # 一般情況的露西亞替換（最後處理，避免遺漏）
        general_patterns = [
            (r'\b露西[亞](?![♪♡～])', 
             lambda m: random.choice(['露醬我', '露醬', '我'])),
        ]
        
        self.general_patterns = [(re.compile(pattern), replacement) 
                                for pattern, replacement in general_patterns]
        
        # 可愛自稱變化模式（30%機率）
        cute_addition_patterns = [
            (r'\b我(很|真的|好)', r'露醬我\1'),
            (r'\b我(想|希望|覺得)', lambda m: random.choice(['露醬我', '我']) + m.group(1)),
        ]
        
        self.cute_patterns = [(re.compile(pattern), replacement) 
                             for pattern, replacement in cute_addition_patterns]
        
        # 重複修正模式
        duplicate_patterns = [
            (r'露醬我露醬', '露醬我'),
            (r'露醬露醬', '露醬'),
            (r'我我', '我'),
        ]
        
        self.duplicate_patterns = [(re.compile(pattern), replacement) 
                                  for pattern, replacement in duplicate_patterns]
        
        logger.debug(f"預編譯了自稱優化相關模式: {len(self.system_patterns) + len(self.action_patterns) + len(self.state_patterns) + len(self.emotion_patterns) + len(self.beginning_patterns) + len(self.time_patterns) + len(self.general_patterns) + len(self.cute_patterns) + len(self.duplicate_patterns)} 個")
    
    def filter(self, response: str, user_input: str = "", context: Dict = None) -> str:
        """
        優化自稱方式
        
        Args:
            response: 原始回應
            user_input: 用戶輸入
            context: 對話上下文
            
        Returns:
            str: 優化後的回應
        """
        if not response:
            return response
        
        original_response = response
        
        # 第一階段：基礎系統詞替換
        for pattern, replacement in self.system_patterns:
            response = pattern.sub(replacement, response)
        
        # 第二階段：動作相關自稱替換
        for pattern, replacement in self.action_patterns:
            if callable(replacement):
                response = pattern.sub(replacement, response)
            else:
                response = pattern.sub(replacement, response)
        
        # 第三階段：狀態相關自稱替換
        for pattern, replacement in self.state_patterns:
            if callable(replacement):
                response = pattern.sub(replacement, response)
            else:
                response = pattern.sub(replacement, response)
        
        # 第四階段：情感表達自稱替換
        for pattern, replacement in self.emotion_patterns:
            if callable(replacement):
                response = pattern.sub(replacement, response)
            else:
                response = pattern.sub(replacement, response)
        
        # 第五階段：對話開頭自稱替換
        for pattern, replacement in self.beginning_patterns:
            if callable(replacement):
                response = pattern.sub(replacement, response)
            else:
                response = pattern.sub(replacement, response)
        
        # 第六階段：時間相關自稱替換
        for pattern, replacement in self.time_patterns:
            if callable(replacement):
                response = pattern.sub(replacement, response)
            else:
                response = pattern.sub(replacement, response)
        
        # 第七階段：一般情況的露西亞替換
        for pattern, replacement in self.general_patterns:
            if callable(replacement):
                response = pattern.sub(replacement, response)
            else:
                response = pattern.sub(replacement, response)
        
        # 第八階段：隨機添加可愛的自稱變化（30%機率）
        if random.random() < 0.3:
            for pattern, replacement in self.cute_patterns:
                if callable(replacement):
                    response = pattern.sub(replacement, response, count=1)
                else:
                    response = pattern.sub(replacement, response, count=1)
        
        # 第九階段：修正可能出現的重複
        for pattern, replacement in self.duplicate_patterns:
            response = pattern.sub(replacement, response)
        
        return response
    
    def validate(self, response: str, user_input: str = "", context: Dict = None) -> Tuple[bool, str]:
        """驗證自稱是否合適"""
        if not response:
            return False, "empty_response"
        
        # 檢查是否還有未處理的系統詞彙
        system_terms = ['AI助手', 'AI助理', '機器人', '助手', '程式', '系統']
        for term in system_terms:
            if term in response:
                return False, f"contains_system_term:{term}"
        
        # 檢查是否有過多的重複
        if '露醬我露醬' in response or '露醬露醬露醬' in response:
            return False, "excessive_repetition"
        
        return True, "passed"
    
    def optimize_self_reference_style(self, response: str, style: str = "balanced") -> str:
        """
        根據指定風格優化自稱
        
        Args:
            response: 回應內容
            style: 優化風格 ('cute', 'natural', 'balanced')
            
        Returns:
            str: 優化後的回應
        """
        if style == "cute":
            # 更多使用「露醬我」、「露醬」
            response = re.sub(r'\b我(?=[很真的好想希望覺得])', '露醬我', response)
        elif style == "natural":
            # 更多使用「我」
            response = re.sub(r'\b露醬我(?![很真的好])', '我', response)
            response = re.sub(r'\b露醬(?![我很真的好])', '我', response)
        elif style == "balanced":
            # 平衡使用，保持原有邏輯
            pass
        
        return response
    
    def get_reference_stats(self) -> Dict[str, Any]:
        """獲取自稱優化統計"""
        return {
            'system_patterns': len(self.system_patterns),
            'action_patterns': len(self.action_patterns),
            'state_patterns': len(self.state_patterns),
            'emotion_patterns': len(self.emotion_patterns),
            'beginning_patterns': len(self.beginning_patterns),
            'time_patterns': len(self.time_patterns),
            'general_patterns': len(self.general_patterns),
            'cute_patterns': len(self.cute_patterns),
            'duplicate_patterns': len(self.duplicate_patterns),
            'total_patterns': (len(self.system_patterns) + len(self.action_patterns) + 
                             len(self.state_patterns) + len(self.emotion_patterns) + 
                             len(self.beginning_patterns) + len(self.time_patterns) + 
                             len(self.general_patterns) + len(self.cute_patterns) + 
                             len(self.duplicate_patterns))
        }
