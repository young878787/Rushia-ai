#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
溫柔甜蜜增強過濾器
專門增強回應的溫柔甜蜜程度
"""

import re
import logging
import random
from typing import Dict, Any, Optional, Tuple, List
from .base_filter import BaseResponseFilter

logger = logging.getLogger(__name__)

class SweetenessEnhancerFilter(BaseResponseFilter):
    """溫柔甜蜜增強過濾器"""
    
    def __init__(self, chat_instance=None):
        super().__init__(chat_instance)
        
        # 初始化增強規則
        self._initialize_enhancement_rules()
        
        logger.info("溫柔甜蜜增強過濾器初始化完成")
    
    def _initialize_enhancement_rules(self):
        """初始化增強規則"""
        
        # 溫柔詞彙增強
        self.gentle_enhancements = [
            (r'好的', '好的呢♪'),
            (r'是的', '是的呢♡'),
            (r'當然', '當然可以啦♪'),
            (r'沒關係', '沒關係的♡'),
            (r'不用擔心', '不用擔心啦♪露醬在這裡呢♡'),
            (r'很開心', '超級開心的♪'),
            (r'很高興', '非常高興呢♡'),
            (r'很棒', '超棒的♪'),
            (r'真的嗎', '真的嗎♡好期待呢♪'),
        ]
        
        # 親密稱呼增強
        self.intimate_enhancements = [
            (r'你', lambda m: random.choice(['你', '你♪', '親愛的', '寶貝'])),
            (r'(?<![\w♪♡～])和你(?![\w♪♡～])', '和你♡'),
            (r'(?<![\w♪♡～])陪你(?![\w♪♡～])', '陪著你♪'),
            (r'(?<![\w♪♡～])想你(?![\w♪♡～])', '想你呢♡'),
        ]
        
        # 情感表達增強
        self.emotion_enhancements = [
            (r'一起', '一起♪'),
            (r'陪伴', '溫暖的陪伴♡'),
            (r'溫暖', '暖暖的♡'),
            (r'開心', '開心得不得了♪'),
            (r'快樂', '超級快樂♡'),
            (r'喜歡', '超級喜歡♪'),
            (r'愛', '好愛好愛♡'),
        ]
        
        # 句尾溫柔化
        self.ending_enhancements = [
            (r'呢$', '呢♪'),
            (r'哦$', '哦♡'),
            (r'啊$', '啊♪'),
            (r'吧$', '吧♡'),
            (r'吧！$', '吧♪'),
            (r'了$', '了呢♡'),
        ]
        
        # 編譯正則表達式
        self.gentle_patterns = [(re.compile(pattern), replacement) 
                               for pattern, replacement in self.gentle_enhancements]
        self.intimate_patterns = [(re.compile(pattern), replacement) 
                                 for pattern, replacement in self.intimate_enhancements]
        self.emotion_patterns = [(re.compile(pattern), replacement) 
                                for pattern, replacement in self.emotion_enhancements]
        self.ending_patterns = [(re.compile(pattern), replacement) 
                               for pattern, replacement in self.ending_enhancements]
    
    def filter(self, response: str, user_input: str = "", context: Dict = None) -> str:
        """增強溫柔甜蜜程度"""
        if not response:
            return response
        
        original_response = response
        
        # 檢查是否需要增強
        if not self._should_enhance(response, user_input):
            return response
        
        # 第一階段：基礎溫柔詞彙增強
        for pattern, replacement in self.gentle_patterns:
            response = pattern.sub(replacement, response)
        
        # 第二階段：親密稱呼增強（適度使用）
        if random.random() < 0.3:  # 30% 機率使用親密稱呼
            for pattern, replacement in self.intimate_patterns:
                if callable(replacement):
                    response = pattern.sub(lambda m: replacement(m), response)
                else:
                    response = pattern.sub(replacement, response)
        
        # 第三階段：情感表達增強
        for pattern, replacement in self.emotion_patterns:
            response = pattern.sub(replacement, response)
        
        # 第四階段：句尾溫柔化
        for pattern, replacement in self.ending_patterns:
            response = pattern.sub(replacement, response)
        
        # 第五階段：添加溫柔的連接詞
        response = self._add_gentle_connectors(response)
        
        # 第六階段：確保適度（避免過度甜膩）
        response = self._moderate_sweetness(response)
        
        return response
    
    def _should_enhance(self, response: str, user_input: str) -> bool:
        """判斷是否需要增強"""
        # 檢查用戶輸入的情感傾向
        intimate_keywords = ['想', '愛', '喜歡', '陪', '一起', '溫暖', '甜蜜', '親密']
        user_wants_intimacy = any(keyword in user_input for keyword in intimate_keywords)
        
        # 檢查回應是否已經足夠甜蜜
        sweetness_count = response.count('♪') + response.count('♡') + response.count('～')
        is_already_sweet = sweetness_count >= 2
        
        # 如果用戶想要親密或回應不夠甜蜜，則增強
        return user_wants_intimacy or not is_already_sweet
    
    def _add_gentle_connectors(self, response: str) -> str:
        """添加溫柔的連接詞"""
        # 在句子之間添加溫柔的連接
        response = re.sub(r'♪\s*', '♪ ', response)
        response = re.sub(r'♡\s*', '♡ ', response)
        
        # 在適當的地方添加小停頓
        if len(response) > 20 and '…' not in response and '...' not in response:
            middle = len(response) // 2
            # 找到中間附近的句子分隔點
            for i in range(middle - 5, middle + 5):
                if i < len(response) and response[i] in '，。！？':
                    response = response[:i+1] + '♪ ' + response[i+1:]
                    break
        
        return response
    
    def _moderate_sweetness(self, response: str) -> str:
        """調節甜蜜程度，避免過度"""
        # 限制連續的表情符號
        response = re.sub(r'♪{3,}', '♪♪', response)
        response = re.sub(r'♡{3,}', '♡♡', response)
        response = re.sub(r'～{3,}', '～～', response)
        
        # 移除重複的符號組合
        response = re.sub(r'♪\s*♡\s*♪\s*♡', '♪♡', response)
        response = re.sub(r'♡\s*♪\s*♡\s*♪', '♡♪', response)
        
        # 移除重複的「呢♡」
        response = re.sub(r'(呢♡\s*){2,}', '呢♡', response)
        
        # 避免過多的「超級」
        response = re.sub(r'(超級.*?)超級', r'\1', response)
        
        # 清理多餘空格
        response = re.sub(r'\s{2,}', ' ', response)
        
        # 清理句尾重複符號
        response = re.sub(r'♪\s*♡\s*$', '♡', response)
        response = re.sub(r'♡\s*♪\s*$', '♪', response)
        
        return response
    
    def validate(self, response: str, user_input: str = "", context: Dict = None) -> Tuple[bool, str]:
        """驗證增強效果是否合適"""
        if not response:
            return False, "empty_response"
        
        # 檢查甜蜜程度是否適中
        sweetness_count = response.count('♪') + response.count('♡') + response.count('～')
        if sweetness_count > 6:
            return False, "too_sweet"
        
        # 檢查是否有重複的表達
        if re.search(r'(.{5,})\1', response):
            return False, "repetitive_sweetness"
        
        return True, "passed"
