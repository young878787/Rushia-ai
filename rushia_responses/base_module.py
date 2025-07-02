#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回應模組基類
提供所有回應模組的共同介面和基礎功能
"""

import random
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

class BaseResponseModule:
    """回應模組基類"""
    
    def __init__(self, chat_instance):
        """
        初始化回應模組
        
        Args:
            chat_instance: RushiaLoRAChat 實例
        """
        self.chat_instance = chat_instance
        self.module_name = self.__class__.__name__
        logger.info(f"初始化回應模組: {self.module_name}")
    
    def get_user_profile(self):
        """獲取用戶資料"""
        return self.chat_instance.user_profile if self.chat_instance else {}
    
    def get_conversation_history(self):
        """獲取對話歷史"""
        return self.chat_instance.conversation_history if self.chat_instance else []
    
    def get_taiwan_time(self):
        """獲取台灣時間"""
        taiwan_tz = timezone(timedelta(hours=8))
        return datetime.now(taiwan_tz)
    
    def get_user_name_suffix(self):
        """獲取用戶名稱後綴"""
        user_name = self.get_user_profile().get('name', '')
        return f"{user_name}♪" if user_name else "♪"
    
    def format_response_with_emotion(self, base_response, emotion_level='normal'):
        """
        為回應添加情感符號
        
        Args:
            base_response: 基礎回應文字
            emotion_level: 情感強度 ('low', 'normal', 'high')
        """
        if not base_response:
            return base_response
        
        emotions = {
            'low': ['♪', '♡'],
            'normal': ['♪', '♡', '～'],
            'high': ['♪♡', '♡♪', '～♪', '♡～']
        }
        
        emotion_symbols = emotions.get(emotion_level, emotions['normal'])
        
        if not any(symbol in base_response for symbol in ['♪', '♡', '～']):
            base_response += random.choice(emotion_symbols)
        
        return base_response
    
    def is_keyword_match(self, user_input, keywords):
        """
        檢查用戶輸入是否包含關鍵字
        
        Args:
            user_input: 用戶輸入
            keywords: 關鍵字列表
        """
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in keywords)
    
    def get_random_response(self, responses):
        """獲取隨機回應"""
        return random.choice(responses) if responses else None
    
    def log_response(self, user_input, response, response_type="general"):
        """記錄回應（用於調試）"""
        logger.debug(f"[{self.module_name}] {response_type} - 輸入: {user_input[:30]}... 回應: {response[:50]}...")
    
    def validate_response(self, response):
        """驗證回應是否合適"""
        if not response:
            return False
        
        if len(response.strip()) < 2:
            return False
        
        # 檢查是否包含不當內容
        inappropriate_terms = ['錯誤', '無法', '系統', '程式', 'AI', '助手']
        if any(term in response for term in inappropriate_terms):
            return False
        
        return True
    
    def get_response(self, user_input):
        """
        主要回應方法 - 子類應該覆寫此方法
        
        Args:
            user_input: 用戶輸入
            
        Returns:
            str: 回應文字，如果無法處理則返回 None
        """
        raise NotImplementedError("子類必須實現 get_response 方法")
