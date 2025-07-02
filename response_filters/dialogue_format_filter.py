#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
對話格式過濾器
清理對話格式標記、冒號分隔等格式問題
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple
from .base_filter import BaseResponseFilter

logger = logging.getLogger(__name__)

class DialogueFormatFilter(BaseResponseFilter):
    """對話格式過濾器 - 清理各種對話格式標記"""
    
    def __init__(self, chat_instance=None):
        super().__init__(chat_instance)
        
        # 預編譯對話格式相關的正則表達式
        self._compile_patterns()
        
        logger.info("對話格式過濾器初始化完成")
    
    def _compile_patterns(self):
        """預編譯對話格式相關的正則表達式模式"""
        
        # 基本對話格式模式（從主程式遷移）
        dialogue_patterns = [
            r'\s*(安|用戶|使用者|用户|USER|User|你)[:：]\s*.*$',
            r'\s*(露西[亞亜雅安asia西亚]*|るしあ|rushia)[:：]\s*',
            r'^[^♪♡～]*[:：]\s*',  # 移除任何以冒號開頭的格式
            r'\s+[:：]\s*',  # 移除句中的冒號格式
        ]
        
        self.dialogue_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in dialogue_patterns]
        
        # 聊天室格式模式
        chatroom_patterns = [
            r'^\[.*?\]\s*',  # [時間] 或 [用戶名]
            r'^<.*?>\s*',    # <用戶名>
            r'^\*.*?\*\s*',  # *動作描述*
            r'^【.*?】\s*',   # 【標題】
            r'^\(.*?\)\s*',  # (旁白)
        ]
        
        self.chatroom_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in chatroom_patterns]
        
        # 引用格式模式
        quote_patterns = [
            r'^>\s*.*$',     # > 引用
            r'^\|\s*.*$',    # | 引用
            r'^「.*?」.*?說',  # 「話語」某人說
            r'^".*?".*?said', # "quote" someone said
        ]
        
        self.quote_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in quote_patterns]
        
        # 元資訊格式模式
        meta_patterns = [
            r'^\d{1,2}:\d{2}.*?',  # 時間戳
            r'^@\w+\s*',           # @提及
            r'^#\w+\s*',           # #標籤
            r'^\w+\s*>>.*?',       # 回覆格式
        ]
        
        self.meta_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in meta_patterns]
        
        logger.debug(f"預編譯了對話格式相關模式: {len(self.dialogue_patterns) + len(self.chatroom_patterns) + len(self.quote_patterns) + len(self.meta_patterns)} 個")
    
    def filter(self, response: str, user_input: str = "", context: Dict = None) -> str:
        """
        過濾對話格式標記
        
        Args:
            response: 原始回應
            user_input: 用戶輸入
            context: 對話上下文
            
        Returns:
            str: 清理後的回應
        """
        if not response:
            return response
        
        original_response = response
        
        # 第一階段：清理基本對話格式
        for pattern in self.dialogue_patterns:
            response = pattern.sub('', response)
        
        # 第二階段：清理聊天室格式
        for pattern in self.chatroom_patterns:
            response = pattern.sub('', response)
        
        # 第三階段：清理引用格式
        for pattern in self.quote_patterns:
            response = pattern.sub('', response)
        
        # 第四階段：清理元資訊格式
        for pattern in self.meta_patterns:
            response = pattern.sub('', response)
        
        # 第五階段：清理多餘的空白和換行
        response = re.sub(r'\n+', ' ', response)  # 換行轉空格
        response = re.sub(r'\s+', ' ', response.strip())  # 多空格合併
        
        # 第六階段：清理連續的標點符號
        response = re.sub(r'[。！？]{3,}', '。', response)  # 過多句號
        response = re.sub(r'[♪♡～]{4,}', '♪♡', response)  # 過多表情符號
        
        # 第七階段：檢查清理效果
        if not response.strip() and original_response.strip():
            logger.warning(f"對話格式過濾器清空了回應: {original_response[:50]}...")
            # 嘗試保留一些基本內容
            return self._attempt_recovery(original_response)
        
        return response
    
    def _attempt_recovery(self, original_response: str) -> str:
        """
        嘗試從過度清理的回應中恢復一些內容
        
        Args:
            original_response: 原始回應
            
        Returns:
            str: 恢復後的回應
        """
        # 嘗試找到沒有格式標記的純文字部分
        lines = original_response.split('\n')
        
        for line in lines:
            # 移除明顯的格式標記後檢查
            clean_line = re.sub(r'^[^\w]*', '', line.strip())
            clean_line = re.sub(r'[:：].*$', '', clean_line)
            
            if len(clean_line) > 3 and not re.search(r'^(用戶|用户|USER)', clean_line):
                return clean_line
        
        # 如果都無法恢復，返回安全回應
        return "嗯嗯♪我在聽呢～♡"
    
    def validate(self, response: str, user_input: str = "", context: Dict = None) -> Tuple[bool, str]:
        """驗證回應是否包含對話格式標記"""
        if not response:
            return False, "empty_response"
        
        # 檢查是否包含明顯的對話格式
        format_indicators = [
            '用戶:', '用户:', 'USER:', 'User:',
            '露西亞:', 'rushia:', 'Rushia:',
            '[', ']', '<', '>',
            '「', '」', '"', '"'
        ]
        
        for indicator in format_indicators:
            if indicator in response:
                return False, f"contains_dialogue_format:{indicator}"
        
        # 檢查是否有冒號分隔格式
        if re.search(r'^[^♪♡～]*[:：]', response):
            return False, "contains_colon_format"
        
        # 檢查是否有時間戳格式
        if re.search(r'^\d{1,2}:\d{2}', response):
            return False, "contains_timestamp"
        
        return True, "passed"
    
    def clean_specific_format(self, response: str, format_type: str) -> str:
        """
        清理特定類型的格式
        
        Args:
            response: 回應內容
            format_type: 格式類型 ('dialogue', 'chatroom', 'quote', 'meta')
            
        Returns:
            str: 清理後的回應
        """
        if format_type == 'dialogue':
            for pattern in self.dialogue_patterns:
                response = pattern.sub('', response)
        elif format_type == 'chatroom':
            for pattern in self.chatroom_patterns:
                response = pattern.sub('', response)
        elif format_type == 'quote':
            for pattern in self.quote_patterns:
                response = pattern.sub('', response)
        elif format_type == 'meta':
            for pattern in self.meta_patterns:
                response = pattern.sub('', response)
        
        return response.strip()
    
    def get_format_stats(self) -> Dict[str, int]:
        """獲取格式清理統計"""
        return {
            'dialogue_patterns': len(self.dialogue_patterns),
            'chatroom_patterns': len(self.chatroom_patterns),
            'quote_patterns': len(self.quote_patterns),
            'meta_patterns': len(self.meta_patterns),
            'total_patterns': (len(self.dialogue_patterns) + 
                             len(self.chatroom_patterns) + 
                             len(self.quote_patterns) + 
                             len(self.meta_patterns))
        }
