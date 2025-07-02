#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回應過濾器基礎類
定義所有過濾器的統一介面
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class BaseResponseFilter(ABC):
    """回應過濾器抽象基類"""
    
    def __init__(self, chat_instance=None):
        """
        初始化過濾器
        
        Args:
            chat_instance: RushiaLoRAChat 實例
        """
        self.chat_instance = chat_instance
        self.filter_name = self.__class__.__name__
        self.enabled = True
        
        # 統計資料
        self.stats = {
            'processed_count': 0,
            'modified_count': 0,
            'rejected_count': 0,
            'error_count': 0
        }
        
        logger.debug(f"初始化過濾器: {self.filter_name}")
    
    @abstractmethod
    def filter(self, response: str, user_input: str = "", context: Dict = None) -> str:
        """
        過濾回應內容（抽象方法）
        
        Args:
            response: 原始回應
            user_input: 用戶輸入
            context: 對話上下文
            
        Returns:
            str: 過濾後的回應
        """
        pass
    
    def validate(self, response: str, user_input: str = "", context: Dict = None) -> Tuple[bool, str]:
        """
        驗證回應是否合適（可選實現）
        
        Args:
            response: 要驗證的回應
            user_input: 用戶輸入  
            context: 對話上下文
            
        Returns:
            Tuple[bool, str]: (是否通過驗證, 失敗原因)
        """
        return True, "not_implemented"
    
    def is_enabled(self) -> bool:
        """檢查過濾器是否啟用"""
        return self.enabled
    
    def enable(self):
        """啟用過濾器"""
        self.enabled = True
        logger.info(f"過濾器 {self.filter_name} 已啟用")
    
    def disable(self):
        """禁用過濾器"""
        self.enabled = False
        logger.info(f"過濾器 {self.filter_name} 已禁用")
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取過濾器統計資料"""
        total = max(self.stats['processed_count'], 1)
        return {
            'filter_name': self.filter_name,
            'enabled': self.enabled,
            'processed_count': self.stats['processed_count'],
            'modified_count': self.stats['modified_count'],
            'rejected_count': self.stats['rejected_count'],
            'error_count': self.stats['error_count'],
            'modification_rate': (self.stats['modified_count'] / total) * 100,
            'rejection_rate': (self.stats['rejected_count'] / total) * 100,
            'error_rate': (self.stats['error_count'] / total) * 100
        }
    
    def reset_stats(self):
        """重置統計資料"""
        self.stats = {
            'processed_count': 0,
            'modified_count': 0,
            'rejected_count': 0,
            'error_count': 0
        }
        logger.debug(f"過濾器 {self.filter_name} 統計資料已重置")
    
    def _update_stats(self, was_modified: bool = False, was_rejected: bool = False, had_error: bool = False):
        """更新統計資料"""
        self.stats['processed_count'] += 1
        if was_modified:
            self.stats['modified_count'] += 1
        if was_rejected:
            self.stats['rejected_count'] += 1
        if had_error:
            self.stats['error_count'] += 1
    
    def _safe_filter(self, response: str, user_input: str = "", context: Dict = None) -> str:
        """安全的過濾執行，包含錯誤處理和統計"""
        if not self.enabled:
            return response
        
        if not response:
            self._update_stats()
            return response
        
        try:
            original_response = response
            filtered_response = self.filter(response, user_input, context)
            
            # 更新統計
            was_modified = original_response != filtered_response
            was_rejected = filtered_response == "[REJECTED]" or not filtered_response
            
            self._update_stats(was_modified=was_modified, was_rejected=was_rejected)
            
            return filtered_response
            
        except Exception as e:
            logger.error(f"過濾器 {self.filter_name} 執行失敗: {e}")
            self._update_stats(had_error=True)
            return response  # 返回原始回應，不因錯誤而中斷
