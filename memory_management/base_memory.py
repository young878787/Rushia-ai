#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
記憶管理基礎模組
提供記憶管理系統的共同介面和基礎功能
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class BaseMemoryManager(ABC):
    """記憶管理模組基類"""
    
    def __init__(self, chat_instance=None):
        """
        初始化記憶管理模組
        
        Args:
            chat_instance: RushiaLoRAChat 實例
        """
        self.chat_instance = chat_instance
        self.module_name = self.__class__.__name__
        logger.info(f"初始化記憶管理模組: {self.module_name}")
    
    @abstractmethod
    def add(self, data: Any) -> bool:
        """
        添加資料到記憶中
        
        Args:
            data: 要添加的資料
            
        Returns:
            bool: 是否成功添加
        """
        pass
    
    @abstractmethod
    def get(self, query: Any) -> Any:
        """
        從記憶中獲取資料
        
        Args:
            query: 查詢條件
            
        Returns:
            Any: 查詢結果
        """
        pass
    
    @abstractmethod
    def update(self, query: Any, data: Any) -> bool:
        """
        更新記憶中的資料
        
        Args:
            query: 查詢條件
            data: 新的資料
            
        Returns:
            bool: 是否成功更新
        """
        pass
    
    @abstractmethod
    def cleanup(self, force: bool = False) -> int:
        """
        清理過期或無用的記憶
        
        Args:
            force: 是否強制清理
            
        Returns:
            int: 清理的項目數量
        """
        pass
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """獲取記憶統計資訊"""
        return {
            'module_name': self.module_name,
            'total_items': 0,
            'memory_usage': 0,
            'last_cleanup': None
        }
    
    def _get_current_timestamp(self) -> float:
        """獲取當前時間戳"""
        return time.time()
    
    def _is_expired(self, timestamp: float, max_age_days: int = 30) -> bool:
        """檢查時間戳是否過期"""
        max_age_seconds = max_age_days * 24 * 3600
        return (self._get_current_timestamp() - timestamp) > max_age_seconds
