#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上下文快取管理模組
負責管理對話上下文、情境快取、用戶情感狀態等短期記憶資料
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from .base_memory import BaseMemoryManager

logger = logging.getLogger(__name__)


class ContextCacheManager(BaseMemoryManager):
    """上下文快取管理器 - 管理對話上下文和情境快取"""
    
    def __init__(self, max_emotions: int = 30, max_themes: int = 20, 
                 max_preferences: int = 50, cleanup_interval: int = 3600):
        """
        初始化上下文快取管理器
        
        Args:
            max_emotions: 最大情感記錄數量
            max_themes: 最大主題記錄數量  
            max_preferences: 最大偏好記錄數量
            cleanup_interval: 清理間隔（秒）
        """
        super().__init__()
        self.max_emotions = max_emotions
        self.max_themes = max_themes
        self.max_preferences = max_preferences
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        
        # 初始化快取結構
        self._cache = {
            'user_emotions': [],  # [(emotion, intensity, timestamp), ...]
            'conversation_themes': [],  # [(theme, confidence, timestamp), ...]
            'user_expressed_affection': False,
            'last_topic_change': None,
            'intimate_level': 0,  # 0-5 親密度等級
            'current_mood': 'neutral',
            'user_preferences': {},  # {preference_type: [(value, weight, timestamp), ...]}
            'conversation_depth': 0,
            'last_interaction_time': None,
            'session_context': {},  # 當前會話上下文
            'temporal_patterns': {},  # 時間模式記錄
            'interaction_frequency': {}  # 互動頻率統計
        }
        
        logger.info("ContextCacheManager 初始化完成")
    
    def add(self, data_type: str, value: Any, **kwargs) -> bool:
        """
        添加上下文資料
        
        Args:
            data_type: 資料類型 ('emotion', 'theme', 'preference', 'session', 等)
            value: 資料值
            **kwargs: 額外參數（如權重、置信度等）
            
        Returns:
            bool: 是否成功添加
        """
        try:
            timestamp = time.time()
            
            if data_type == 'emotion':
                intensity = kwargs.get('intensity', 0.5)
                self._cache['user_emotions'].append((value, intensity, timestamp))
                self._limit_list_size('user_emotions', self.max_emotions)
                
            elif data_type == 'theme':
                confidence = kwargs.get('confidence', 0.5)
                self._cache['conversation_themes'].append((value, confidence, timestamp))
                self._limit_list_size('conversation_themes', self.max_themes)
                
            elif data_type == 'preference':
                preference_type = kwargs.get('type', 'general')
                weight = kwargs.get('weight', 1.0)
                
                if preference_type not in self._cache['user_preferences']:
                    self._cache['user_preferences'][preference_type] = []
                
                self._cache['user_preferences'][preference_type].append((value, weight, timestamp))
                self._limit_preference_size(preference_type)
                
            elif data_type == 'session':
                key = kwargs.get('key', 'general')
                self._cache['session_context'][key] = {
                    'value': value,
                    'timestamp': timestamp,
                    **kwargs
                }
                
            elif data_type == 'interaction':
                interaction_type = kwargs.get('type', 'general')
                if interaction_type not in self._cache['interaction_frequency']:
                    self._cache['interaction_frequency'][interaction_type] = []
                
                self._cache['interaction_frequency'][interaction_type].append({
                    'value': value,
                    'timestamp': timestamp,
                    **kwargs
                })
                
            else:
                # 直接設置到快取中
                self._cache[data_type] = value
            
            self._check_cleanup()
            return True
            
        except Exception as e:
            logger.error(f"添加上下文資料失敗: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        獲取上下文資料
        
        Args:
            key: 資料鍵
            default: 預設值
            
        Returns:
            Any: 資料值
        """
        try:
            return self._cache.get(key, default)
        except Exception as e:
            logger.error(f"獲取上下文資料失敗: {e}")
            return default
    
    def update(self, updates: Dict[str, Any]) -> bool:
        """
        批量更新上下文資料
        
        Args:
            updates: 更新字典
            
        Returns:
            bool: 是否成功更新
        """
        try:
            for key, value in updates.items():
                if key in self._cache:
                    self._cache[key] = value
                else:
                    logger.warning(f"未知的上下文鍵: {key}")
            
            return True
            
        except Exception as e:
            logger.error(f"更新上下文資料失敗: {e}")
            return False
    
    def cleanup(self, force: bool = False) -> bool:
        """
        清理過期的上下文資料
        
        Args:
            force: 是否強制清理
            
        Returns:
            bool: 是否成功清理
        """
        try:
            current_time = time.time()
            
            if not force and current_time - self.last_cleanup < self.cleanup_interval:
                return True
            
            # 清理過期的情感記錄（保留最近1小時）
            hour_ago = current_time - 3600
            self._cache['user_emotions'] = [
                (emotion, intensity, timestamp) 
                for emotion, intensity, timestamp in self._cache['user_emotions']
                if timestamp > hour_ago
            ]
            
            # 清理過期的主題記錄（保留最近2小時）
            two_hours_ago = current_time - 7200
            self._cache['conversation_themes'] = [
                (theme, confidence, timestamp)
                for theme, confidence, timestamp in self._cache['conversation_themes']
                if timestamp > two_hours_ago
            ]
            
            # 清理過期的偏好記錄（保留最近24小時）
            day_ago = current_time - 86400
            for pref_type in self._cache['user_preferences']:
                self._cache['user_preferences'][pref_type] = [
                    (value, weight, timestamp)
                    for value, weight, timestamp in self._cache['user_preferences'][pref_type]
                    if timestamp > day_ago
                ]
            
            # 清理過期的會話上下文（保留最近30分鐘）
            thirty_min_ago = current_time - 1800
            expired_sessions = [
                key for key, data in self._cache['session_context'].items()
                if data.get('timestamp', 0) < thirty_min_ago
            ]
            for key in expired_sessions:
                del self._cache['session_context'][key]
            
            # 清理互動頻率記錄（保留最近7天）
            week_ago = current_time - 604800
            for interaction_type in self._cache['interaction_frequency']:
                self._cache['interaction_frequency'][interaction_type] = [
                    record for record in self._cache['interaction_frequency'][interaction_type]
                    if record.get('timestamp', 0) > week_ago
                ]
            
            self.last_cleanup = current_time
            logger.debug("上下文快取清理完成")
            return True
            
        except Exception as e:
            logger.error(f"清理上下文快取失敗: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        獲取上下文快取統計資訊
        
        Returns:
            Dict: 統計資訊
        """
        try:
            return {
                'emotion_count': len(self._cache['user_emotions']),
                'theme_count': len(self._cache['conversation_themes']),
                'preference_types': len(self._cache['user_preferences']),
                'total_preferences': sum(
                    len(prefs) for prefs in self._cache['user_preferences'].values()
                ),
                'session_contexts': len(self._cache['session_context']),
                'interaction_types': len(self._cache['interaction_frequency']),
                'intimate_level': self._cache['intimate_level'],
                'current_mood': self._cache['current_mood'],
                'conversation_depth': self._cache['conversation_depth'],
                'last_interaction': self._cache.get('last_interaction_time'),
                'memory_usage': self._estimate_memory_usage()
            }
            
        except Exception as e:
            logger.error(f"獲取統計資訊失敗: {e}")
            return {}
    
    def get_recent_emotions(self, limit: int = 5) -> List[Tuple[str, float, float]]:
        """
        獲取最近的情感記錄
        
        Args:
            limit: 返回數量限制
            
        Returns:
            List: [(emotion, intensity, timestamp), ...]
        """
        try:
            return sorted(
                self._cache['user_emotions'], 
                key=lambda x: x[2], 
                reverse=True
            )[:limit]
            
        except Exception as e:
            logger.error(f"獲取最近情感記錄失敗: {e}")
            return []
    
    def get_recent_themes(self, limit: int = 3) -> List[Tuple[str, float, float]]:
        """
        獲取最近的對話主題
        
        Args:
            limit: 返回數量限制
            
        Returns:
            List: [(theme, confidence, timestamp), ...]
        """
        try:
            return sorted(
                self._cache['conversation_themes'],
                key=lambda x: x[2],
                reverse=True
            )[:limit]
            
        except Exception as e:
            logger.error(f"獲取最近主題記錄失敗: {e}")
            return []
    
    def get_user_preferences_summary(self) -> Dict[str, Any]:
        """
        獲取用戶偏好摘要
        
        Returns:
            Dict: 偏好摘要
        """
        try:
            summary = {}
            
            for pref_type, preferences in self._cache['user_preferences'].items():
                if not preferences:
                    continue
                
                # 計算加權平均和統計
                weighted_items = {}
                total_weight = 0
                
                for value, weight, timestamp in preferences:
                    if value not in weighted_items:
                        weighted_items[value] = {'weight': 0, 'count': 0, 'latest': 0}
                    
                    weighted_items[value]['weight'] += weight
                    weighted_items[value]['count'] += 1
                    weighted_items[value]['latest'] = max(
                        weighted_items[value]['latest'], timestamp
                    )
                    total_weight += weight
                
                # 排序並取前三名
                top_preferences = sorted(
                    weighted_items.items(),
                    key=lambda x: (x[1]['weight'], x[1]['latest']),
                    reverse=True
                )[:3]
                
                summary[pref_type] = {
                    'top_items': [(item, data['weight']) for item, data in top_preferences],
                    'total_items': len(weighted_items),
                    'total_weight': total_weight
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"獲取偏好摘要失敗: {e}")
            return {}
    
    def update_intimate_level(self, change: int) -> int:
        """
        更新親密度等級
        
        Args:
            change: 變化值（可為負數）
            
        Returns:
            int: 新的親密度等級
        """
        try:
            current_level = self._cache['intimate_level']
            new_level = max(0, min(5, current_level + change))
            self._cache['intimate_level'] = new_level
            
            logger.debug(f"親密度等級從 {current_level} 變更為 {new_level}")
            return new_level
            
        except Exception as e:
            logger.error(f"更新親密度等級失敗: {e}")
            return self._cache.get('intimate_level', 0)
    
    def update_conversation_depth(self, depth: int) -> bool:
        """
        更新對話深度
        
        Args:
            depth: 對話深度值
            
        Returns:
            bool: 是否成功更新
        """
        try:
            self._cache['conversation_depth'] = max(0, depth)
            return True
            
        except Exception as e:
            logger.error(f"更新對話深度失敗: {e}")
            return False
    
    def set_mood(self, mood: str) -> bool:
        """
        設置當前心情
        
        Args:
            mood: 心情值
            
        Returns:
            bool: 是否成功設置
        """
        try:
            self._cache['current_mood'] = mood
            self._cache['last_interaction_time'] = time.time()
            return True
            
        except Exception as e:
            logger.error(f"設置心情失敗: {e}")
            return False
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        獲取完整的上下文摘要
        
        Returns:
            Dict: 上下文摘要
        """
        try:
            return {
                'intimate_level': self._cache['intimate_level'],
                'current_mood': self._cache['current_mood'],
                'conversation_depth': self._cache['conversation_depth'],
                'user_expressed_affection': self._cache['user_expressed_affection'],
                'recent_emotions': self.get_recent_emotions(3),
                'recent_themes': self.get_recent_themes(3),
                'preferences_summary': self.get_user_preferences_summary(),
                'last_topic_change': self._cache.get('last_topic_change'),
                'last_interaction': self._cache.get('last_interaction_time'),
                'session_active': len(self._cache['session_context']) > 0
            }
            
        except Exception as e:
            logger.error(f"獲取上下文摘要失敗: {e}")
            return {}
    
    def reset_session(self) -> bool:
        """
        重置會話上下文（保留長期記憶）
        
        Returns:
            bool: 是否成功重置
        """
        try:
            # 清空會話相關的臨時資料
            self._cache['session_context'].clear()
            self._cache['conversation_depth'] = 0
            self._cache['current_mood'] = 'neutral'
            self._cache['last_topic_change'] = None
            
            logger.info("會話上下文已重置")
            return True
            
        except Exception as e:
            logger.error(f"重置會話上下文失敗: {e}")
            return False
    
    def _limit_list_size(self, key: str, max_size: int) -> None:
        """限制列表大小"""
        if len(self._cache[key]) > max_size:
            self._cache[key] = self._cache[key][-max_size:]
    
    def _limit_preference_size(self, preference_type: str) -> None:
        """限制偏好記錄大小"""
        if len(self._cache['user_preferences'][preference_type]) > self.max_preferences:
            # 保留最新的記錄
            self._cache['user_preferences'][preference_type] = sorted(
                self._cache['user_preferences'][preference_type],
                key=lambda x: x[2],
                reverse=True
            )[:self.max_preferences]
    
    def _check_cleanup(self) -> None:
        """檢查是否需要清理"""
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self.cleanup()
    
    def _estimate_memory_usage(self) -> int:
        """估算記憶體使用量（位元組）"""
        try:
            import sys
            total_size = 0
            
            for key, value in self._cache.items():
                total_size += sys.getsizeof(value)
                if isinstance(value, (list, dict)):
                    total_size += sum(sys.getsizeof(item) for item in (
                        value if isinstance(value, list) else value.values()
                    ))
            
            return total_size
            
        except Exception:
            return 0
