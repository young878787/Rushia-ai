#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
記憶管理模組主入口
提供統一的記憶管理介面，整合對話歷史、用戶資料、上下文快取等功能
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from .conversation_history import ConversationHistoryManager
from .user_profile import UserProfileManager
from .context_cache import ContextCacheManager

logger = logging.getLogger(__name__)


class MemoryManager:
    """統一的記憶管理器 - 整合所有記憶管理功能"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化記憶管理器
        
        Args:
            config: 配置字典，包含各子管理器的配置
        """
        self.config = config or {}
        
        # 初始化子管理器
        self.conversation = ConversationHistoryManager(
            **self.config.get('conversation', {})
        )
        
        self.user_profile = UserProfileManager(
            **self.config.get('user_profile', {})
        )
        
        self.context_cache = ContextCacheManager(
            **self.config.get('context_cache', {})
        )
        
        logger.info("MemoryManager 統一記憶管理器初始化完成")
    
    def add_conversation(self, user_input: str, response: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加對話記錄
        
        Args:
            user_input: 用戶輸入
            response: 機器人回應
            metadata: 額外元資料
            
        Returns:
            bool: 是否成功添加
        """
        try:
            # 添加到對話歷史
            success = self.conversation.add((user_input, response))
            
            if success:
                # 更新用戶資料統計 - 使用正確的方法
                self.user_profile.add({
                    'type': 'conversation',
                    'user_input': user_input,
                    'response': response,
                    'metadata': metadata or {}
                })
                
                # 更新上下文
                self.context_cache.update({
                    'last_interaction_time': self.conversation.get_last_interaction_time()
                })
            
            return success
            
        except Exception as e:
            logger.error(f"添加對話記錄失敗: {e}")
            return False
    
    def get_conversation_history(self, limit: int = 10) -> List[Tuple[str, str]]:
        """
        獲取對話歷史（向後兼容格式）
        
        Args:
            limit: 返回數量限制
            
        Returns:
            List: [(user_input, response), ...]
        """
        try:
            conversations = self.conversation.get_recent_conversations(limit)
            # conversations 是 [(user_input, bot_response), ...] 格式
            return conversations
            
        except Exception as e:
            logger.error(f"獲取對話歷史失敗: {e}")
            return []
    
    def get_user_profile_dict(self) -> Dict[str, Any]:
        """
        獲取用戶資料字典（向後兼容格式）
        
        Returns:
            Dict: 用戶資料字典
        """
        try:
            profile_data = self.user_profile.get_profile_summary()
            
            # 轉換為舊格式
            basic_info = profile_data.get('basic_info', {})
            return {
                'interests': profile_data.get('interests', {}).get('top_items', []),
                'mood_history': [
                    (mood.get('mood', 'neutral'), mood.get('timestamp', 0)) 
                    for mood in profile_data.get('moods', [])
                ],
                'conversation_count': basic_info.get('conversation_count', 0),
                'favorite_topics': profile_data.get('favorite_topics', {}),
                'name': basic_info.get('name'),
                'personality_traits': profile_data.get('personality_traits', []),
                'special_memories': profile_data.get('special_memories', []),
                'interaction_patterns': profile_data.get('interaction_patterns', {}),
                'last_interaction': basic_info.get('last_seen'),
                'total_interactions': basic_info.get('conversation_count', 0)
            }
            
        except Exception as e:
            logger.error(f"獲取用戶資料字典失敗: {e}")
            return {}
    
    def get_context_cache_dict(self) -> Dict[str, Any]:
        """
        獲取上下文快取字典（向後兼容格式）
        
        Returns:
            Dict: 上下文快取字典
        """
        try:
            cache_summary = self.context_cache.get_context_summary()
            
            # 轉換為舊格式
            return {
                'user_emotions': [
                    emotion[0] for emotion in cache_summary.get('recent_emotions', [])
                ],
                'conversation_themes': [
                    theme[0] for theme in cache_summary.get('recent_themes', [])
                ],
                'user_expressed_affection': cache_summary.get('user_expressed_affection', False),
                'last_topic_change': cache_summary.get('last_topic_change'),
                'intimate_level': cache_summary.get('intimate_level', 0),
                'current_mood': cache_summary.get('current_mood', 'neutral'),
                'user_preferences': cache_summary.get('preferences_summary', {}),
                'conversation_depth': cache_summary.get('conversation_depth', 0),
                'last_interaction_time': cache_summary.get('last_interaction')
            }
            
        except Exception as e:
            logger.error(f"獲取上下文快取字典失敗: {e}")
            return {}
    
    def update_user_mood(self, mood: str, intensity: float = 0.5) -> bool:
        """
        更新用戶心情
        
        Args:
            mood: 心情
            intensity: 強度
            
        Returns:
            bool: 是否成功更新
        """
        try:
            # 更新用戶資料中的心情歷史
            mood_success = self.user_profile.add({
                'type': 'mood',
                'mood': mood,
                'intensity': intensity
            })
            
            # 更新上下文快取中的當前心情
            cache_success = self.context_cache.set_mood(mood)
            
            return mood_success and cache_success
            
        except Exception as e:
            logger.error(f"更新用戶心情失敗: {e}")
            return False
    
    def update_user_interest(self, interest: str, weight: float = 1.0) -> bool:
        """
        更新用戶興趣
        
        Args:
            interest: 興趣項目
            weight: 權重
            
        Returns:
            bool: 是否成功更新
        """
        try:
            return self.user_profile.add({
                'type': 'interest',
                'interest': interest,
                'weight': weight
            })
            
        except Exception as e:
            logger.error(f"更新用戶興趣失敗: {e}")
            return False
    
    def add_special_memory(self, memory: str, category: str = 'general') -> bool:
        """
        添加特殊記憶
        
        Args:
            memory: 記憶內容
            category: 記憶類別
            
        Returns:
            bool: 是否成功添加
        """
        try:
            return self.user_profile.add({
                'type': 'special_memory',
                'memory': memory,
                'category': category
            })
            
        except Exception as e:
            logger.error(f"添加特殊記憶失敗: {e}")
            return False
    
    def update_context_emotion(self, emotion: str, intensity: float = 0.5) -> bool:
        """
        更新上下文情感
        
        Args:
            emotion: 情感類型
            intensity: 情感強度
            
        Returns:
            bool: 是否成功更新
        """
        try:
            return self.context_cache.add('emotion', emotion, intensity=intensity)
            
        except Exception as e:
            logger.error(f"更新上下文情感失敗: {e}")
            return False
    
    def update_conversation_theme(self, theme: str, confidence: float = 0.5) -> bool:
        """
        更新對話主題
        
        Args:
            theme: 主題
            confidence: 置信度
            
        Returns:
            bool: 是否成功更新
        """
        try:
            return self.context_cache.add('theme', theme, confidence=confidence)
            
        except Exception as e:
            logger.error(f"更新對話主題失敗: {e}")
            return False
    
    def cleanup_all(self, force: bool = False) -> bool:
        """
        清理所有記憶模組
        
        Args:
            force: 是否強制清理
            
        Returns:
            bool: 是否成功清理
        """
        try:
            results = [
                self.conversation.cleanup(),
                self.user_profile.cleanup(),
                self.context_cache.cleanup()
            ]
            
            # 將結果轉換為布林值（非零即成功）
            success = all(result is not None for result in results)
            logger.info(f"記憶管理器清理完成，成功: {success}")
            return success
            
        except Exception as e:
            logger.error(f"清理記憶管理器失敗: {e}")
            return False
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        獲取綜合統計資訊
        
        Returns:
            Dict: 統計資訊
        """
        try:
            return {
                'conversation_stats': self.conversation.get_stats(),
                'user_profile_stats': self.user_profile.get_stats(),
                'context_cache_stats': self.context_cache.get_stats(),
                'total_memory_usage': (
                    self.conversation.get_stats().get('memory_usage', 0) +
                    self.user_profile.get_stats().get('memory_usage', 0) +
                    self.context_cache.get_stats().get('memory_usage', 0)
                )
            }
            
        except Exception as e:
            logger.error(f"獲取綜合統計資訊失敗: {e}")
            return {}
    
    def reset_session(self) -> bool:
        """
        重置會話（保留長期記憶）
        
        Returns:
            bool: 是否成功重置
        """
        try:
            # 只重置上下文快取的會話部分
            success = self.context_cache.reset_session()
            
            if success:
                logger.info("會話已重置，長期記憶保留")
            
            return success
            
        except Exception as e:
            logger.error(f"重置會話失敗: {e}")
            return False
    
    def export_memory_data(self) -> Dict[str, Any]:
        """
        導出記憶資料（用於備份或遷移）
        
        Returns:
            Dict: 完整的記憶資料
        """
        try:
            return {
                'conversation_history': self.conversation.get_all_conversations(),
                'user_profile': self.user_profile.get_profile_summary(),
                'context_cache': self.context_cache.get_context_summary(),
                'export_timestamp': self.conversation.get_last_interaction_time(),
                'stats': self.get_comprehensive_stats()
            }
            
        except Exception as e:
            logger.error(f"導出記憶資料失敗: {e}")
            return {}
    
    def import_memory_data(self, data: Dict[str, Any]) -> bool:
        """
        導入記憶資料（用於恢復或遷移）
        
        Args:
            data: 記憶資料
            
        Returns:
            bool: 是否成功導入
        """
        try:
            # 這裡可以實現資料導入邏輯
            # 由於涉及複雜的資料結構，暫時記錄日誌
            logger.info("記憶資料導入功能待實現")
            return True
            
        except Exception as e:
            logger.error(f"導入記憶資料失敗: {e}")
            return False
    
    def clear_conversation_history(self) -> bool:
        """
        清空對話歷史
        
        Returns:
            bool: 是否成功清空
        """
        try:
            return self.conversation.clear_history()
        except Exception as e:
            logger.error(f"清空對話歷史失敗: {e}")
            return False


# 便捷函數
def create_memory_manager(config: Optional[Dict[str, Any]] = None) -> MemoryManager:
    """
    創建記憶管理器實例
    
    Args:
        config: 配置字典
        
    Returns:
        MemoryManager: 記憶管理器實例
    """
    return MemoryManager(config)


# 導出主要類和函數
__all__ = [
    'MemoryManager',
    'ConversationHistoryManager',
    'UserProfileManager', 
    'ContextCacheManager',
    'create_memory_manager'
]
