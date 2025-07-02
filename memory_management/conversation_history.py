#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
對話歷史管理模組
專門處理對話歷史的存儲、檢索和管理
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from .base_memory import BaseMemoryManager
import sys

logger = logging.getLogger(__name__)

class ConversationHistoryManager(BaseMemoryManager):
    """對話歷史管理器"""
    
    def __init__(self, chat_instance=None, max_history: int = 10, max_response_history: int = 12):
        super().__init__(chat_instance)
        
        # 配置參數
        self.max_history = max_history
        self.max_response_history = max_response_history
        self.max_topic_history = 5
        
        # 對話歷史存儲
        self.conversation_history = []  # [(user_input, bot_response, timestamp), ...]
        self.recent_responses = []  # [response1, response2, ...]
        self.topic_history = []  # [(topic, timestamp), ...]
        
        # 統計資料
        self.stats = {
            'total_conversations': 0,
            'total_user_words': 0,
            'total_bot_words': 0,
            'last_cleanup_time': None
        }
        
        logger.info(f"對話歷史管理器初始化完成 (最大歷史: {max_history})")
    
    def add(self, data: Tuple[str, str]) -> bool:
        """
        添加對話到歷史記錄
        
        Args:
            data: (user_input, bot_response) 元組
            
        Returns:
            bool: 是否成功添加
        """
        try:
            user_input, bot_response = data
            timestamp = self._get_current_timestamp()
            
            # 添加到對話歷史
            self.conversation_history.append((user_input, bot_response, timestamp))
            
            # 添加到回應歷史
            self.recent_responses.append(bot_response)
            
            # 更新統計
            self.stats['total_conversations'] += 1
            self.stats['total_user_words'] += len(user_input)
            self.stats['total_bot_words'] += len(bot_response)
            
            # 自動清理超出限制的記錄
            self._auto_cleanup()
            
            logger.debug(f"對話記錄已添加: 用戶[{len(user_input)}字] -> 機器人[{len(bot_response)}字]")
            return True
            
        except Exception as e:
            logger.error(f"添加對話記錄失敗: {e}")
            return False
    
    def get(self, query: Dict[str, Any]) -> List[Tuple[str, str, float]]:
        """
        獲取對話歷史
        
        Args:
            query: 查詢條件 {'limit': int, 'from_time': float, 'to_time': float}
            
        Returns:
            List: 對話歷史列表
        """
        try:
            limit = query.get('limit', len(self.conversation_history))
            from_time = query.get('from_time', 0)
            to_time = query.get('to_time', float('inf'))
            
            # 篩選時間範圍內的對話
            filtered_history = [
                (user_input, bot_response, timestamp)
                for user_input, bot_response, timestamp in self.conversation_history
                if from_time <= timestamp <= to_time
            ]
            
            # 返回最近的記錄
            return filtered_history[-limit:] if limit > 0 else filtered_history
            
        except Exception as e:
            logger.error(f"獲取對話歷史失敗: {e}")
            return []
    
    def get_recent_conversations(self, count: int = 5) -> List[Tuple[str, str]]:
        """獲取最近的對話（不包含時間戳）"""
        recent = self.conversation_history[-count:] if count > 0 else self.conversation_history
        return [(user_input, bot_response) for user_input, bot_response, _ in recent]
    
    def get_recent_responses(self, count: int = 5) -> List[str]:
        """獲取最近的機器人回應"""
        return self.recent_responses[-count:] if count > 0 else self.recent_responses
    
    def add_topic(self, topic: str) -> bool:
        """添加話題到話題歷史"""
        try:
            timestamp = self._get_current_timestamp()
            self.topic_history.append((topic, timestamp))
            
            # 限制話題歷史長度
            if len(self.topic_history) > self.max_topic_history:
                self.topic_history = self.topic_history[-self.max_topic_history:]
            
            return True
        except Exception as e:
            logger.error(f"添加話題失敗: {e}")
            return False
    
    def get_topic_history(self) -> List[Tuple[str, float]]:
        """獲取話題歷史"""
        return self.topic_history.copy()
    
    def update(self, query: Dict[str, Any], data: Any) -> bool:
        """
        更新對話記錄（通常不需要）
        
        Args:
            query: 查詢條件
            data: 新的資料
            
        Returns:
            bool: 是否成功更新
        """
        # 對話歷史通常不需要更新，只需要添加新記錄
        logger.warning("對話歷史通常不支援更新操作")
        return False
    
    def cleanup(self, force: bool = False) -> int:
        """
        清理過期的對話記錄
        
        Args:
            force: 是否強制清理
            
        Returns:
            int: 清理的項目數量
        """
        try:
            cleaned_count = 0
            
            # 限制對話歷史長度
            if len(self.conversation_history) > self.max_history:
                old_count = len(self.conversation_history)
                self.conversation_history = self.conversation_history[-self.max_history:]
                cleaned_count += old_count - len(self.conversation_history)
            
            # 限制回應歷史長度
            if len(self.recent_responses) > self.max_response_history:
                old_count = len(self.recent_responses)
                self.recent_responses = self.recent_responses[-self.max_response_history:]
                cleaned_count += old_count - len(self.recent_responses)
            
            # 限制話題歷史長度
            if len(self.topic_history) > self.max_topic_history:
                old_count = len(self.topic_history)
                self.topic_history = self.topic_history[-self.max_topic_history:]
                cleaned_count += old_count - len(self.topic_history)
            
            if cleaned_count > 0:
                logger.debug(f"對話歷史清理完成，清理了 {cleaned_count} 項記錄")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"清理對話歷史失敗: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        獲取對話歷史統計資訊
        
        Returns:
            Dict: 統計資訊
        """
        try:
            import sys
            return {
                'conversation_count': len(self.conversation_history),
                'response_count': len(self.recent_responses),
                'topic_count': len(self.topic_history),
                'total_conversations': self.stats['total_conversations'],
                'total_user_words': self.stats['total_user_words'],
                'total_bot_words': self.stats['total_bot_words'],
                'last_cleanup_time': self.stats.get('last_cleanup_time'),
                'memory_usage': (
                    sys.getsizeof(self.conversation_history) + 
                    sys.getsizeof(self.recent_responses) + 
                    sys.getsizeof(self.topic_history)
                )
            }
        except Exception as e:
            logger.error(f"獲取對話統計失敗: {e}")
            return {}
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """獲取記憶統計資訊"""
        return {
            'module_name': self.module_name,
            'total_conversations': len(self.conversation_history),
            'total_responses': len(self.recent_responses),
            'total_topics': len(self.topic_history),
            'max_history': self.max_history,
            'max_response_history': self.max_response_history,
            'max_topic_history': self.max_topic_history,
            'stats': self.stats.copy()
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """獲取對話摘要統計"""
        if not self.conversation_history:
            return {'total': 0, 'avg_user_length': 0, 'avg_bot_length': 0}
        
        total_conversations = len(self.conversation_history)
        total_user_chars = sum(len(user_input) for user_input, _, _ in self.conversation_history)
        total_bot_chars = sum(len(bot_response) for _, bot_response, _ in self.conversation_history)
        
        return {
            'total': total_conversations,
            'avg_user_length': total_user_chars / total_conversations,
            'avg_bot_length': total_bot_chars / total_conversations,
            'total_user_chars': total_user_chars,
            'total_bot_chars': total_bot_chars
        }
    
    def find_similar_conversations(self, query_text: str, limit: int = 3) -> List[Tuple[str, str, float]]:
        """尋找相似的對話記錄"""
        try:
            query_words = set(query_text.lower().split())
            similar_conversations = []
            
            for user_input, bot_response, timestamp in self.conversation_history:
                input_words = set(user_input.lower().split())
                # 計算簡單的相似度（共同詞彙比例）
                if input_words and query_words:
                    similarity = len(query_words & input_words) / len(query_words | input_words)
                    if similarity > 0.3:  # 相似度閾值
                        similar_conversations.append((user_input, bot_response, similarity))
            
            # 按相似度排序並返回前N個
            similar_conversations.sort(key=lambda x: x[2], reverse=True)
            return similar_conversations[:limit]
            
        except Exception as e:
            logger.error(f"尋找相似對話失敗: {e}")
            return []
    
    def get_recent_topics(self, count: int = 5) -> List[Dict[str, Any]]:
        """獲取最近的話題記錄"""
        try:
            recent_topics = self.topic_history[-count:] if count > 0 else self.topic_history
            return [{'topic': topic, 'timestamp': timestamp} for topic, timestamp in recent_topics]
        except Exception as e:
            logger.error(f"獲取最近話題失敗: {e}")
            return []
    
    def get_last_interaction_time(self) -> Optional[float]:
        """獲取最後一次互動時間"""
        if self.conversation_history:
            return self.conversation_history[-1][2]  # 返回最後一次對話的時間戳
        return None
    
    def _auto_cleanup(self):
        """自動清理超出限制的記錄"""
        # 限制對話歷史長度
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        # 限制回應歷史長度
        if len(self.recent_responses) > self.max_response_history:
            self.recent_responses = self.recent_responses[-self.max_response_history:]
    
    def clear_history(self) -> bool:
        """
        清空所有對話歷史
        
        Returns:
            bool: 是否成功清空
        """
        try:
            self.conversation_history.clear()
            self.recent_responses.clear()
            self.topic_history.clear()
            
            # 重置統計
            self.stats['total_conversations'] = 0
            self.stats['total_user_words'] = 0
            self.stats['total_bot_words'] = 0
            
            logger.info("對話歷史已清空")
            return True
            
        except Exception as e:
            logger.error(f"清空對話歷史失敗: {e}")
            return False

    def _get_current_timestamp(self) -> float:
        """獲取當前時間戳"""
        import time
        return time.time()
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """獲取所有對話記錄（用於導出）"""
        return [
            {
                'user_input': user_input,
                'bot_response': bot_response, 
                'timestamp': timestamp
            }
            for user_input, bot_response, timestamp in self.conversation_history
        ]
