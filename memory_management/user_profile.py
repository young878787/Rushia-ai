#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用戶資料管理模組
專門處理用戶個人資料的存儲、更新和分析
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter
from .base_memory import BaseMemoryManager
import time

logger = logging.getLogger(__name__)

class UserProfileManager(BaseMemoryManager):
    """用戶資料管理器"""
    
    def __init__(self, chat_instance=None):
        super().__init__(chat_instance)
        
        # 用戶基本資料
        self.profile = {
            'name': None,  # 用戶名稱
            'conversation_count': 0,  # 對話次數
            'first_seen': None,  # 首次見面時間
            'last_seen': None,  # 最後見面時間
            'communication_style': 'unknown',  # 溝通風格
            'preferred_response_length': 'medium',  # 偏好回應長度
            'personality_traits': []  # 個性特徵
        }
        
        # 用戶偏好和興趣
        self.interests = {}  # 興趣愛好字典
        self.favorite_topics = defaultdict(int)  # 喜歡的話題統計
        self.special_memories = []  # 特殊記憶列表
        
        # 情緒和心理狀態追蹤
        self.moods = []  # [(mood, timestamp), ...]
        self.max_mood_history = 50  # 最大心情記錄數
        
        # 互動模式分析
        self.interaction_patterns = {
            'active_hours': defaultdict(int),  # 活躍時段統計
            'message_lengths': [],  # 訊息長度記錄
            'response_preferences': defaultdict(int),  # 回應偏好
            'emotional_triggers': defaultdict(list)  # 情感觸發詞
        }
        
        logger.info("用戶資料管理器初始化完成")
    
    def add(self, data: Dict[str, Any]) -> bool:
        """
        添加或更新用戶資料
        
        Args:
            data: 包含用戶資料的字典
            
        Returns:
            bool: 是否成功添加
        """
        try:
            data_type = data.get('type', 'general')
            
            if data_type == 'conversation':
                return self._add_conversation_data(data)
            elif data_type == 'mood':
                return self._add_mood_data(data)
            elif data_type == 'interest':
                return self._add_interest_data(data)
            elif data_type == 'favorite_topic':
                return self._add_topic_data(data)
            elif data_type == 'special_memory':
                return self._add_memory_data(data)
            elif data_type == 'profile':
                return self._update_profile_data(data)
            else:
                logger.warning(f"未知的資料類型: {data_type}")
                return False
                
        except Exception as e:
            logger.error(f"添加用戶資料失敗: {e}")
            return False
    
    def _add_conversation_data(self, data: Dict[str, Any]) -> bool:
        """添加對話相關數據"""
        try:
            # 更新對話計數
            self.profile['conversation_count'] = self.profile.get('conversation_count', 0) + 1
            self.profile['last_seen'] = time.time()
            return True
        except Exception as e:
            logger.error(f"添加對話數據失敗: {e}")
            return False
    
    def _add_mood_data(self, data: Dict[str, Any]) -> bool:
        """添加心情數據"""
        try:
            mood = data.get('mood', 'neutral')
            intensity = data.get('intensity', 0.5)
            timestamp = time.time()
            
            mood_entry = {
                'mood': mood,
                'intensity': intensity,
                'timestamp': timestamp
            }
            
            self.moods.append(mood_entry)
            
            # 限制心情歷史長度
            if len(self.moods) > 50:
                self.moods = self.moods[-30:]
            
            return True
        except Exception as e:
            logger.error(f"添加心情數據失敗: {e}")
            return False
    
    def _add_interest_data(self, data: Dict[str, Any]) -> bool:
        """添加興趣數據"""
        try:
            interest = data.get('interest')
            weight = data.get('weight', 1.0)
            
            if not interest:
                return False
            
            if interest in self.interests:
                self.interests[interest]['weight'] += weight
                self.interests[interest]['count'] += 1
                self.interests[interest]['last_updated'] = time.time()
            else:
                self.interests[interest] = {
                    'weight': weight,
                    'count': 1,
                    'first_seen': time.time(),
                    'last_updated': time.time()
                }
            
            return True
        except Exception as e:
            logger.error(f"添加興趣數據失敗: {e}")
            return False
    
    def _add_topic_data(self, data: Dict[str, Any]) -> bool:
        """添加話題數據"""
        try:
            topic = data.get('topic')
            weight = data.get('weight', 1.0)
            
            if not topic:
                return False
            
            if topic in self.favorite_topics:
                self.favorite_topics[topic] += weight
            else:
                self.favorite_topics[topic] = weight
            
            return True
        except Exception as e:
            logger.error(f"添加話題數據失敗: {e}")
            return False
    
    def _add_memory_data(self, data: Dict[str, Any]) -> bool:
        """添加特殊記憶數據"""
        try:
            memory = data.get('memory')
            category = data.get('category', 'general')
            
            if not memory:
                return False
            
            memory_entry = {
                'memory': memory,
                'category': category,
                'timestamp': time.time()
            }
            
            self.special_memories.append(memory_entry)
            
            # 限制特殊記憶數量
            if len(self.special_memories) > 20:
                self.special_memories = self.special_memories[-15:]
            
            return True
        except Exception as e:
            logger.error(f"添加特殊記憶失敗: {e}")
            return False
    
    def _update_profile_data(self, data: Dict[str, Any]) -> bool:
        """更新基本資料"""
        try:
            # 只更新允許的欄位
            allowed_fields = ['name', 'communication_style', 'preferred_response_length', 'last_seen']
            
            for field in allowed_fields:
                if field in data:
                    if field == 'last_seen':
                        # 轉換 datetime 為 timestamp
                        if hasattr(data[field], 'timestamp'):
                            self.profile[field] = data[field].timestamp()
                        else:
                            self.profile[field] = time.time()
                    else:
                        self.profile[field] = data[field]
            
            return True
        except Exception as e:
            logger.error(f"更新基本資料失敗: {e}")
            return False
    
    def _analyze_input_pattern(self, user_input: str, timestamp: float):
        """分析用戶輸入模式"""
        # 記錄訊息長度
        self.interaction_patterns['message_lengths'].append(len(user_input))
        if len(self.interaction_patterns['message_lengths']) > 100:
            self.interaction_patterns['message_lengths'] = self.interaction_patterns['message_lengths'][-100:]
        
        # 記錄活躍時段
        hour = datetime.fromtimestamp(timestamp).hour
        self.interaction_patterns['active_hours'][hour] += 1
        
        # 分析情感詞彙（簡單版本）
        positive_words = ['開心', '高興', '快樂', '幸福', '好', '棒', '讚', '愛', '喜歡']
        negative_words = ['難過', '傷心', '沮喪', '不好', '累', '壓力', '煩惱', '生氣']
        
        for word in positive_words:
            if word in user_input:
                self.interaction_patterns['emotional_triggers']['positive'].append(word)
        
        for word in negative_words:
            if word in user_input:
                self.interaction_patterns['emotional_triggers']['negative'].append(word)
    
    def _update_topic_stats(self, user_input: str):
        """更新話題統計"""
        # 簡單的話題識別（基於關鍵詞）
        words = user_input.split()
        for word in words:
            if len(word) > 1:  # 忽略單字
                self.favorite_topics[word] += 1
    
    def get(self, query: Dict[str, Any]) -> Any:
        """
        獲取用戶資料
        
        Args:
            query: 查詢條件
            
        Returns:
            Any: 查詢結果
        """
        try:
            query_type = query.get('type', 'profile')
            
            if query_type == 'profile':
                return self.profile.copy()
            elif query_type == 'interests':
                return self.interests.copy()
            elif query_type == 'mood_history':
                limit = query.get('limit', len(self.moods))
                return self.moods[-limit:] if limit > 0 else self.moods
            elif query_type == 'favorite_topics':
                limit = query.get('limit', 10)
                return dict(Counter(self.favorite_topics).most_common(limit))
            elif query_type == 'interaction_patterns':
                return self.interaction_patterns.copy()
            elif query_type == 'summary':
                return self.get_user_summary()
            else:
                logger.warning(f"未知的查詢類型: {query_type}")
                return None
                
        except Exception as e:
            logger.error(f"獲取用戶資料失敗: {e}")
            return None
    
    def update(self, query: Dict[str, Any], data: Any) -> bool:
        """
        更新用戶資料
        
        Args:
            query: 查詢條件
            data: 新的資料
            
        Returns:
            bool: 是否成功更新
        """
        try:
            update_type = query.get('type', 'profile')
            
            if update_type == 'profile':
                return self._update_profile_data(data)
            elif update_type == 'name':
                self.profile['name'] = data
                return True
            elif update_type == 'communication_style':
                self.profile['communication_style'] = data
                return True
            elif update_type == 'add_memory':
                self.special_memories.append({
                    'content': data,
                    'timestamp': self._get_current_timestamp()
                })
                return True
            else:
                logger.warning(f"未知的更新類型: {update_type}")
                return False
                
        except Exception as e:
            logger.error(f"更新用戶資料失敗: {e}")
            return False
    
    def cleanup(self, force: bool = False) -> int:
        """
        清理過期的用戶資料
        
        Args:
            force: 是否強制清理
            
        Returns:
            int: 清理的項目數量
        """
        try:
            cleaned_count = 0
            current_time = time.time()
            
            # 清理過舊的心情記錄（保留最近50條）
            if len(self.moods) > self.max_mood_history:
                old_count = len(self.moods)
                self.moods = self.moods[-self.max_mood_history:]
                cleaned_count += old_count - len(self.moods)
            
            # 清理過舊的特殊記憶（保留最近20條）
            if len(self.special_memories) > 20:
                old_count = len(self.special_memories)
                self.special_memories = self.special_memories[-20:]
                cleaned_count += old_count - len(self.special_memories)
            
            # 清理低權重的興趣（保留權重>0.1的）
            if force:
                low_interest_keys = [k for k, v in self.interests.items() if v.get('weight', 0) < 0.1]
                for key in low_interest_keys:
                    del self.interests[key]
                    cleaned_count += 1
            
            if cleaned_count > 0:
                logger.debug(f"用戶資料清理完成，清理了 {cleaned_count} 項記錄")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"清理用戶資料失敗: {e}")
            return 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """獲取記憶統計資訊"""
        return {
            'module_name': self.module_name,
            'conversation_count': self.profile['conversation_count'],
            'mood_records': len(self.moods),
            'interests_count': len(self.interests),
            'favorite_topics_count': len(self.favorite_topics),
            'special_memories_count': len(self.special_memories),
            'first_seen': self.profile['first_seen'],
            'last_seen': self.profile['last_seen']
        }
    
    def get_user_summary(self) -> Dict[str, Any]:
        """獲取用戶摘要資訊"""
        # 計算平均心情
        recent_moods = self.moods[-10:] if self.moods else []
        avg_mood_score = 0
        if recent_moods:
            mood_scores = {'positive': 1, 'neutral': 0, 'negative': -1}
            total_score = sum(mood_scores.get(mood, 0) for mood, _, _ in recent_moods)
            avg_mood_score = total_score / len(recent_moods)
        
        # 找出最活躍時段
        most_active_hour = max(self.interaction_patterns['active_hours'].items(), 
                             key=lambda x: x[1])[0] if self.interaction_patterns['active_hours'] else None
        
        # 計算平均訊息長度
        avg_message_length = (sum(self.interaction_patterns['message_lengths']) / 
                            len(self.interaction_patterns['message_lengths'])) if self.interaction_patterns['message_lengths'] else 0
        
        # 獲取最愛話題
        top_topics = dict(Counter(self.favorite_topics).most_common(5))
        
        return {
            'name': self.profile.get('name'),
            'conversation_count': self.profile['conversation_count'],
            'avg_mood_score': avg_mood_score,
            'most_active_hour': most_active_hour,
            'avg_message_length': avg_message_length,
            'top_topics': top_topics,
            'communication_style': self.profile['communication_style'],
            'total_interests': len(self.interests),
            'relationship_duration_days': self._get_relationship_duration_days()
        }
    
    def _get_relationship_duration_days(self) -> int:
        """計算關係持續天數"""
        if self.profile['first_seen'] and self.profile['last_seen']:
            duration_seconds = self.profile['last_seen'] - self.profile['first_seen']
            return int(duration_seconds / (24 * 3600))
        return 0
    
    def analyze_communication_style(self) -> str:
        """分析用戶溝通風格"""
        if not self.interaction_patterns['message_lengths']:
            return 'unknown'
        
        avg_length = sum(self.interaction_patterns['message_lengths']) / len(self.interaction_patterns['message_lengths'])
        
        if avg_length > 50:
            return 'detailed'  # 詳細型
        elif avg_length > 20:
            return 'moderate'  # 適中型
        else:
            return 'concise'   # 簡潔型
    
    def get_personality_insights(self) -> Dict[str, Any]:
        """獲取個性洞察"""
        insights = {
            'dominant_emotions': [],
            'communication_patterns': {},
            'interaction_preferences': {},
            'topic_interests': {}
        }
        
        try:
            # 分析主要情緒
            if self.moods:
                mood_counter = Counter(mood for mood, _, _ in self.moods[-20:])
                insights['dominant_emotions'] = mood_counter.most_common(3)
            
            # 分析溝通模式
            insights['communication_patterns'] = {
                'style': self.analyze_communication_style(),
                'avg_message_length': sum(self.interaction_patterns['message_lengths']) / len(self.interaction_patterns['message_lengths']) if self.interaction_patterns['message_lengths'] else 0,
                'most_active_hour': max(self.interaction_patterns['active_hours'].items(), key=lambda x: x[1])[0] if self.interaction_patterns['active_hours'] else None
            }
            
            # 分析話題興趣
            insights['topic_interests'] = dict(Counter(self.favorite_topics).most_common(5))
            
        except Exception as e:
            logger.error(f"分析個性洞察失敗: {e}")
        
        return insights
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """
        獲取用戶資料摘要
        
        Returns:
            Dict: 用戶資料摘要
        """
        try:
            # 計算興趣統計
            interest_items = [(interest, data['weight']) for interest, data in self.interests.items()]
            top_interests = sorted(interest_items, key=lambda x: x[1], reverse=True)[:5]
            
            # 計算心情統計
            recent_moods = self.moods[-10:] if self.moods else []
            
            # 獲取最近話題
            recent_topics = list(self.favorite_topics.items())
            recent_topics.sort(key=lambda x: x[1], reverse=True)
            top_topics = recent_topics[:5]
            
            return {
                'basic_info': {
                    'name': self.profile.get('name'),
                    'conversation_count': self.profile.get('conversation_count', 0),
                    'communication_style': self.profile.get('communication_style', 'unknown'),
                    'first_seen': self.profile.get('first_seen'),
                    'last_seen': self.profile.get('last_seen')
                },
                'interests': {
                    'top_items': [item[0] for item in top_interests],
                    'total_count': len(self.interests)
                },
                'moods': recent_moods,
                'favorite_topics': dict(top_topics),
                'personality_traits': self.profile.get('personality_traits', []),
                'special_memories': self.special_memories,
                'interaction_patterns': self.interaction_patterns,
                'last_interaction': self.profile.get('last_seen'),
                'total_interactions': self.profile.get('conversation_count', 0)
            }
            
        except Exception as e:
            logger.error(f"獲取用戶資料摘要失敗: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        獲取用戶資料統計資訊
        
        Returns:
            Dict: 統計資訊
        """
        try:
            import sys
            return {
                'conversation_count': self.profile.get('conversation_count', 0),
                'interests_count': len(self.interests),
                'favorite_topics_count': len(self.favorite_topics),
                'special_memories_count': len(self.special_memories),
                'moods_count': len(self.moods),
                'communication_style': self.profile.get('communication_style', 'unknown'),
                'first_seen': self.profile.get('first_seen'),
                'last_seen': self.profile.get('last_seen'),
                'memory_usage': (
                    sys.getsizeof(self.profile) +
                    sys.getsizeof(self.interests) +
                    sys.getsizeof(self.favorite_topics) +
                    sys.getsizeof(self.special_memories) +
                    sys.getsizeof(self.moods)
                )
            }
        except Exception as e:
            logger.error(f"獲取用戶資料統計失敗: {e}")
            return {}
