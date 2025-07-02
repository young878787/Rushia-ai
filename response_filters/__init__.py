#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回應過濾器管理模組
統一管理所有回應清理和優化功能
"""

from .base_filter import BaseResponseFilter
from .content_cleaner import ContentCleanerFilter
from .analysis_language_filter import AnalysisLanguageFilter
from .character_confusion_filter import CharacterConfusionFilter
from .dialogue_format_filter import DialogueFormatFilter
from .self_reference_filter import SelfReferenceFilter
from .quality_validator import QualityValidatorFilter
from .sweetness_enhancer import SweetenessEnhancerFilter

__all__ = [
    'FilterManager',
    'BaseResponseFilter',
    'ContentCleanerFilter',
    'AnalysisLanguageFilter',
    'CharacterConfusionFilter',
    'DialogueFormatFilter',
    'SelfReferenceFilter',
    'QualityValidatorFilter',
    'SweetenessEnhancerFilter'
]

import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class FilterManager:
    """回應過濾器管理器 - 統一協調所有過濾器"""
    
    def __init__(self, chat_instance=None):
        """
        初始化過濾器管理器
        
        Args:
            chat_instance: RushiaLoRAChat 實例
        """
        self.chat_instance = chat_instance
        
        # 初始化所有過濾器（按照處理順序）
        self.filters = [
            ContentCleanerFilter(chat_instance),           # 1. 基礎內容清理
            AnalysisLanguageFilter(chat_instance),         # 2. 分析性語言移除
            CharacterConfusionFilter(chat_instance),       # 3. 角色混亂處理
            DialogueFormatFilter(chat_instance),           # 4. 對話格式清理
            SelfReferenceFilter(chat_instance),            # 5. 自稱優化
            SweetenessEnhancerFilter(chat_instance),       # 6. 溫柔甜蜜增強
            QualityValidatorFilter(chat_instance)          # 7. 最終品質驗證
        ]
        
        # 統計資料
        self.stats = {
            'total_processed': 0,
            'total_modified': 0,
            'filter_stats': {filter.__class__.__name__: 0 for filter in self.filters},
            'rejection_reasons': {}
        }
        
        logger.info(f"過濾器管理器初始化完成，載入 {len(self.filters)} 個過濾器")
    
    def process_response(self, response: str, user_input: str = "", context: Dict = None) -> Tuple[str, Dict]:
        """
        完整處理回應，通過所有過濾器
        
        Args:
            response: 原始回應
            user_input: 用戶輸入
            context: 對話上下文
            
        Returns:
            Tuple[str, Dict]: (處理後的回應, 處理統計)
        """
        if not response:
            return response, {'processed': False, 'reason': 'empty_input'}
        
        self.stats['total_processed'] += 1
        original_response = response
        processing_log = []
        
        # 依序通過所有過濾器
        for filter_instance in self.filters:
            try:
                # 記錄處理前狀態
                before_length = len(response)
                
                # 應用過濾器
                filtered_response = filter_instance._safe_filter(response, user_input, context)
                
                # 檢查是否有變化
                if filtered_response != response:
                    after_length = len(filtered_response)
                    processing_log.append({
                        'filter': filter_instance.__class__.__name__,
                        'before_length': before_length,
                        'after_length': after_length,
                        'modified': True
                    })
                    self.stats['filter_stats'][filter_instance.__class__.__name__] += 1
                    response = filtered_response
                else:
                    processing_log.append({
                        'filter': filter_instance.__class__.__name__,
                        'modified': False
                    })
                
                # 如果回應被完全清空或拒絕，停止處理
                if not response or response == "[REJECTED]":
                    reason = f"rejected_by_{filter_instance.__class__.__name__}"
                    self.stats['rejection_reasons'][reason] = self.stats['rejection_reasons'].get(reason, 0) + 1
                    
                    # 記錄更詳細的拒絕原因
                    if hasattr(filter_instance, '_last_rejection_reason'):
                        detailed_reason = getattr(filter_instance, '_last_rejection_reason', 'unknown')
                        logger.warning(f"回應被 {filter_instance.__class__.__name__} 拒絕: {detailed_reason}, 內容: {original_response[:50]}...")
                    else:
                        logger.warning(f"回應被 {filter_instance.__class__.__name__} 拒絕: {original_response[:50]}...")
                    
                    return None, {
                        'processed': True,
                        'rejected': True,
                        'reason': reason,
                        'processing_log': processing_log
                    }
                    
            except Exception as e:
                logger.error(f"過濾器 {filter_instance.__class__.__name__} 處理失敗: {e}")
                # 繼續使用原始回應，不因單個過濾器失敗而中斷
                continue
        
        # 檢查是否有修改
        was_modified = original_response != response
        total_modifications = len([log for log in processing_log if log['modified']])
        
        if was_modified:
            self.stats['total_modified'] += 1
        
        # 獲取品質分數
        quality_score = 10  # 預設分數
        if self.filters:
            # 使用最後一個品質驗證器的評分
            quality_filters = [f for f in self.filters if 'Quality' in f.__class__.__name__]
            if quality_filters:
                try:
                    quality_result = quality_filters[0].get_quality_score(response, user_input, context)
                    quality_score = quality_result.get('total_score', 10)
                except:
                    quality_score = 8  # 保守分數
        
        processing_stats = {
            'processed': True,
            'rejected': False,
            'modifications_made': was_modified,
            'total_modifications': total_modifications,
            'original_length': len(original_response),
            'final_length': len(response),
            'quality_score': quality_score,
            'processing_log': processing_log,
            'filters_applied': total_modifications
        }
        
        logger.debug(f"回應處理完成: {len(processing_log)} 個過濾器, {processing_stats['filters_applied']} 個生效")
        
        return response, processing_stats
    
    def validate_response(self, response: str, user_input: str = "", context: Dict = None) -> Tuple[bool, str]:
        """
        快速驗證回應是否合適（不修改內容）
        
        Args:
            response: 要驗證的回應
            user_input: 用戶輸入
            context: 對話上下文
            
        Returns:
            Tuple[bool, str]: (是否通過驗證, 失敗原因)
        """
        if not response:
            return False, "empty_response"
        
        # 只使用驗證類過濾器
        validator_filters = [f for f in self.filters if 'Validator' in f.__class__.__name__]
        
        for filter_instance in validator_filters:
            try:
                if hasattr(filter_instance, 'validate'):
                    is_valid, reason = filter_instance.validate(response, user_input, context)
                    if not is_valid:
                        return False, f"{filter_instance.__class__.__name__}:{reason}"
            except Exception as e:
                logger.error(f"驗證器 {filter_instance.__class__.__name__} 驗證失敗: {e}")
                continue
        
        return True, "passed"
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """獲取過濾器統計資料"""
        return {
            'total_processed': self.stats['total_processed'],
            'total_modified': self.stats['total_modified'],
            'modification_rate': (self.stats['total_modified'] / max(self.stats['total_processed'], 1)) * 100,
            'filter_effectiveness': self.stats['filter_stats'],
            'rejection_reasons': self.stats['rejection_reasons'],
            'active_filters': [f.__class__.__name__ for f in self.filters]
        }
    
    def add_filter(self, filter_instance: 'BaseResponseFilter', position: int = -1):
        """動態添加過濾器"""
        if position == -1:
            self.filters.append(filter_instance)
        else:
            self.filters.insert(position, filter_instance)
        
        self.stats['filter_stats'][filter_instance.__class__.__name__] = 0
        logger.info(f"添加過濾器: {filter_instance.__class__.__name__}")
    
    def remove_filter(self, filter_class_name: str) -> bool:
        """移除指定過濾器"""
        for i, filter_instance in enumerate(self.filters):
            if filter_instance.__class__.__name__ == filter_class_name:
                removed_filter = self.filters.pop(i)
                logger.info(f"移除過濾器: {filter_class_name}")
                return True
        
        logger.warning(f"未找到過濾器: {filter_class_name}")
        return False
    
    def reset_stats(self):
        """重置統計資料"""
        self.stats = {
            'total_processed': 0,
            'total_modified': 0,
            'filter_stats': {filter.__class__.__name__: 0 for filter in self.filters},
            'rejection_reasons': {}
        }
        logger.info("過濾器統計資料已重置")
