#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
內容清理過濾器
處理基礎的內容清理，如繁簡轉換、重複詞語等
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple
from .base_filter import BaseResponseFilter

logger = logging.getLogger(__name__)

class ContentCleanerFilter(BaseResponseFilter):
    """內容清理過濾器 - 基礎內容清理和格式化"""
    
    def __init__(self, chat_instance=None):
        super().__init__(chat_instance)
        
        # 嘗試初始化 OpenCC
        self.opencc_converter = None
        try:
            import opencc
            self.opencc_converter = opencc.OpenCC('s2t')
            logger.info("OpenCC 繁簡轉換器初始化成功")
        except ImportError:
            logger.warning("OpenCC 未安裝，跳過繁簡轉換功能")
        
        # 預編譯清理相關的正則表達式
        self._compile_patterns()
        
        logger.info("內容清理過濾器初始化完成")
    
    def _compile_patterns(self):
        """預編譯內容清理相關的正則表達式模式"""
        
        # 名字修正模式（從主程式遷移）
        name_corrections = [
            (r'露西[亚亞]', '露西亞'),
            (r'露希[雅亞]', '露西亞'),
            (r'露\s*西\s*[亚亞]', '露西亞'),
            (r'露\s*Lucia', '露西亞'),
            (r'러시아', '露西亞'),
            (r'ルシア', '露西亞'),
            (r'Rushia', '露西亞'),
            (r'露西asia', '露西亞'),
            (r'露西西[亚亞]', '露西亞'),
            (r'露西[𝑎a][亚亞]?', '露西亞'),
        ]
        
        self.name_correction_patterns = [(re.compile(pattern, re.IGNORECASE), replacement) 
                                        for pattern, replacement in name_corrections]
        
        # 重複短語清理模式
        repetition_patterns = [
            (r'(ありがとう[ねございます]*\s*){3,}', 'ありがとう♪'),
            (r'(おやすみ[なさい]*\s*){3,}', 'おやすみ♪'),
            (r'(はい\s*){3,}', 'はい♪'),
            (r'(嗯嗯\s*){3,}', '嗯嗯♪'),
            (r'(真的\s*){3,}', '真的♪'),
            (r'(好的\s*){3,}', '好的♪'),
        ]
        
        self.repetition_patterns = [(re.compile(pattern), replacement) 
                                   for pattern, replacement in repetition_patterns]
        
        # 系統用詞清理（從主程式遷移並擴充）
        system_terms = [
            'AI助手', 'AI助理', '小智', 'chen', '數據處理', '数据处理',
            '系統維護', '系统维护', '功能正常', '日常任務', '日常任务',
            '運作順暢', '运作顺畅', '用戶', '用户', '使用者',
            '客服', '机器人', '機器人', '程式', '程序', '頻道', '數據',
            '系統', '維護', '請求', '廚房', '準備', '晚餐', '做飯',
            '烹飪', '菜品', '廚具', '食材', '料理',
            # 新增：工作相關的奇怪延伸
            '今天第一天來上班', '第一天來上班', '來上班', '請問♪', 
            '工作日', '上班時間', '下班時間', '工作內容', '職場',
            '同事', '老闆', '主管', '員工', '辦公室', '工作的事情',
            '工作', '上班', '下班', '加班', '出差', '會議', '績效',
            '部門', '薪水', '考核', '聚餐', '報到'
        ]
        
        self.system_terms = system_terms
        
        # 標點符號優化模式
        punctuation_patterns = [
            (r'[。！？]{3,}', '♪'),      # 過多句號轉表情符號
            (r'♪\s*♪+', '♪'),           # 合併多個♪
            (r'♡\s*♡+', '♡'),           # 合併多個♡
            (r'～\s*～+', '～'),          # 合併多個～
            (r'\s+', ' '),              # 多空格合併
        ]
        
        self.punctuation_patterns = [(re.compile(pattern), replacement) 
                                    for pattern, replacement in punctuation_patterns]
        
        # 特定詞彙修正模式（處理 OpenCC 沒有正確轉換的詞）
        vocabulary_corrections = [
            (r'材質', '材料'),
            (r'纔', '才'),
            (r'喫', '吃'),
            (r'妳', '你'),
            # 移除過度修正的詞彙
            # (r'徵', '征'),  # 這個可能會誤修正
            # (r'贈', '送'),  # 這個可能會誤修正
            # (r'獲', '得'),  # 這個可能會誤修正
            # 只保留明確的粵語/繁體詞彙修正
            (r'嚟', '來'),
            (r'啲', '些'),
            (r'乜', '什麼'),
            (r'邊度', '哪裡'),  # 更精確的匹配
            (r'點解', '為什麼'),  # 更精確的匹配
            (r'唔', '不'),
            (r'冇', '沒有'),
        ]
        self.vocabulary_correction_patterns = [(re.compile(pattern), replacement) for pattern, replacement in vocabulary_corrections]

        logger.debug(f"預編譯了內容清理相關模式: {len(self.name_correction_patterns) + len(self.repetition_patterns) + len(self.punctuation_patterns)} 個")
    
    def _apply_vocabulary_corrections(self, response: str) -> str:
        """應用特定詞彙修正"""
        for pattern, replacement in self.vocabulary_correction_patterns:
            response = pattern.sub(replacement, response)
        return response
    
    def filter(self, response: str, user_input: str = "", context: Dict = None) -> str:
        """
        基礎內容清理
        
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
        
        # 第一階段：繁簡轉換
        if self.opencc_converter:
            try:
                response = self.opencc_converter.convert(response)
            except Exception as e:
                logger.warning(f"繁簡轉換失敗: {e}")
        
        # 新增：詞彙修正
        response = self._apply_vocabulary_corrections(response)
        
        # 第二階段：名字修正
        for pattern, replacement in self.name_correction_patterns:
            response = pattern.sub(replacement, response)
        
        # 第三階段：重複短語清理
        for pattern, replacement in self.repetition_patterns:
            response = pattern.sub(replacement, response)
        
        # 第四階段：系統用詞移除
        for term in self.system_terms:
            response = response.replace(term, '')
        
        # 第五階段：標點符號優化
        for pattern, replacement in self.punctuation_patterns:
            response = pattern.sub(replacement, response)
        
        # 第六階段：最終清理
        response = response.strip()
        
        # 第七階段：特殊情況處理
        response = self._handle_special_cases(response, user_input)
        
        return response
    
    def _handle_special_cases(self, response: str, user_input: str) -> str:
        """
        處理特殊情況
        
        Args:
            response: 回應內容
            user_input: 用戶輸入
            
        Returns:
            str: 處理後的回應
        """
        # 處理重複的「想」字組合（改善而不刪除）
        response = re.sub(r'想([^想]{1,8})\s+想([^想]{1,8})', r'想\1，也想\2', response)
        
        # 特殊情況：如果是純情感表達，保留但優化
        if re.match(r'^想\w+.*?想\w+.*?$', response.strip()) and len(response.strip()) < 30:
            response = re.sub(r'想(\w+)\s+想(\w+)', r'想\1，也想\2', response)
        
        # 確保不會意外清空重要內容
        if not response.strip():
            logger.warning("內容清理過濾器意外清空了回應")
            return "嗯嗯♪"
        
        return response
    
    def validate(self, response: str, user_input: str = "", context: Dict = None) -> Tuple[bool, str]:
        """驗證內容清理是否合適"""
        if not response:
            return False, "empty_response"
        
        # 檢查是否還有未清理的系統用詞
        for term in ['數據', '系統', '程式', '機器人']:
            if term in response:
                return False, f"contains_system_term:{term}"
        
        # 檢查是否有過多的重複
        if re.search(r'(.{2,})\1{3,}', response):
            return False, "excessive_repetition"
        
        # 檢查長度是否合理
        if len(response.strip()) < 2:
            return False, "too_short"
        
        return True, "passed"
    
    def clean_specific_type(self, response: str, clean_type: str) -> str:
        """
        清理特定類型的內容
        
        Args:
            response: 回應內容
            clean_type: 清理類型 ('names', 'repetition', 'system', 'punctuation')
            
        Returns:
            str: 清理後的回應
        """
        if clean_type == 'names':
            for pattern, replacement in self.name_correction_patterns:
                response = pattern.sub(replacement, response)
        elif clean_type == 'repetition':
            for pattern, replacement in self.repetition_patterns:
                response = pattern.sub(replacement, response)
        elif clean_type == 'system':
            for term in self.system_terms:
                response = response.replace(term, '')
        elif clean_type == 'punctuation':
            for pattern, replacement in self.punctuation_patterns:
                response = pattern.sub(replacement, response)
        
        return response.strip()
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """獲取清理統計"""
        return {
            'name_correction_patterns': len(self.name_correction_patterns),
            'repetition_patterns': len(self.repetition_patterns),
            'system_terms_count': len(self.system_terms),
            'punctuation_patterns': len(self.punctuation_patterns),
            'opencc_available': self.opencc_converter is not None,
            'total_patterns': (len(self.name_correction_patterns) + 
                             len(self.repetition_patterns) + 
                             len(self.punctuation_patterns))
        }
    
    def _clean_special_chars_for_length_check(self, response: str) -> str:
        """
        清理特殊字符以進行長度檢查
        移除表情符號等不影響實際內容的字符
        
        Args:
            response: 回應內容
            
        Returns:
            str: 清理後用於長度檢查的內容
        """
        if not response:
            return ""
        
        # 移除常見的表情符號和裝飾字符
        cleaned = response
        
        # 移除表情符號
        cleaned = cleaned.replace('♪', '')
        cleaned = cleaned.replace('♡', '')
        cleaned = cleaned.replace('～', '')
        cleaned = cleaned.replace('♥', '')
        cleaned = cleaned.replace('♬', '')
        cleaned = cleaned.replace('♫', '')
        cleaned = cleaned.replace('★', '')
        cleaned = cleaned.replace('☆', '')
        
        # 移除多餘空格
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def _apply_vocabulary_corrections(self, response: str) -> str:
        """應用特定詞彙修正"""
        for pattern, replacement in self.vocabulary_correction_patterns:
            response = pattern.sub(replacement, response)
        return response
