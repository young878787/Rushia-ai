#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析性語言過濾器
專門移除模型思考過程和分析性語言洩露
"""

import re
import logging
import random
from typing import Dict, Any, Optional, Tuple, List
from .base_filter import BaseResponseFilter

logger = logging.getLogger(__name__)

class AnalysisLanguageFilter(BaseResponseFilter):
    """分析性語言過濾器 - 徹底移除模型內心話和分析性表達"""
    
    def __init__(self, chat_instance=None):
        super().__init__(chat_instance)
        
        # 預編譯正則表達式以提高效能
        self._compile_patterns()
        
        logger.info("分析性語言過濾器初始化完成")
    
    def _compile_patterns(self):
        """預編譯所有正則表達式模式 - 從主程式遷移並增強"""
        
        # 核心分析性語言模式 - 最強化版本
        core_patterns = [
            # 核心問題：「當前的情境是...」系列 - 精確匹配
            r'[。！？]*\s*當前的情境是.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*當前.*?情境.*?(是|要|需要|應該).*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*現在的情境.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*目前的情境.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*這個情境.*?(是|要|需要|應該).*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*在.*?情境.*?(下|中).*?(?=[。！？♪♡～]|$)',
            
            # 回應分析系列
            r'[。！？]*\s*回覆說.*?然後.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*我回應.*?表現.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*這樣回應.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*回應.*?方式.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*我的回應.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*回覆.*?內容.*?(?=[。！？♪♡～]|$)',
            
            # 分析動詞系列
            r'[。！？]*\s*根據.*?分析.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*基於.*?判斷.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*考慮到.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*因此.*?回應.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*所以.*?表現.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*這是.*?回應.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*接下來.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*在這種情況下.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*針對.*?內容.*?(?=[。！？♪♡～]|$)',
            
            # 角色表現分析
            r'[。！？]*\s*露西亞.*?表現.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*展現.*?特質.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*體現.*?性格.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*反映.*?情感.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*表達.*?態度.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*呈現.*?反應.*?(?=[。！？♪♡～]|$)',
            
            # 系統性語言
            r'[。！？]*\s*模型.*?生成.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*系統.*?處理.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*現在.*?回覆.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*根據用戶.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*處理.*?請求.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*給出.*?回應.*?(?=[。！？♪♡～]|$)',
            
            # 角色設定語言
            r'[。！？]*\s*作為.*?角色.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*按照.*?設定.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*依據.*?指示.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*遵循.*?原則.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*符合.*?人設.*?(?=[。！？♪♡～]|$)',
            
            # 互動目標語言
            r'[。！？]*\s*顯示.*?關心.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*傳達.*?溫暖.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*營造.*?氛圍.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*創造.*?情境.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*建立.*?連結.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*加深.*?關係.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*增進.*?互動.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*促進.*?交流.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*維持.*?對話.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*延續.*?話題.*?(?=[。！？♪♡～]|$)',
            
            # 功能性語言
            r'[。！？]*\s*回應.*?需求.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*滿足.*?期待.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*達到.*?效果.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*實現.*?目標.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*完成.*?任務.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*執行.*?指令.*?(?=[。！？♪♡～]|$)',
            
            # 技術性語言
            r'[。！？]*\s*運行.*?程序.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*啟動.*?機制.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*觸發.*?邏輯.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*調用.*?功能.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*載入.*?模組.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*解析.*?語意.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*識別.*?意圖.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*判斷.*?情境.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*評估.*?需要.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*計算.*?回應.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*生成.*?內容.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*產出.*?結果.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*輸出.*?文字.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*返回.*?訊息.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*提供.*?回饋.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*給予.*?反應.*?(?=[。！？♪♡～]|$)',
            
            # 新增：更多機械化表達
            r'[。！？]*\s*執行.*?指令.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*啟動.*?模式.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*切換.*?狀態.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*載入.*?程序.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*運行.*?算法.*?(?=[。！？♪♡～]|$)',
            
            # 特定危險組合 - 更精確的匹配
            r'[。！？]*\s*.*?回覆說.*?然後.*?(?=[。！？♪♡～]|$)',
            
            # 新增：模型提示詞和行為指令過濾
            r'[。！？]*\s*溫柔示好型.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*關懷支持型.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*親密互動型.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*甜美可愛型.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*溫暖陪伴型.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*積極回應型.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*.*?型\s*「.*?」.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*.*?型\s*.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*指令.*?執行.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*回應類型.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*行為模式.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*回覆模式.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*表現風格.*?(?=[。！？♪♡～]|$)',
            
            # 新增：引用標記和說明文字
            r'[。！？]*\s*「.*?」\s*這樣.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*.*?說明.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*.*?註解.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*.*?提示.*?(?=[。！？♪♡～]|$)',
        ]
        
        # 編譯所有模式
        self.core_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in core_patterns]
        
        # 句尾不完整分析片段模式
        incomplete_patterns = [
            r'然後[♪♡～]*$',
            r'接著[♪♡～]*$',
            r'這樣[♪♡～]*$',
            r'所以[♪♡～]*$',
            r'因此[♪♡～]*$',
            r'現在[♪♡～]*$',
            r'然後就[♪♡～]*$',
            r'接下來[♪♡～]*$',
            r'之後[♪♡～]*$',
            r'再來[♪♡～]*$',
        ]
        
        self.incomplete_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in incomplete_patterns]
        
        # 開頭分析詞彙模式
        beginning_patterns = [
            r'^(因此|所以|目標|策略|方法|深化|保持|探索|製造).*?[。！？]\s*',
        ]
        
        self.beginning_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in beginning_patterns]
        
        logger.debug(f"預編譯了 {len(self.core_patterns)} 個核心模式")
    
    def filter(self, response: str, user_input: str = "", context: Dict = None) -> str:
        """
        過濾分析性語言 - 從主程式遷移的核心邏輯
        
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
        
        # 第一階段：核心分析性語言清理
        for pattern in self.core_patterns:
            response = pattern.sub('', response)
        
        # 第二階段：移除句尾的不完整分析片段
        for pattern in self.incomplete_patterns:
            response = pattern.sub('', response)
        
        # 第三階段：移除句子開頭的分析詞彙
        for pattern in self.beginning_patterns:
            response = pattern.sub('', response)
        
        # 第四階段：清理多餘的空白和標點
        response = re.sub(r'\s+', ' ', response.strip())
        response = re.sub(r'♪\s*♪+', '♪', response)  # 合併多個♪
        
        # 第五階段：檢查是否過度清理
        response = self._post_process_check(response, original_response, user_input)
        
        return response
    
    def _post_process_check(self, filtered_response: str, original_response: str, user_input: str) -> str:
        """
        後處理檢查，確保沒有過度清理
        
        Args:
            filtered_response: 過濾後的回應
            original_response: 原始回應
            user_input: 用戶輸入
            
        Returns:
            str: 最終回應
        """
        # 如果回應被完全清空，提供安全的後備
        if not filtered_response.strip():
            logger.warning(f"分析性語言過濾器清空了回應: {original_response[:50]}...")
            return self._get_safe_fallback(user_input)
        
        # 檢查原始回應是否很短且簡單（如 "好的"、"是的" 等）
        # 這種情況不需要後備回應
        original_clean = original_response.replace('♪', '').replace('♡', '').replace('～', '').strip()
        if len(original_clean) <= 5 and not any(pattern in original_response for pattern in ['分析', '根據', '模型', '系統', '處理']):
            return filtered_response
        
        # 如果回應太短且沒有情感符號，且原始回應較長，可能過度清理了
        clean_length = len(filtered_response.replace('♪', '').replace('♡', '').replace('～', '').strip())
        original_length = len(original_clean)
        if clean_length < 3 and original_length > 10:
            logger.warning(f"分析性語言過濾器可能過度清理: {original_response[:50]}...")
            return self._get_safe_fallback(user_input)
        
        return filtered_response
    
    def _get_safe_fallback(self, user_input: str) -> str:
        """當過度清理時的安全後備回應"""
        safe_responses = [
            "嗯嗯♪我在聽呢～♡",
            "是這樣呀～♪說來聽聽♡", 
            "哇♪聽起來很有趣呢♡",
            "嗯～♪露醬想知道更多呢♡",
            "原來如此♪♡",
            "真的嗎？♪好棒呢♡"
        ]
        
        return random.choice(safe_responses)
    
    def validate(self, response: str, user_input: str = "", context: Dict = None) -> Tuple[bool, str]:
        """驗證回應是否包含分析性語言"""
        if not response:
            return False, "empty_response"
        
        # 檢查是否包含明顯的分析性語言
        analysis_indicators = [
            '當前的情境是', '根據分析', '基於判斷', '考慮到',
            '這樣回應', '我回應', '表現出', '展現', '體現',
            '模型生成', '系統處理', '按照設定'
        ]
        
        for indicator in analysis_indicators:
            if indicator in response:
                return False, f"contains_analysis_language:{indicator}"
        
        # 檢查是否有「回覆說...然後」的模式
        if re.search(r'回覆說.*?然後', response):
            return False, "contains_response_analysis_pattern"
        
        return True, "passed"
    
    def get_pattern_stats(self) -> Dict[str, int]:
        """獲取各種模式的匹配統計"""
        return {
            'core_patterns_count': len(self.core_patterns),
            'incomplete_patterns_count': len(self.incomplete_patterns),
            'beginning_patterns_count': len(self.beginning_patterns),
            'total_patterns': len(self.core_patterns) + len(self.incomplete_patterns) + len(self.beginning_patterns)
        }
