#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色混亂過濾器
處理角色扮演混亂、第三人稱描述、異常角色出現等問題
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple, List
from .base_filter import BaseResponseFilter

logger = logging.getLogger(__name__)

class CharacterConfusionFilter(BaseResponseFilter):
    """角色混亂過濾器 - 清除非露西亞角色內容和角色扮演混亂"""
    
    def __init__(self, chat_instance=None):
        super().__init__(chat_instance)
        
        # 預編譯所有角色混亂相關的正則表達式
        self._compile_patterns()
        
        logger.info("角色混亂過濾器初始化完成")
    
    def _compile_patterns(self):
        """預編譯角色混亂相關的正則表達式模式"""
        
        # 異常角色名稱模式（從主程式遷移）
        character_confusion_patterns = [
            r'.*?艾瑞克.*?$',
            r'.*?Erik.*?$', 
            r'.*?他向前走.*?$',
            r'.*?搞什麼鬼.*?$',
            r'".*?".*?他.*?$',
            r'別以為.*?不知道.*?$',
            r'.*?不知道你在.*?$',
            r'我要以.*?身份.*?$',
            r'現在我要以.*?身份.*?$', 
            r'好的.*?我明白.*?現在.*?$',
            r'好的.*?那我們來.*?角色扮演.*?$',  # 更精確，只匹配角色扮演相關
            r'.*?繼續對話.*?$',
            r'.*?角色扮演.*?$',
            r'.*?身份.*?對話.*?$',
            r'"[^"]*".*?他.*?$',  # 移除包含引號和「他」的內容
            r'.*?搞鬼.*?$',
            r'.*?威脅.*?$',
        ]
        
        self.character_confusion_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in character_confusion_patterns]
        
        # 第三人稱描述模式
        third_person_patterns = [
            r'[。！？]*\s*他.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*她.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*那個人.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*某人.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*這個人.*?(?=[。！？♪♡～]|$)',
        ]
        
        self.third_person_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in third_person_patterns]
        
        # 角色扮演指示模式
        roleplay_patterns = [
            r'[。！？]*\s*扮演.*?角色.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*假裝.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*演.*?角色.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*當作.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*模擬.*?(?=[。！？♪♡～]|$)',
        ]
        
        self.roleplay_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in roleplay_patterns]
        
        # 身份混亂模式
        identity_confusion_patterns = [
            r'[。！？]*\s*我是.*?(助手|AI|機器人|程式).*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*我的身份是.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*我要以.*?身份.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*切換到.*?模式.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*現在我是.*?(?=[。！？♪♡～]|$)',
            # 新增：強力身份聲明檢測
            r'[。！？]*\s*我是[^露。！？♪♡～]*?[，。！？♪♡～].*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*我叫[^露。！？♪♡～]*?[，。！？♪♡～].*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*我的名字是[^露。！？♪♡～]*?[，。！？♪♡～].*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*.*?城堡.*?主人.*?(?=[。！？♪♡～]|$)',
            r'[。！？]*\s*.*?的主人.*?(?=[。！？♪♡～]|$)',
        ]
        
        self.identity_confusion_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in identity_confusion_patterns]
        
        # 異常名字變體模式（從主程式遷移並擴充）
        strange_names = [
            'chen', '小智', '露西安', '燻肉', '艾瑞克', 'Erik',
            '用戶', '用户', '使用者', '客服', '助手', '助理',
            '機器人', '机器人', '程式', '程序', '系統', '系统',
            '愛德華', '賽倫特', 'Edward', 'Serent', '城堡主人',
            '主人', '男爵', '伯爵', '公爵', '王子', '國王',
            '騎士', '勇者', '魔法師', '法師', '戰士', '盜賊',
            '冒險者', '領主', '貴族', '管家', '僕人'
        ]
        
        self.strange_name_patterns = [re.compile(rf'\b{re.escape(name)}\b', re.IGNORECASE) for name in strange_names]
        
        logger.debug(f"預編譯了角色混亂相關模式: {len(self.character_confusion_patterns) + len(self.third_person_patterns) + len(self.roleplay_patterns) + len(self.identity_confusion_patterns) + len(self.strange_name_patterns)} 個")
    
    def filter(self, response: str, user_input: str = "", context: Dict = None) -> str:
        """
        過濾角色混亂內容
        
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
        
        # 第一階段：移除明顯的角色混亂模式
        for pattern in self.character_confusion_patterns:
            response = pattern.sub('', response)
        
        # 第二階段：處理第三人稱描述
        for pattern in self.third_person_patterns:
            # 檢查是否是正常的第三人稱表達
            if not self._is_normal_third_person(response):
                response = pattern.sub('', response)
        
        # 第三階段：移除角色扮演指示
        for pattern in self.roleplay_patterns:
            response = pattern.sub('', response)
        
        # 第四階段：處理身份混亂
        for pattern in self.identity_confusion_patterns:
            response = pattern.sub('', response)
        
        # 第五階段：替換異常名字
        for pattern in self.strange_name_patterns:
            response = pattern.sub('露西亞', response)
        
        # 第六階段：清理空白和重複標點
        response = re.sub(r'\s+', ' ', response.strip())
        
        # 第七階段：檢查是否過度清理
        if not response.strip() and original_response.strip():
            logger.warning(f"角色混亂過濾器清空了回應: {original_response[:50]}...")
            return self._get_recovery_response(user_input)
        
        return response
    
    def _is_normal_third_person(self, response: str) -> bool:
        """
        檢查是否是正常的第三人稱表達
        
        Args:
            response: 回應內容
            
        Returns:
            bool: 是否是正常表達
        """
        # 如果是在講故事或描述別人，可能是正常的
        normal_indicators = [
            '朋友', '家人', '同事', '老師', '學生', '醫生',
            '店員', '司機', '鄰居', '同學', '老闆'
        ]
        
        return any(indicator in response for indicator in normal_indicators)
    
    def _get_recovery_response(self, user_input: str) -> str:
        """當角色混亂過濾導致回應為空時的恢復回應"""
        import random
        
        recovery_responses = [
            "嗯嗯♪讓我重新整理一下思緒♡",
            "抱歉♪剛才有點混亂了呢♡",
            "嗯～♪讓露醬重新回應你♡",
            "哎呀♪剛才說得有點奇怪呢♡",
            "重新來一次♪露醬會好好回應的♡"
        ]
        
        return random.choice(recovery_responses)
    
    def validate(self, response: str, user_input: str = "", context: Dict = None) -> Tuple[bool, str]:
        """驗證回應是否包含角色混亂"""
        if not response:
            return False, "empty_response"
        
        # 檢查異常角色名稱
        problematic_names = ['艾瑞克', 'Erik', 'chen', '小智', '助手', '機器人']
        for name in problematic_names:
            if name in response:
                return False, f"contains_problematic_character:{name}"
        
        # 檢查角色扮演指示
        if any(keyword in response for keyword in ['扮演', '假裝', '當作', '模擬']):
            return False, "contains_roleplay_instruction"
        
        # 檢查身份混亂
        if re.search(r'我是.*?(助手|AI|機器人)', response):
            return False, "contains_identity_confusion"
        
        # 檢查不當的第三人稱描述
        if '"' in response and '他' in response:
            return False, "contains_inappropriate_third_person"
        
        return True, "passed"
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """獲取過濾器詳細統計"""
        base_stats = super().get_stats()
        
        pattern_stats = {
            'character_confusion_patterns': len(self.character_confusion_patterns),
            'third_person_patterns': len(self.third_person_patterns),
            'roleplay_patterns': len(self.roleplay_patterns),
            'identity_confusion_patterns': len(self.identity_confusion_patterns),
            'strange_name_patterns': len(self.strange_name_patterns),
            'total_patterns': (len(self.character_confusion_patterns) + 
                             len(self.third_person_patterns) + 
                             len(self.roleplay_patterns) + 
                             len(self.identity_confusion_patterns) + 
                             len(self.strange_name_patterns))
        }
        
        base_stats.update(pattern_stats)
        return base_stats
