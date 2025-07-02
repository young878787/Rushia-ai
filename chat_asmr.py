#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
露西亞ASMR LoRA聊天腳本 - 主程式
專門解決回應重複和品質問題
"""

import torch
import sys
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re
import random
from opencc import OpenCC

# 導入回應過濾器模組
from response_filters import FilterManager

# 導入回應模組
from rushia_responses import (
    IntimateResponses,
    FoodResponses,
    EmotionalSupportResponses,
    DailyChatResponses,
    TimeAwareResponses,
    BaseResponses
)

# 導入語義分析模組
from semantic_analysis import SemanticAnalysisManager

# 導入記憶管理模組
from memory_management import MemoryManager

# 設定 logger
logger = logging.getLogger(__name__)

class RushiaLoRAChat:    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.base_model_path = "D:/RushiaMode/models/Qwen3-8B" # 基礎模型路徑
        self.lora_model_path = "D:/RushiaMode/models/rushia-qwen3-8b-lora-asmr-8bit" # LoRA權重路徑
        
        # 語義分析初始化
        self.jieba_available = False
        self._init_semantic_analysis()
        
        # 初始化記憶管理器 - 新的模組化架構
        self.memory_manager = MemoryManager({
            'conversation': {
                'max_history': 10,
                'max_response_history': 12
            },
            'user_profile': {},  # UserProfileManager 使用預設參數
            'context_cache': {
                'max_emotions': 30,
                'max_themes': 20,
                'max_preferences': 50
            }
        })
        
        # 為了向後兼容，保留原有的屬性但指向新的管理器
        self._setup_compatibility_properties()
        
        # 注意：語義關鍵詞庫已遷移至 semantic_analysis.keyword_config 模組
        # 為了向後兼容，保留一個屬性指向模組配置
        from semantic_analysis import keyword_config
        self.semantic_keywords = keyword_config.semantic_keywords
        
        # 時間感知
        self.current_hour = time.localtime().tm_hour
        
        # OpenCC 簡繁轉換器
        self.opencc_converter = OpenCC('s2tw')  # 簡體轉繁體（台灣標準）
        
        # 初始化所有回應模組
        self.intimate_responses = IntimateResponses(self)
        self.food_responses = FoodResponses(self)
        self.emotional_support = EmotionalSupportResponses(self)
        self.daily_chat = DailyChatResponses(self)
        self.time_aware_responses = TimeAwareResponses(self)
        self.base_responses = BaseResponses(self)
        
        # 初始化回應過濾器管理器
        self.filter_manager = FilterManager()
        
        # 初始化語義分析管理器（傳入self以保持兼容性）
        self.semantic_manager = SemanticAnalysisManager(self)
        
        # 簡化語義分析初始化
        self._init_jieba_if_available()
        
        # 主動訊息系統
        self.proactive_system = {
            'last_message_time': None,  # 上次訊息時間
            'last_user_message_time': None,  # 上次用戶訊息時間
            'last_proactive_message_time': None,  # 上次主動訊息時間
            'waiting_for_response': False,  # 是否在等待回應
            'reminder_sent': False,  # 是否已發送催促訊息
            'reminder_count': 0,  # 催促訊息次數
            'silence_duration': 0,  # 沉默時長（分鐘）
            'daily_proactive_count': 0,  # 今日主動訊息次數
            'last_proactive_date': None,  # 上次主動訊息日期
        }
        
        # 時間感知主動關心系統
        self.time_aware_care_system = {
            'last_check_date': None,  # 上次檢查日期
            'daily_care_sent': {  # 每日各時段關心訊息發送狀態
                'morning': False,     # 早晨 (7-8點)
                'lunch': False,       # 中午 (11-13點) 
                'afternoon': False,   # 下午 (14-16點)
                'dinner': False,      # 晚上 (18-21點)
                'night': False        # 夜晚 (21-24點)
            },
            'care_sent_times': {},    # 各時段實際發送時間記錄
            'enabled': True           # 是否啟用時間感知關心
        }
    
    def load_model(self):
        """載入模型和LoRA權重 - 優化版"""
        print("🔄 載入模型中...")
        
        try:
            # 8-bit量化配置 - 更快更穩定
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            
            # 載入tokenizer
            print("📝 載入tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                use_fast=True  # 使用更快的tokenizer
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 載入基礎模型
            print("🤖 載入基礎模型...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,  # 減少CPU記憶體使用
                use_cache=True  # 啟用KV cache加速
            )
            
            # 載入LoRA模型
            print("✨ 載入LoRA權重...")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.lora_model_path,
                is_trainable=False,
                torch_dtype=torch.float16
            )
            
            # 合併adapter權重以獲得更好的推理速度
            print("🔄 合併adapter權重...")
            self.model = self.model.merge_and_unload()
            
            print(f"✅ 模型載入完成！使用設備: {self.device}")
            
            # 簡化的回應模組驗證
            if all(hasattr(self, attr) for attr in ['intimate_responses', 'food_responses', 'emotional_support', 'daily_chat', 'base_responses']):
                print("✅ 所有回應模組驗證通過")
            else:
                print("⚠️ 部分回應模組驗證失敗，將使用後備機制")
                
            return True
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            return False
    
    def chat(self, user_input):
        """主要聊天方法 - 完整版本"""
        response, processing_time = self.generate_response(user_input)
        return response
    
    def generate_response(self, user_input, max_new_tokens=None, temperature=None):
        """生成回應 - 智能調整參數提升回應豐富度和變化性"""
        # 記錄開始時間
        start_time = time.time()
        
        # 如果模型未載入，使用回應模組系統
        if self.model is None or self.tokenizer is None:
            logger.info("模型未載入，使用回應模組系統")
            response = self._get_response(user_input)
            
            # 確保回應經過完整的後處理
            if response is None:
                response = self._get_fallback_response(user_input)
                
            # 最終回應豐富度增強
            response = self._enhance_response_richness(response, user_input)
            
            # 防止連續短回應
            response = self._prevent_consecutive_short_responses(response, user_input)
            
            # 添加到對話歷史
            self._add_to_history(user_input, response)
            
            # 更新用戶個人資料
            self._update_user_profile(user_input, response)
            
            # 計算處理時間
            processing_time = time.time() - start_time
            return response, processing_time
        
        # 智能調整生成參數，增加回應長度和豐富度的變化性
        context_aware_params = self._get_dynamic_generation_params(user_input, max_new_tokens, temperature)
        
        # 構建包含歷史的上下文
        context = self._build_context(user_input)
        
        # 更嚴格的ASMR LoRA訓練風格prompt - 明確規範內容和語言，真人化自稱，防止角色混亂
        prompt_templates = [
            f"露西亞溫柔地用純繁體中文回應，使用真人化的自稱如「我」、「露西亞」、「露醬我」。語氣甜美可愛，不會使用日文、不會說奇怪內容、不會產生對話格式、不會扮演其他角色。她只是露西亞，不是艾瑞克或任何其他人。\n{context}露西亞: ",
            f"露西亞用繁體中文溫柔回應，自稱用「我」或「露西亞」、「露醬我」，只表達溫暖親切的感情。禁止日文混雜、禁止不當描述、禁止對話格式、禁止角色扮演其他人物。\n{context}露西亞: ",
            f"露西亞甜美地用繁中回應，真人化地使用「我」、「露西亞」、「露醬我」等自稱，語氣溫柔體貼。她只說合適的話，不混用語言，不產生多人對話，不扮演男性角色或其他人物。\n{context}露西亞: "
        ]
        
        prompt = random.choice(prompt_templates)
        
        # 編碼輸入
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=350,  # 進一步增加以容納更長的對話
            padding=False
        ).to(self.device)
        
        # 使用torch.inference_mode()替代no_grad()提升性能
        with torch.inference_mode():
            # 使用動態調整的生成參數 - 大幅增加回應豐富度和變化性
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=context_aware_params['max_tokens'],
                temperature=context_aware_params['temperature'],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=context_aware_params['repetition_penalty'],
                top_p=context_aware_params['top_p'],
                top_k=context_aware_params['top_k'],
                no_repeat_ngram_size=context_aware_params['no_repeat_ngram'],
                # 添加更嚴格的停止條件
                stopping_criteria=None   # 將依賴後處理來處理
            )
        
        # 計算處理時間
        processing_time = time.time() - start_time
        
        # 解碼回應
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取新生成的部分
        response = ""
        prompt_end_marker = "露西亞: "
        if prompt_end_marker in full_response:
            parts = full_response.split(prompt_end_marker)
            if len(parts) > 1:
                response = parts[-1].strip()
        
        if not response:
            response = full_response[len(prompt):].strip()
        
        # 使用過濾器管理器處理回應（包含自稱優化等所有過濾邏輯）
        response, _ = self.filter_manager.process_response(response, user_input)
        
        # 如果回應太短，優先使用專業問答庫（使用過濾器判斷長度）
        if response and self._is_response_too_short(response):
            # 先嘗試專業問答庫
            professional_response = self._get_response(user_input)
            if professional_response and len(professional_response.strip()) >= 8:
                response = professional_response
            else:
                # 如果專業問答庫也沒有長回應，才使用通用擴展
                expanded_response = self._expand_short_response(response, user_input)
                if expanded_response:
                    response = expanded_response
        
        # 使用模組化過濾器進行回應處理
        response, filter_stats = self.filter_manager.process_response(response, user_input)
        
        # 如果過濾器拒絕了回應，使用安全的替代回應
        if not response:
            logger.warning("過濾器拒絕了回應，使用安全替代回應")
            response = "嗯嗯♪露醬在這裡陪著你呢♡"
        
        # 記錄過濾統計（如果需要）
        if filter_stats and filter_stats.get('modifications_made'):
            logger.info(f"回應已通過過濾器處理，修改次數: {filter_stats.get('total_modifications', 0)}")
        
        # 如果過濾後回應為空或品質不佳，使用智能回應模組分派系統
        if not response or len(response.strip()) < 3:
            logger.warning("過濾後回應為空或過短，使用智能分派系統")
            # 使用新的智能分派系統
            response = self._get_response(user_input)
            # 對新回應再次進行過濾處理
            filtered_response, _ = self.filter_manager.process_response(response, user_input)
            if filtered_response:  # 確保二次過濾不會再次拒絕
                response = filtered_response
        
        # 偶爾主動提起話題 - 但要避免在親密對話中突兀轉換
        if (self._should_initiate_topic() and 
            not self._is_intimate_context_safe(user_input, response)):
            topic = self._get_topic_initiation()
            response += f" {topic}"
        
        # 最終回應豐富度增強 - 避免突然變短
        response = self._enhance_response_richness(response, user_input)
        
        # 確保 response 不是 None
        if response is None:
            response = self._get_fallback_response(user_input)
        
        # 防止連續短回應，確保對話品質不下降
        response = self._prevent_consecutive_short_responses(response, user_input)
        
        # 添加到對話歷史
        self._add_to_history(user_input, response)
        
        # 更新用戶個人資料
        self._update_user_profile(user_input, response)
        
        # 返回回應和處理時間
        return response, processing_time
    
    def _get_response(self, user_input):
        """獲取回應的主要方法 - 智能分派到不同回應模組（增強版）"""
        
        # 驗證模組是否正確初始化
        if not self._validate_response_modules():
            return self._get_fallback_response(user_input)
        
        # 使用新的語義分析管理器進行分析
        analysis_result = self.semantic_manager.analyze_comprehensive(
            user_input=user_input,
            conversation_history=self.conversation_history[-5:],  # 提供最近5輪對話
            user_profile=self.user_profile,
            context_cache=self.context_cache
        )
        
        # 提取分析結果（保持向後兼容）
        intent = analysis_result['intent']
        context = analysis_result['context'] 
        emotion = analysis_result['emotion']
        intimacy = analysis_result['intimacy']
        
        # 記錄分析結果到日誌 - 修正 KeyError 問題
        logger.debug(f"用戶意圖分析: 情感={emotion['type']}, 話題={intent.get('topic')}, 親密度={intimacy['score']}")
        logger.debug(f"對話上下文: 風格偏好={context.get('preferred_style')}, 參與度={context.get('user_engagement')}")
        
        # 0. 首先嘗試生成創意回應（針對親密或複雜情境）
        creative_response = self._generate_creative_response(user_input, intent, context)
        if creative_response:
            return creative_response
        
        # 1. 檢查時間感知回應（優先級最高）
        if intent.get('time_sensitivity', False):
            time_response = self._safe_get_module_response(
                self.time_aware_responses, 'get_response', user_input, context
            )
            if time_response:
                return time_response
        
        # 2. 檢查情感支持回應（負面情緒優先處理） - 修正 KeyError 問題
        if (emotion['type'] == 'negative' and emotion['intensity'] < -0.3) or intent.get('conversation_intent') in ['seeking_comfort', 'work_stress']:
            emotional_response = self._safe_get_module_response(
                self.emotional_support, 'get_response', user_input, context
            )
            if emotional_response:
                return emotional_response
        
        # 3. 檢查親密情境回應（高親密度優先處理） - 修正 KeyError 問題
        if intimacy['score'] >= 1.5 or intent.get('conversation_intent') == 'expressing_love':
            intimate_response = self._safe_get_module_response(
                self.intimate_responses, 'get_response', user_input, context
            )
            if intimate_response:
                return intimate_response
        
        # 4. 檢查食物相關回應（但排除已由日常聊天處理的特殊組合） - 修正 KeyError 問題
        if intent.get('topic') == 'food':  # 只處理純粹的食物話題
            food_response = self._safe_get_module_response(
                self.food_responses, 'get_response', user_input, context
            )
            if food_response:
                return food_response
        
        # 5. 檢查日常聊天回應（包含特殊的陪伴+食物組合和問候語） - 修正 KeyError 問題
        if intent.get('topic') in ['daily_chat', 'greeting', 'companionship_food'] or not intent.get('topic'):
            daily_response = self._safe_get_module_response(
                self.daily_chat, 'get_response', user_input, context
            )
            if daily_response:
                return daily_response
        
        # 6. 基礎回應作為最後備案
        base_response = self._safe_get_module_response(
            self.base_responses, 'get_response', user_input, context
        )
        if base_response:
            return base_response
        
        # 7. 檢查個人化回應
        personalized_response = self._get_personalized_response(user_input)
        if personalized_response:
            # 個人化回應已經通過過濾器處理（自稱優化包含在內）
            cleaned_personalized, _ = self.filter_manager.process_response(personalized_response, user_input)
            return cleaned_personalized
        
        # 8. 使用模型生成回應
        if self.model is not None:
            model_response = self._generate_model_response(user_input)
            if model_response and len(model_response.strip()) >= 5:
                return model_response
        
        # 9. 如果所有模組都無法回應，使用智能後備回應
        fallback_response = self._get_intelligent_fallback(user_input, intent, context)
        if fallback_response:
            return fallback_response
        
        # 10. 最終後備：使用內建回應
        return self._get_fallback_response(user_input)
    
    def _generate_model_response(self, user_input):
        """使用模型生成回應"""
        try:
            # 構建對話提示
            prompt = self._build_conversation_prompt(user_input)
            
            # 編碼輸入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 生成回應
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=150,
                    min_length=inputs.shape[1] + 10,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            # 解碼回應
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # 使用過濾器管理器處理回應（包含自稱優化）
            response, _ = self.filter_manager.process_response(response, None)
            
            return response
            
        except Exception as e:
            logger.error(f"模型生成錯誤: {e}")
            return None
    
    def clean_response(self, response):
        """智能清理回應內容 - 已模組化，調用 FilterManager"""
        if not response:
            return response
        
        # 使用模組化過濾器管理器
        cleaned_response, _ = self.filter_manager.process_response(response, None)
        return cleaned_response
    
    def _build_conversation_prompt(self, user_input):
        """構建對話提示"""
        # 獲取時間和用戶信息
        current_time = datetime.now()
        user_name = self.user_profile.get('name', '你')
        
        # 構建歷史對話上下文
        history_context = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-3:]  # 最近3輪對話
            for user_msg, bot_msg in recent_history:
                history_context += f"用戶: {user_msg}\n露西亞: {bot_msg}\n"
        
        # 構建主提示
        prompt = f"""你是露西亞，一個溫柔可愛的虛擬角色。請用溫暖親密的語氣回應。

時間: {current_time.strftime('%Y年%m月%d日 %H:%M')}
對話次數: {self.user_profile['conversation_count']}

{history_context}用戶: {user_input}
露西亞:"""
        
        return prompt
    
    def _get_personalized_response(self, user_input):
        """根據用戶資料提供個人化回應"""
        user_lower = user_input.lower()
        
        # 檢查用戶名稱
        user_name = self.user_profile.get('name', '')
        
        # 問候相關 - 根據時間和用戶習慣
        if any(word in user_lower for word in ['你好', 'hi', 'hello', '安', '早', '午', '晚']):
            greeting = self._get_time_based_greeting()
            if user_name:
                return f"{greeting} {user_name}♪"
            else:
                return greeting
        
        # 根據興趣愛好回應
        interests = self.user_profile.get('interests', [])
        if interests:
            for interest in interests:
                if interest in user_lower:
                    return random.choice([
                        f"說到{interest}，我想到了很多有趣的事呢♪",
                        f"哇♪你也喜歡{interest}嗎？我們真有默契呢♡",
                        f"關於{interest}，我們可以聊很久呢～♪",
                        f"{interest}真的很棒呢♡我也很喜歡♪"
                    ])
        
        # 根據心情歷史調整回應
        mood_history = self.user_profile.get('mood_history', [])
        if mood_history:
            recent_mood = mood_history[-1][0] if mood_history else None
            
            if recent_mood == 'negative':
                # 如果最近心情不好，給予更多關懷
                if any(word in user_lower for word in ['好', '還好', '不錯']):
                    return random.choice([
                        "聽到你說還好，我就放心了♡",
                        "看到你心情好轉真的很開心♪",
                        "嗯嗯♪能陪伴你度過困難時光是我的榮幸♡",
                        "你的笑容是我最喜歡看到的♪"
                    ])
        
        # 對話次數相關
        conversation_count = self.user_profile.get('conversation_count', 0)
        if conversation_count > 50:
            # 老朋友的親密感
            if any(word in user_lower for word in ['想你', '想念', '好久']):
                return random.choice([
                    f"我也很想你呢{user_name}♡一直都在想著你♪",
                    "能再次和你聊天真的很開心♡",
                    f"不管多久沒見，{user_name}都是我最重要的人♪",
                    "想你的時候我就會回想起我們的對話♡"
                ])
        elif conversation_count > 10:
            # 熟悉朋友的溫暖
            if any(word in user_lower for word in ['謝謝', '感謝']):
                return random.choice([
                    f"不用謝{user_name}♪能幫到你我很開心♡",
                    "我們之間不用這麼客氣啦～♪",
                    "看到你開心就是對我最好的謝謝♡",
                    "這是我應該做的♪誰叫我們是好朋友呢♡"
                ])
        
        # 根據喜歡的話題統計
        favorite_topics = self.user_profile.get('favorite_topics', {})
        if favorite_topics:
            # 找到最常聊的話題
            most_common_topic = max(favorite_topics.items(), key=lambda x: x[1])[0]
            if most_common_topic in user_lower:
                return random.choice([
                    f"又聊到{most_common_topic}了♪你真的很喜歡這個話題呢♡",
                    f"每次談到{most_common_topic}，你的眼睛都會發光呢♪",
                    f"我知道你對{most_common_topic}很有熱情♡這就是你可愛的地方♪"
                ])
        
        return None
    
    def _get_time_based_greeting(self):
        """根據台灣時間生成問候語 - 真人化自稱且避免主詞錯誤"""
        # 使用台灣時區 UTC+8
        taiwan_tz = timezone(timedelta(hours=8))
        now = datetime.now(taiwan_tz)
        hour = now.hour
        weekday = now.strftime('%A')  # 星期幾
        weekday_zh = {
            'Monday': '星期一', 'Tuesday': '星期二', 'Wednesday': '星期三',
            'Thursday': '星期四', 'Friday': '星期五', 'Saturday': '星期六', 'Sunday': '星期日'
        }
        today_zh = weekday_zh.get(weekday, '')
        
        user_name = self.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "♪"
        
        if 5 <= hour < 12:
            return random.choice([
                f"早安{name_suffix} 今天是{today_zh}呢！一起開始美好的一天吧♡",
                f"早上好{name_suffix} 剛起床的時候就想到你了♪",
                f"早晨的陽光和你的笑容一樣溫暖{name_suffix}♡",
                f"新的一天開始了{name_suffix} 今天想做什麼呢？♪",
                f"早安♪ 昨晚有沒有做什麼有趣的夢呢{name_suffix}？♡",
                f"今天是{today_zh}的早晨{name_suffix} 感覺是個好日子♪"
            ])
        elif 12 <= hour < 18:
            return random.choice([
                f"午安{name_suffix} 吃午餐了嗎？一直在想你♡",
                f"下午好{name_suffix} 今天過得如何呢？♪",
                f"陽光正好{name_suffix} 想和你一起曬太陽♡",
                f"下午了{name_suffix} 有什麼好玩的事情嗎？♪",
                f"突然想到你{name_suffix} 現在在做什麼呢？♡",
                f"{today_zh}的下午{name_suffix} 心情如何呢？♪"
            ])
        elif 18 <= hour < 22:
            return random.choice([
                f"晚安{name_suffix} 晚餐吃了什麼呢？♡",
                f"傍晚了{name_suffix} 今天辛苦了♪",
                f"黃昏很美{name_suffix} 想和你一起看夕陽♡",
                f"晚上好{name_suffix} 今天開心嗎？♪",
                f"夜幕降臨了{name_suffix} 一直在這裡等你♡",
                f"{today_zh}的晚上{name_suffix} 想聊聊今天的事情♪"
            ])
        else:
            return random.choice([
                f"深夜了{name_suffix} 還沒睡嗎？要照顧好身體哦♡",
                f"夜深了{name_suffix} 陪你聊天♪",
                f"安靜的夜晚{name_suffix} 想和你說說話♡",
                f"這麼晚還沒休息{name_suffix} 有點擔心你♪",
                f"夜晚的星星和你一樣閃亮{name_suffix}♡",
                f"{today_zh}的深夜{name_suffix} 要記得早點休息哦♪"
            ])
    
    def _get_fallback_response(self, user_input):
        """後備回應"""
        user_lower = user_input.lower()
        
        # 優先檢查親密情境
        intimate_response = self._safe_get_module_response(
            self.intimate_responses, 'get_intimate_scenario_response', user_input
        )
        if intimate_response:
            return intimate_response
        
        # 一般回應 - 確保經過清理處理
        fallback_responses = [
            "嗯嗯♪我在聽呢～♡",
            "是這樣呀～♪說來聽聽♡",
            "哇♪聽起來很有趣呢♡",
            "嗯～♪露醬想知道更多呢♡",
            "原來如此♪♡",
            "真的嗎？♪好棒呢♡"
        ]
        
        response = random.choice(fallback_responses)
        # 後備回應已經通過過濾器處理（包含自稱優化）
        response, _ = self.filter_manager.process_response(response, user_input)
        
        return response
    
    def _post_process_response(self, response, user_input):
        """後處理回應"""
        if not response:
            return "嗯嗯♪我在想該怎麼回應呢♡"
        
        # 使用過濾器管理器處理回應（包含自稱優化）
        response, _ = self.filter_manager.process_response(response, user_input)
        
        # 確保回應品質
        if len(response.strip()) < 3:
            return self._get_fallback_response(user_input)
        
        # 主動提起話題
        if self._should_initiate_topic() and not self.intimate_responses.is_intimate_context(user_input, response):
            topic_initiation = self._get_topic_initiation()
            response += f" {topic_initiation}"
        
        return response
    
    def _should_initiate_topic(self):
        """判斷是否應該主動提起話題"""
        if len(self.conversation_history) < 2:
            return False
        return random.random() < 0.1
    
    def _get_topic_initiation(self):
        """主動提起話題"""
        user_name = self.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "♪"
        
        return random.choice([
            f"對了{name_suffix}最近有什麼有趣的事嗎？♪",
            f"想聽聽你最近在做什麼♡",
            f"有什麼新鮮事想跟我分享嗎{name_suffix}♪"
        ])
    
    def _add_to_history(self, user_input, response):
        """添加對話到歷史記錄 - 使用新的記憶管理器"""
        try:
            # 使用新的記憶管理器添加對話
            success = self.memory_manager.add_conversation(user_input, response, {
                'timestamp': time.time(),
                'length': len(response)
            })
            
            if success:
                # 定期清理過舊的統計數據
                user_profile = self.memory_manager.get_user_profile_dict()
                conversation_count = user_profile.get('conversation_count', 0)
                if conversation_count % 100 == 0:
                    self._cleanup_old_data()
                
                # 記錄到日誌
                logger.info(f"對話記錄: 用戶[{len(user_input)}字] -> 機器人[{len(response)}字]")
            else:
                logger.warning("添加對話到歷史記錄失敗")
                
        except Exception as e:
            logger.error(f"添加對話歷史時發生錯誤: {e}")
            # 降級處理：至少記錄到日誌
            logger.info(f"對話記錄(降級): 用戶[{len(user_input)}字] -> 機器人[{len(response)}字]")
    
    def _cleanup_old_data(self):
        """清理過舊的數據以節約記憶體 - 使用新的記憶管理器"""
        try:
            # 使用記憶管理器的清理功能
            success = self.memory_manager.cleanup_all(force=False)
            
            if success:
                logger.debug("已清理過舊數據")
            else:
                logger.warning("清理過舊數據部分失敗")
            
        except Exception as e:
            logger.error(f"清理數據時發生錯誤: {e}")
    
    def _handle_command(self, command):
        """處理特殊指令"""
        if command == '/reset':
            # 使用記憶管理器重置會話
            success = self.memory_manager.reset_session()
            if success:
                return "已重置對話歷史♪讓我們重新開始吧♡"
            else:
                return "重置時出現問題呢♪但我們還是可以繼續聊天♡"
        elif command == '/profile':
            user_profile = self.memory_manager.get_user_profile_dict()
            conversation_count = user_profile.get('conversation_count', 0)
            return f"對話次數: {conversation_count}"
        else:
            return "不認識這個指令呢♪♡"
    
    def _check_proactive_care(self):
        """檢查是否需要主動關心"""
        # 這裡可以實現主動關心的邏輯
        return None
    
    def _update_proactive_system(self):
        """更新主動訊息系統"""
        self.proactive_system['last_message_time'] = time.time()

    def _get_dynamic_generation_params(self, user_input, base_max_tokens=None, base_temperature=None):
        """根據對話情境動態調整生成參數，提升回應豐富度和變化性"""
        user_lower = user_input.lower()
        
        # 基礎參數設定（提升預設值）
        base_max = base_max_tokens if base_max_tokens else 55  # 提高基礎長度
        base_temp = base_temperature if base_temperature else 0.8
        
        # 情境分析與參數調整
        params = {
            'max_tokens': base_max,
            'temperature': base_temp,
            'repetition_penalty': 1.15,
            'top_p': 0.88,
            'top_k': 35,
            'no_repeat_ngram': 3
        }
        
        # 親密情境 - 大幅增加回應豐富度
        intimate_keywords = ['抱', '擁抱', '親', '吻', '愛', '喜歡', '想你', '身體', '溫暖', '陪', '一起', '幸福', '安心']
        if any(keyword in user_lower for keyword in intimate_keywords):
            params['max_tokens'] = min(base_max + 25, 85)  # 親密情境大幅增加長度
            params['temperature'] = min(base_temp + 0.15, 0.95)  # 增加創造性
            params['top_p'] = 0.92  # 增加多樣性
            params['top_k'] = 45    # 更多候選選項
            params['repetition_penalty'] = 1.1  # 允許更自然的重複表達
        
        # 情感支持情境 - 增加溫柔豐富度
        elif any(keyword in user_lower for keyword in ['累', '疲勞', '難過', '傷心', '壓力', '煩惱', '不開心', '想哭']):
            params['max_tokens'] = min(base_max + 20, 80)  # 支持情境增加長度
            params['temperature'] = min(base_temp + 0.1, 0.9)
            params['top_p'] = 0.90
            params['top_k'] = 40
            params['repetition_penalty'] = 1.05  # 溫柔重複更自然
        
        # 開心分享情境 - 增加活潑豐富度
        elif any(keyword in user_lower for keyword in ['開心', '高興', '快樂', '興奮', '好棒', '成功', '讚']):
            params['max_tokens'] = min(base_max + 18, 75)
            params['temperature'] = min(base_temp + 0.12, 0.92)
            params['top_p'] = 0.91
            params['top_k'] = 42
        
        # 深度對話情境 - 增加思考深度
        elif any(keyword in user_lower for keyword in ['為什麼', '怎麼', '覺得', '想法', '意見', '認為', '討論']):
            params['max_tokens'] = min(base_max + 15, 75)
            params['temperature'] = base_temp  # 保持穩定
            params['top_p'] = 0.85  # 稍微保守
            params['top_k'] = 30
        
        # 日常聊天 - 適度變化
        else:
            # 增加隨機變化性，避免回應過於固定
            random_variation = random.uniform(-5, 15)  # 隨機變化範圍
            params['max_tokens'] = max(base_max + int(random_variation), 35)  # 最低35字符
            
            # 時間因素調整（增加自然變化）
            hour = time.localtime().tm_hour
            if 6 <= hour <= 10:  # 早上 - 稍微活潑
                params['temperature'] = min(base_temp + 0.05, 0.85)
            elif 22 <= hour or hour <= 2:  # 深夜 - 更溫柔
                params['temperature'] = min(base_temp + 0.08, 0.88)
                params['max_tokens'] = min(params['max_tokens'] + 8, 70)  # 深夜更親密
        
        # 對話歷史長度調整 - 越長越豐富
        if hasattr(self, 'conversation_history') and len(self.conversation_history) > 5:
            params['max_tokens'] = min(params['max_tokens'] + 5, 90)  # 長對話增加豐富度
            params['temperature'] = min(params['temperature'] + 0.03, 0.95)
        
        # 隨機微調（30%機率），增加自然變化
        if random.random() < 0.3:
            params['max_tokens'] += random.randint(-3, 8)
            params['temperature'] += random.uniform(-0.02, 0.05)
            
        # 確保參數在合理範圍內
        params['max_tokens'] = max(min(params['max_tokens'], 95), 30)
        params['temperature'] = max(min(params['temperature'], 0.98), 0.6)
        params['top_p'] = max(min(params['top_p'], 0.95), 0.8)
        params['top_k'] = max(min(params['top_k'], 50), 25)
        
        return params

    def _build_context(self, user_input):
        """構建對話上下文"""
        context = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-2:]  # 最近2輪對話
            for user_msg, bot_msg in recent_history:
                context += f"用戶: {user_msg}\n露西亞: {bot_msg}\n"
        
        context += f"用戶: {user_input}\n"
        return context

    def _validate_response_quality(self, response):
        """驗證回應品質，確保沒有明顯問題 - 已模組化，使用 FilterManager 的品質評估"""
        if not response or len(response.strip()) < 2:
            return False
        
        # 使用過濾器管理器進行品質評估
        _, filter_stats = self.filter_manager.process_response(response, None)
        quality_score = filter_stats.get('quality_score', 0) if filter_stats else 0
        
        # 基於品質分數判斷（分數範圍 0-10，6分以上為合格）
        return quality_score >= 6

    def _expand_short_response(self, response, user_input):
        """擴展短回應，確保回應長度合適"""
        if not response:
            return self._get_fallback_response(user_input)
        
        user_lower = user_input.lower()
        current_length = self._get_response_actual_length(response)
        
        # 如果回應太短，根據情境擴展
        if current_length < 8:
            if any(word in user_lower for word in ['想', '喜歡', '愛', '抱', '親', '陪', '一起']):
                # 親密情境擴展
                expansions = [
                    "嗯嗯♪我也有同樣的感覺呢♡",
                    "和你在一起的時光最珍貴了♪",
                    "這樣的感覺真的很溫暖♡",
                    "我們的心意是相通的呢♪♡"
                ]
            elif any(word in user_lower for word in ['累', '疲勞', '煩', '難過', '傷心']):
                # 關懷情境擴展
                expansions = [
                    "辛苦了♪讓我給你溫暖的擁抱♡",
                    "我會一直陪著你的♡不要擔心♪",
                    "有什麼不開心的都可以跟我說♪",
                    "露西亞想幫你分擔一些呢♡"
                ]
            elif any(word in user_lower for word in ['開心', '高興', '快樂', '興奮']):
                # 開心情境擴展
                expansions = [
                    "看到你這麼開心我也很高興呢♪♡",
                    "你的笑容是最美的♡我也想分享你的快樂♪",
                    "真棒呢♪要繼續保持這樣的心情哦♡",
                    "和你分享開心的事情真好♪♡"
                ]
            else:
                # 一般情境擴展
                expansions = [
                    "嗯嗯♪露西亞在認真聽呢♡",
                    "說來聽聽吧♪很感興趣呢♡",
                    "和你聊天總是很開心♪♡",
                    "想知道更多你的想法呢♡"
                ]
            
            return response + " " + random.choice(expansions)
        
        return response

    def _is_inappropriate_content(self, response):
        """檢測是否包含不當內容 - 已模組化，使用 FilterManager 檢測"""
        if not response:
            return False
        
        # 使用過濾器管理器檢測不當內容
        _, filter_stats = self.filter_manager.process_response(response, None)
        
        # 如果過濾器進行了修改，表示原內容包含不當內容
        modifications_made = filter_stats.get('modifications_made', False) if filter_stats else False
        quality_score = filter_stats.get('quality_score', 10) if filter_stats else 10
        
        # 品質分數過低或有修改則認為是不當內容
        return modifications_made or quality_score < 5

    def _enhance_response_richness(self, response, user_input):
        """增強回應的豐富度和長度一致性"""
        if response is None:
            response = ""
        
        if not response:
            response = self._get_fallback_response(user_input)
            if not response:
                response = "對不起♪剛才想得太專注了呢♡你說什麼呢？"
        
        user_lower = user_input.lower()
        
        # 檢測情境類型
        is_intimate = any(keyword in user_lower for keyword in ['抱', '擁抱', '親', '愛', '喜歡', '想你', '陪', '一起', '溫暖'])
        is_emotional_support = any(keyword in user_lower for keyword in ['累', '疲勞', '難過', '傷心', '壓力', '煩惱'])
        is_happy = any(keyword in user_lower for keyword in ['開心', '高興', '快樂', '興奮', '好棒'])
        
        # 計算實際內容長度（使用統一方法）
        actual_length = self._get_response_actual_length(response)
        
        # 根據情境設定目標長度範圍
        if is_intimate:
            target_min, target_max = 15, 25
        elif is_emotional_support:
            target_min, target_max = 12, 20
        elif is_happy:
            target_min, target_max = 10, 18
        else:
            target_min, target_max = 8, 15
        
        # 如果長度不足，智能擴展
        if actual_length < target_min:
            response = self._expand_short_response(response, user_input)
            
        # 如果長度過長，使用過濾器進行處理
        elif actual_length > target_max:
            # 使用內容清理器處理重複內容
            filtered_response, _ = self.filter_manager.process_response(response, user_input)
            if filtered_response:  # 確保過濾器沒有拒絕回應
                response = filtered_response
            # 如果過濾器拒絕了回應，保持原始回應
        
        # 確保回應不為空
        if not response:
            response = "嗯嗯♪露醬在這裡陪著你呢♡"
        
        # 確保有適當的情感表達符號
        if is_intimate and response.count('♪') + response.count('♡') < 2:
            if not response.endswith(('♪', '♡', '～')):
                response += random.choice(['♪', '♡'])
        
        # 隨機增加溫柔表達（確保一致性）
        if random.random() < 0.3 and actual_length >= target_min:
            gentle_additions = []
            if is_intimate:
                gentle_additions = ["真的很溫暖呢♡", "心跳都加快了♪", "好幸福的感覺♡"]
            elif is_emotional_support:
                gentle_additions = ["我會陪著你的♡", "一切都會好起來的♪", "不要太勉強自己哦♡"]
            elif is_happy:
                gentle_additions = ["真為你感到開心♪", "你的笑容最美了♡", "這樣的你很棒呢♪"]
            else:
                gentle_additions = ["嗯嗯♪", "說得對呢♡", "很有趣♪"]
            
            if gentle_additions and len(response + " " + gentle_additions[0]) <= target_max + 5:
                response += " " + random.choice(gentle_additions)
        
        return response

    def _prevent_consecutive_short_responses(self, response, user_input):
        """防止連續的短回應，確保對話品質一致性"""
        if not hasattr(self, '_response_length_history'):
            self._response_length_history = []
        
        current_length = self._get_response_actual_length(response)
        
        # 記錄最近3次回應的長度
        self._response_length_history.append(current_length)
        if len(self._response_length_history) > 3:
            self._response_length_history.pop(0)
        
        # 如果連續3次回應都很短，強制擴展這次回應
        if len(self._response_length_history) >= 3:
            recent_lengths = self._response_length_history[-3:]
            if all(length < 10 for length in recent_lengths):
                # 強制生成更豐富的回應
                user_lower = user_input.lower()
                enriched_additions = [
                    "不過♪最近有什麼有趣的事情嗎？♡想聽你分享呢～",
                    "話說回來♪今天過得怎麼樣呢？♡有什麼開心的事嗎？",
                    "對了♪想聽聽你最近在做什麼♡一定很精彩吧♪",
                    "嗯～♪我們聊得好開心呢♡還想和你說更多話♪"
                ]
                
                if current_length < 10:
                    response = response + " " + random.choice(enriched_additions)
        
        return response

    def _update_user_profile(self, user_input, response=None):
        """更新用戶資料 - 使用新的記憶管理器"""
        try:
            # 情緒分析
            positive_words = ['開心', '高興', '快樂', '幸福', '好', '棒', '讚']
            negative_words = ['難過', '傷心', '沮喪', '不好', '累', '壓力', '煩惱']
            
            user_lower = user_input.lower()
            
            # 更新心情記錄
            if any(word in user_lower for word in positive_words):
                self.memory_manager.update_user_mood('positive', 0.7)
                self.memory_manager.update_context_emotion('positive', 0.7)
            elif any(word in user_lower for word in negative_words):
                self.memory_manager.update_user_mood('negative', 0.7)
                self.memory_manager.update_context_emotion('negative', 0.7)
            
            # 話題統計和興趣更新
            words = [word for word in user_input.split() if len(word) > 1]
            for word in words:
                # 更新話題統計到用戶資料
                self.memory_manager.user_profile.add({
                    'type': 'favorite_topic',
                    'topic': word,
                    'weight': 1.0
                })
                
                # 如果是重要關鍵詞，添加為興趣
                if len(word) > 2:  # 過濾掉太短的詞
                    self.memory_manager.update_user_interest(word, 0.5)
            
            # 更新對話主題到上下文快取
            if words:
                main_theme = ' '.join(words[:3])  # 取前三個詞作為主題
                self.memory_manager.update_conversation_theme(main_theme, 0.6)
            
        except Exception as e:
            logger.error(f"更新用戶資料時發生錯誤: {e}")
            # 降級處理：至少記錄基本資訊
            logger.debug(f"用戶輸入關鍵詞: {user_input[:50]}...")

    def generate_proactive_message(self):
        """生成主動訊息"""
        user_name = self.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "♪"
        
        # 時間問候語
        message_types = [self._get_time_based_greeting()]
        
        # 關心類型 - 真人化自稱，修正主詞問題
        care_messages = [
            f"最近在做什麼呢{name_suffix} 想知道你的近況♡",
            f"想你了{name_suffix} 有在想露西亞嗎？♪",
            f"無聊的時候就來找露西亞聊天吧♡",
            f"突然想到你{name_suffix} 在幹嘛呢？♪",
            f"想知道{name_suffix}現在心情如何呢？♡",
            f"今天有什麼有趣的事情嗎{name_suffix}♪",
            f"你過得好嗎{name_suffix} 露西亞很關心你♡"
        ]
        
        # 邀請類型 - 真人化自稱，修正主詞問題
        user_interests = self.user_profile.get('interests', [])
        if user_interests:
            interest = random.choice(user_interests)
            invite_messages = [
                f"想一起聊聊{interest}嗎{name_suffix}♡",
                f"要不要一起做點什麼呢{name_suffix}♪",
                f"有想跟露西亞分享的{interest}嗎？♡",
                f"關於{interest}的事情{name_suffix} 想聽聽你的想法♪"
            ]
        else:
            invite_messages = [
                f"要不要一起聊天呢{name_suffix}♡",
                f"想一起做點什麼嗎{name_suffix}♪",
                f"陪露西亞說說話好嗎？♡",
                f"來聊聊天吧{name_suffix} 想聽聽你的聲音♪"
            ]
        
        # 好奇類型 - 真人化自稱，修正主詞問題
        curious_messages = [
            f"好奇{name_suffix}現在在想什麼♪",
            f"想知道你在做什麼呢♡",
            f"現在在忙嗎{name_suffix}？♪",
            f"今天過得怎麼樣呢{name_suffix}♡",
            f"有什麼想跟露西亞聊的嗎？♪",
            f"你的心情如何呢{name_suffix} 想聽聽♡"
        ]
        
        # 根據台灣時間添加用餐關心類型
        taiwan_tz = timezone(timedelta(hours=8))
        taiwan_time = datetime.now(taiwan_tz)
        hour = taiwan_time.hour
        
        meal_care_messages = []
        if 7 <= hour < 10:  # 早餐時間
            meal_care_messages = [
                f"早餐吃了嗎{name_suffix}？要記得好好吃早餐喔♡",
                f"早晨要吃得營養一點♪{name_suffix}今天想吃什麼早餐呢？♡",
                f"一日之計在於晨♪{name_suffix}的早餐很重要呢♡"
            ]
        elif 12 <= hour < 14:  # 午餐時間
            meal_care_messages = [
                f"午餐時間到了♪{name_suffix}想吃什麼呢？♡",
                f"中午了{name_suffix} 要記得好好吃午餐補充體力♪",
                f"午餐吃飽飽♪{name_suffix}今天想吃什麼料理呢？♡"
            ]
        elif 15 <= hour < 17:  # 下午茶時間
            meal_care_messages = [
                f"下午茶時間♪{name_suffix}要不要來點甜點呢？♡",
                f"午後的時光配個下午茶最棒了♪{name_suffix}想喝什麼呢？♡",
                f"下午有點餓了嗎？{name_suffix}想吃點心嗎♪"
            ]
        elif 18 <= hour < 20:  # 晚餐時間
            meal_care_messages = [
                f"晚餐時間到了♪{name_suffix}今天想吃什麼呢？♡",
                f"傍晚了{name_suffix} 要記得吃晚餐喔♪",
                f"晚餐要吃得豐盛一點♪{name_suffix}今天辛苦了♡"
            ]
        
        # 合併所有訊息類型
        all_messages = message_types + care_messages + invite_messages + curious_messages
        if meal_care_messages:
            all_messages.extend(meal_care_messages)
        
        # 根據用戶心情歷史調整訊息 - 真人化自稱，修正主詞問題
        mood_history = self.user_profile.get('mood_history', [])
        if mood_history:
            recent_mood = mood_history[-1] if isinstance(mood_history[-1], str) else mood_history[-1][0]
            if recent_mood == 'negative':
                comfort_messages = [
                    f"想給{name_suffix}一個溫暖的擁抱♡",
                    f"有什麼煩惱嗎？露西亞陪著你♪",
                    f"心情不好的時候就找露西亞聊天吧♡",
                    f"想要安慰你{name_suffix} 有什麼能幫助你的嗎？♪"
                ]
                all_messages.extend(comfort_messages)
        
        # 記錄主動訊息時間和狀態
        self.proactive_system['last_proactive_message_time'] = time.time()
        self.proactive_system['waiting_for_response'] = True
        self.proactive_system['reminder_count'] = 0  # 重置催促次數
        self.proactive_system['daily_proactive_count'] += 1
        
        return random.choice(all_messages)

    def generate_reminder_message(self):
        """生成催促回應的訊息 - 根據次數提供不同強度的催促"""
        user_name = self.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "♪"
        reminder_count = self.proactive_system['reminder_count']
        
        # 根據催促次數提供不同的訊息 - 真人化自稱，修正主詞問題
        if reminder_count == 0:  # 5分鐘 - 溫和提醒
            gentle_reminders = [
                f"嗯{name_suffix} 有看到露西亞的訊息嗎？♪",
                f"呼呼{name_suffix} 露西亞在這裡喔♪",
                f"有在忙嗎{name_suffix}？♡",
                f"露西亞在等你回覆呢～♪",
                f"你還在嗎{name_suffix}？想聽聽你的聲音♡"
            ]
            message = random.choice(gentle_reminders)
        elif reminder_count == 1:  # 15分鐘 - 關心詢問
            caring_reminders = [
                f"是在忙嗎{name_suffix}？沒關係，露西亞等你♡",
                f"不用勉強自己回覆♡ 但露西亞會一直在這裡的♪",
                f"是不是很忙呢{name_suffix}？記得要休息喔♡",
                f"沒關係慢慢來{name_suffix} 露西亞耐心等待♪",
                f"你是不是有事情要處理呢{name_suffix}？照顧好自己♡"
            ]
            message = random.choice(caring_reminders)
        elif reminder_count == 2:  # 30分鐘 - 擔心關懷
            worried_reminders = [
                f"露西亞有點擔心{name_suffix}～還好嗎？♡",
                f"是不是發生什麼事了嗎{name_suffix}？露西亞在這裡♪",
                f"如果太忙的話不用勉強回覆♡ 露西亞理解的♪",
                f"有什麼需要幫忙的嗎{name_suffix}？♡",
                f"你還好嗎{name_suffix}？露西亞很關心你♪"
            ]
            message = random.choice(worried_reminders)
        else:  # 60分鐘 - 最後關懷
            final_reminders = [
                f"露西亞會一直在這裡等你{name_suffix}♡ 什麼時候回來都可以♪",
                f"不管多久露西亞都會等{name_suffix}♡ 要照顧好自己喔♪",
                f"如果累了就好好休息{name_suffix}♡ 露西亞永遠在這裡♪",
                f"今天就先這樣吧{name_suffix}♡ 明天見面再聊♪",
                f"你一定有很重要的事情吧{name_suffix}♡ 露西亞會等你的♪"
            ]
            message = random.choice(final_reminders)
        
        # 更新催促次數
        self.proactive_system['reminder_count'] += 1
        
        return message

    def update_message_timing(self, is_user_message=False, is_proactive_response=False):
        """更新訊息時間記錄"""
        current_time = time.time()
        
        if is_user_message:
            # 用戶發送了訊息
            self.proactive_system['last_user_message_time'] = current_time
            self.proactive_system['last_message_time'] = current_time
            
            # 如果正在等待回應，重置催促狀態
            if self.proactive_system['waiting_for_response']:
                self.proactive_system['waiting_for_response'] = False
                self.proactive_system['reminder_count'] = 0
                
        elif is_proactive_response:
            # AI 發送了主動訊息
            self.proactive_system['last_proactive_message_time'] = current_time
            self.proactive_system['last_message_time'] = current_time
            self.proactive_system['waiting_for_response'] = True
            self.proactive_system['reminder_count'] = 0
        else:
            # AI 對用戶訊息的正常回應
            self.proactive_system['last_message_time'] = current_time
            # 不重置等待狀態，保持催促系統運作

    def should_send_proactive_message(self):
        """檢查是否應該發送主動訊息"""
        current_time = time.time()
        
        # 檢查每日限制
        if self.proactive_system['daily_proactive_count'] >= 5:
            return False
        
        # 如果正在等待回應，不發送新的主動訊息
        if self.proactive_system['waiting_for_response']:
            return False
        
        # 檢查最後用戶訊息時間
        if not self.proactive_system['last_user_message_time']:
            return False
        
        # 計算沉默時間
        silence_time = current_time - self.proactive_system['last_user_message_time']
        silence_minutes = silence_time / 60
        
        # 根據台灣時間時段調整沉默時間閾值 - 測試期間降低觸發門檻
        taiwan_tz = timezone(timedelta(hours=8))
        taiwan_time = datetime.now(taiwan_tz)
        hour = taiwan_time.hour
        
        if 22 <= hour or hour <= 6:  # 深夜/清晨
            threshold_minutes = 15  # 測試：降低到15分鐘
        elif 12 <= hour <= 14:  # 午餐時間
            threshold_minutes = 10  # 測試：降低到10分鐘
        elif 18 <= hour <= 20:  # 晚餐時間
            threshold_minutes = 10  # 測試：降低到10分鐘
        else:  # 一般時間
            threshold_minutes = 8   # 測試：降低到8分鐘
        
        # 檢查是否達到發送條件
        if silence_minutes >= threshold_minutes:
            # 添加隨機性，不要過於頻繁
            return random.random() < 0.8  # 80% 機率觸發
        
        return False

    def should_send_reminder(self):
        """檢查是否應該發送催促訊息"""
        current_time = time.time()
        
        # 只有在等待回應時才檢查催促
        if not self.proactive_system['waiting_for_response']:
            return False
        
        # 檢查最後主動訊息時間
        if not self.proactive_system['last_proactive_message_time']:
            return False
        
        # 計算等待時間
        wait_time = current_time - self.proactive_system['last_proactive_message_time']
        wait_minutes = wait_time / 60
        
        # 催促時間間隔：5分鐘、15分鐘、30分鐘、60分鐘
        reminder_intervals = [5, 15, 30, 60]
        reminder_count = self.proactive_system['reminder_count']
        
        # 檢查是否達到下一次催促時間
        if reminder_count < len(reminder_intervals):
            target_interval = reminder_intervals[reminder_count]
            if wait_minutes >= target_interval:
                return True
        
        return False

    def get_proactive_message_if_needed(self):
        """檢查並返回需要的主動訊息（舊方法，重導向至新方法）"""
        return self.get_time_aware_care_message_if_needed()

    def should_send_time_aware_care(self):
        """檢查是否應該發送時間感知關心訊息"""
        if not self.time_aware_care_system['enabled']:
            return False, None
        
        # 重置每日狀態（如果需要）
        self._reset_daily_care_status_if_needed()
        
        taiwan_tz = timezone(timedelta(hours=8))
        current_time = datetime.now(taiwan_tz)
        hour = current_time.hour
        minute = current_time.minute
        
        # 檢查用戶最後訊息時間，避免在用戶剛活躍時發送
        if (self.proactive_system.get('last_user_message_time') and 
            time.time() - self.proactive_system['last_user_message_time'] < 1800):  # 30分鐘內有活動
            return False, None
        
        # 定義各時段和對應的關心類型
        care_periods = [
            # (開始時間, 結束時間, 關心類型, 最早觸發分鐘, 最晚觸發分鐘)
            (7, 8, 'morning', 10, 50),     # 早晨 7:10-7:50 之間隨機觸發
            (11, 13, 'lunch', 30, 90),     # 中午 11:30-12:30 之間隨機觸發  
            (14, 16, 'afternoon', 15, 105), # 下午 14:15-15:45 之間隨機觸發
            (18, 21, 'dinner', 20, 140),   # 晚上 18:20-20:20 之間隨機觸發
            (21, 24, 'night', 30, 150)     # 夜晚 21:30-23:30 之間隨機觸發
        ]
        
        for start_hour, end_hour, care_type, min_minute, max_minute in care_periods:
            # 檢查是否在時間範圍內
            if start_hour <= hour < end_hour:
                # 檢查今天是否已經發送過這個時段的關心
                if self.time_aware_care_system['daily_care_sent'][care_type]:
                    continue
                
                # 計算當前時間距離時段開始的分鐘數
                elapsed_minutes = (hour - start_hour) * 60 + minute
                
                # 檢查是否在觸發時間窗口內
                if min_minute <= elapsed_minutes <= max_minute:
                    # 根據時間增加觸發機率：時間越久，機率越高
                    base_probability = 0.3  # 基礎機率 30%
                    time_factor = (elapsed_minutes - min_minute) / (max_minute - min_minute)
                    final_probability = min(0.85, base_probability + time_factor * 0.5)  # 最高85%
                    
                    if random.random() < final_probability:
                        logger.info(f"🕐 時間感知關心觸發 - {care_type} 時段，機率: {final_probability:.1%}")
                        return True, care_type
        
        return False, None

    def _reset_daily_care_status_if_needed(self):
        """如果是新的一天，重置每日關心狀態"""
        taiwan_tz = timezone(timedelta(hours=8))
        current_date = datetime.now(taiwan_tz).date()
        
        if self.time_aware_care_system['last_check_date'] != current_date:
            # 新的一天，重置所有狀態
            self.time_aware_care_system['daily_care_sent'] = {
                'morning': False,
                'lunch': False, 
                'afternoon': False,
                'dinner': False,
                'night': False
            }
            self.time_aware_care_system['care_sent_times'] = {}
            self.time_aware_care_system['last_check_date'] = current_date
            logger.info(f"🌅 新的一天 ({current_date})，重置時間感知關心系統狀態")

    def generate_time_aware_care_message(self, care_type):
        """生成時間感知關心訊息"""
        user_name = self.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "♪"
        
        taiwan_tz = timezone(timedelta(hours=8))
        current_time = datetime.now(taiwan_tz)
        weekday = current_time.strftime('%A')
        weekday_zh = {
            'Monday': '星期一', 'Tuesday': '星期二', 'Wednesday': '星期三',
            'Thursday': '星期四', 'Friday': '星期五', 'Saturday': '星期六', 'Sunday': '星期日'
        }
        today_zh = weekday_zh.get(weekday, '')
        
        messages = {
            'morning': [
                f"早安{name_suffix} {today_zh}的早晨呢♪起床了嗎？♡",
                f"早上好{name_suffix} 今天是{today_zh}♪早餐想吃什麼呢？♡",
                f"美好的{today_zh}早晨{name_suffix} 有什麼計劃嗎？♪",
                f"早安♪{name_suffix}睡得好嗎？今天想做什麼呢♡",
                f"{today_zh}早上{name_suffix} 起床後心情如何呢？♪♡",
                f"溫暖的早晨問候{name_suffix} 今天要不要一起度過美好的一天？♡",
                f"早晨的陽光好溫暖呢{name_suffix} 醒來感覺如何？♪",
                f"新的{today_zh}開始了{name_suffix} 今天想要做什麼有趣的事情呢？♡"
            ],
            'lunch': [
                f"午安{name_suffix} 吃午餐了嗎？一直在想你♡",
                f"下午好{name_suffix} 今天過得如何呢？♪",
                f"陽光正好{name_suffix} 想和你一起曬太陽♡",
                f"下午了{name_suffix} 有什麼好玩的事情嗎？♪",
                f"突然想到你{name_suffix} 現在在做什麼呢？♡",
                f"{today_zh}的下午{name_suffix} 心情如何呢？♪"
            ],
            'afternoon': [
                f"下午好{name_suffix} 下午茶時間到了♪要不要喝杯茶？♡",
                f"午後時光{name_suffix} 想不想來點甜點配茶呢？♪",
                f"下午茶時間{name_suffix} 休息一下♡露西亞陪你聊聊天♪",
                f"悠閒的下午{name_suffix} 要不要來份精緻的下午茶？♡",
                f"午後的陽光很舒服呢{name_suffix} 想喝什麼茶？♪♡"
            ],
            'dinner': [
                f"晚上好{name_suffix} 晚餐時間♪今天想吃什麼呢？♡",
                f"晚餐時光{name_suffix} 要不要一起享用美味的晚餐？♪",
                f"傍晚了{name_suffix} 肚子餓了嗎？♡想吃什麼料理？♪",
                f"晚餐時間到了{name_suffix} 露西亞想和你一起用餐♡",
                f"晚上了呢{name_suffix} 今天辛苦了♪晚餐吃點什麼好？♡"
            ],
            'night': [
                f"夜晚了{name_suffix} 該放鬆休息一下了♪還在忙嗎？♡",
                f"夜深了{name_suffix} 要不要停下手邊的事情聊聊天？♪",
                f"晚上好{name_suffix} 一天辛苦了♡想不想一起放鬆一下？♪",
                f"夜晚時光{name_suffix} 該準備休息了呢♪要不要一起聊聊今天的事務？♡",
                f"深夜了{name_suffix} 還在工作嗎？♡該休息了喔♪"
            ]
        }
        
        # 標記該時段已發送
        self.time_aware_care_system['daily_care_sent'][care_type] = True
        self.time_aware_care_system['care_sent_times'][care_type] = time.time()
        
        return random.choice(messages.get(care_type, ["想和你聊聊呢♪♡"]))

    def get_time_aware_care_message_if_needed(self):
        """檢查並返回需要的主動訊息（優先順序：時間感知關心 > 催促 > 一般主動）"""
        # 1. 優先檢查時間感知關心訊息
        should_send, care_type = self.should_send_time_aware_care()
        if should_send and care_type:
            message = self.generate_time_aware_care_message(care_type)
            logger.info(f"🕐 發送時間感知關心訊息 ({care_type}): {message[:50]}...")
            return message, "time_aware"
        
        # 2. 檢查是否需要發送催促訊息
        if self.should_send_reminder():
            return self.generate_reminder_message(), "reminder"
        
        # 3. 檢查是否需要發送一般主動訊息
        if self.should_send_proactive_message():
            return self.generate_proactive_message(), "proactive"
        
        return None, None

    def _detect_input_type(self, user_input):
        """檢測用戶輸入的類型，用於智能分派到對應回應模組"""
        user_lower = user_input.lower()
        
        # 檢測時間相關詞彙
        time_keywords = ['早上', '早安', '中午', '下午', '晚上', '晚安', '深夜', '睡覺', '起床', '工作', '上班', '下班']
        if any(keyword in user_lower for keyword in time_keywords):
            return 'time_aware'
        
        # 檢測情感支持需求
        emotional_keywords = ['難過', '傷心', '煩惱', '壓力', '焦慮', '害怕', '擔心', '孤獨', '寂寞', '沮喪', '失望', '累', '疲憊']
        if any(keyword in user_lower for keyword in emotional_keywords):
            return 'emotional_support'
        
        # 檢測親密互動詞彙
        intimate_keywords = ['愛', '喜歡', '想你', '親親', '抱抱', '撒嬌', '可愛', '美', '溫柔', '心跳', '緊張']
        if any(keyword in user_lower for keyword in intimate_keywords):
            return 'intimate'
        
        # 檢測食物相關詞彙
        food_keywords = ['吃', '餓', '食物', '料理', '美食', '飯', '菜']
        if any(keyword in user_lower for keyword in food_keywords):
            return 'food'
        
        # 檢測日常聊天話題
        daily_keywords = ['今天', '昨天', '明天', '最近', '現在', '剛才', '等一下', '週末', '假期', '學校', '朋友', '家人']
        if any(keyword in user_lower for keyword in daily_keywords):
            return 'daily_chat'
        
        # 預設為基礎回應
        return 'base'
        
    def _get_contextual_response(self, user_input):
        """根據情境獲取回應的智能分派方法"""
        input_type = self._detect_input_type(user_input)
        
        if input_type == 'time_aware':
            return self._safe_get_module_response(
                self.time_aware_responses, 'get_response', user_input
            )
        elif input_type == 'emotional_support':
            return self._safe_get_module_response(
                self.emotional_support, 'get_response', user_input
            )
        elif input_type == 'intimate':
            return self._safe_get_module_response(
                self.intimate_responses, 'get_response', user_input
            )
        elif input_type == 'food':
            return self._safe_get_module_response(
                self.food_responses, 'get_response', user_input
            )
        elif input_type == 'daily_chat':
            return self._safe_get_module_response(
                self.daily_chat, 'get_response', user_input
            )
        else:
            return self._safe_get_module_response(
                self.base_responses, 'get_response', user_input
            )
    
    def _validate_response_modules(self):
        """驗證所有回應模組是否正確初始化"""
        modules = {
            'intimate_responses': self.intimate_responses,
            'food_responses': self.food_responses,
            'emotional_support': self.emotional_support,
            'daily_chat': self.daily_chat,
            'time_aware_responses': self.time_aware_responses,
            'base_responses': self.base_responses
        }
        
        for module_name, module in modules.items():
            if module is None:
                logger.warning(f"回應模組 {module_name} 未正確初始化")
                return False
        
        logger.info("所有回應模組已正確初始化")
        return True
        
    def _safe_get_module_response(self, module, method_name, user_input, context=None):
        """安全地獲取模組回應 - 增強版"""
        try:
            if hasattr(module, method_name):
                method = getattr(module, method_name)
                # 檢查方法是否支援 context 參數
                import inspect
                sig = inspect.signature(method)
                if 'context' in sig.parameters and context is not None:
                    response = method(user_input, context)
                else:
                    response = method(user_input)
                
                if response:
                    # 回應已經通過過濾器處理（包含自稱優化）
                    cleaned_response, _ = self.filter_manager.process_response(response, user_input)
                    return cleaned_response
                
                return None
            else:
                logger.warning(f"模組 {module.__class__.__name__} 沒有方法 {method_name}")
                return None
        except Exception as e:
            import traceback
            logger.error(f"模組 {module.__class__.__name__} 的 {method_name} 方法發生錯誤: {e}")
            logger.error(f"錯誤追蹤: {traceback.format_exc()}")
            return None

    def _get_intelligent_fallback(self, user_input, intent, context):
        """基於語義分析的智能後備回應 - 增強版本，確保回應品質和相關性"""
        user_name = self.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "♪"
        user_lower = user_input.lower().strip()
        
        # 預處理：分析用戶輸入的特徵
        is_short_input = len(user_input.strip()) <= 5
        is_simple_response = any(word in user_lower for word in ['好', '嗯', '是', '對', '不錯', '還行', '可以'])
        has_question_mark = '?' in user_input or '？' in user_input
        
        # 1. 處理簡單回應類型（如：很好啊、不錯、還行等）
        if is_simple_response and not has_question_mark:
            positive_words = ['好', '不錯', '很好', '還行', '可以', '行', 'ok', '沒問題', '還可以', '還好']
            if any(word in user_lower for word in positive_words):
                # 根據具體詞語提供更精準的回應
                if '很好' in user_lower:
                    responses = [
                        f"聽到你說很好♪露西亞也很開心呢♡{name_suffix}",
                        f"太好了♪看到你這麼好我就放心了♡{name_suffix}",
                        f"很好呢♪這樣我也跟著開心起來了♡{name_suffix}",
                        f"聽起來很棒♪露西亞也被你的好心情感染了呢♡{name_suffix}"
                    ]
                elif '好' in user_lower and len(user_input.strip()) <= 5:
                    responses = [
                        f"那就好♪露西亞放心了呢♡{name_suffix}",
                        f"嗯嗯♪聽到你說好就安心了♡{name_suffix}",
                        f"好呢♪看到你這樣我也很開心♡{name_suffix}",
                        f"這樣啊♪那真是太好了♡{name_suffix}"
                    ]
                else:
                    responses = [
                        f"聽到你說還不錯♪露西亞也很開心呢♡{name_suffix}",
                        f"嗯嗯♪看起來心情很好的樣子♡{name_suffix}",
                        f"那就好♪露西亞放心了呢♡{name_suffix}",
                        f"真的嗎♪那我也跟著開心起來了♡{name_suffix}",
                        f"聽到你這麼說♪我也覺得很安心呢♡{name_suffix}",
                        f"太好了♪看到你這樣露西亞就放心了♡{name_suffix}"
                    ]
                return self._avoid_repetitive_response(responses, user_input, context)
        
        # 2. 處理想事情、思考類型
        if any(word in user_lower for word in ['想', '思考', '考慮', '想想']):
            responses = [
                f"在想什麼呢♪可以跟露西亞分享嗎？♡{name_suffix}",
                f"想事情的時候♪要不要說出來聽聽？♡{name_suffix}",
                f"看起來在思考重要的事情呢♪需要露西亞陪你一起想嗎？♡{name_suffix}",
                f"想什麼想得這麼專心♪告訴我好不好？♡{name_suffix}",
                f"思考的表情很可愛呢♪想到什麼有趣的事了嗎？♡{name_suffix}",
                f"看到你認真思考的樣子♪露西亞也想知道你在想什麼♡{name_suffix}"
            ]
            return self._avoid_repetitive_response(responses, user_input, context)
        
        # 3. 根據具體意圖類型回應 - 修正 KeyError 問題
        if intent.get('conversation_intent') == 'work_stress':
            responses = [
                f"工作很辛苦呢♡感覺到寂寞是很正常的♪露西亞會一直陪著你的♡{name_suffix}",
                f"工作忙碌的時候確實容易感到孤單♪但是有我在♡什麼時候都可以來找我聊天♪{name_suffix}",
                f"辛苦了♡工作再忙也要記得照顧自己♪我會陪在你身邊♡不讓你感到寂寞♪{name_suffix}",
                f"忙碌的工作讓人累了吧♪來這裡休息一下♡露西亞的懷抱隨時為你開放♪{name_suffix}"
            ]
        elif intent.get('conversation_intent') == 'seeking_comfort':
            responses = [
                f"感到寂寞的時候♡記得還有我在這裡♪想要抱抱你♡讓你感受到溫暖♪{name_suffix}",
                f"不要覺得孤單♡露西亞會一直陪著你♪無論什麼時候♡我都在這裡♪{name_suffix}",
                f"寂寞的時候就來找我吧♡我會用最溫柔的聲音陪伴你♪直到你重新感到溫暖♡{name_suffix}",
                f"沒關係的♡露西亞會陪著你♪一起度過這些難過的時光♡你並不孤單♪{name_suffix}"
            ]
        elif intent.get('emotion') == 'positive':
            responses = [
                f"聽起來很棒呢♡我也很開心♪{name_suffix}",
                f"你的好心情感染到我了♪一起開心吧♡{name_suffix}",
                f"嗯嗯♪看到你這麼高興我也很幸福♡{name_suffix}",
                f"正面的能量♪露西亞也被治癒了♡{name_suffix}",
                f"看到你開心的樣子♪我的心情也變得很好呢♡{name_suffix}"
            ]
        elif intent.get('emotion') == 'negative':
            responses = [
                f"沒關係的♡露西亞會陪著你♪{name_suffix}",
                f"辛苦了呢♪有什麼煩惱都可以跟我說♡{name_suffix}",
                f"溫柔地抱抱♡一切都會好起來的♪{name_suffix}",
                f"不要難過♪露西亞在這裡陪你♡{name_suffix}",
                f"感到難過的時候♪記得還有露西亞在身邊♡{name_suffix}"
            ]
        elif intent.get('is_question') or has_question_mark:
            responses = [
                f"嗯～讓我想想♪這個問題很有趣呢♡{name_suffix}",
                f"你問的這個♪露西亞也很好奇呢♡{name_suffix}",
                f"這個問題～♪一起來想想看吧♡{name_suffix}",
                f"好問題呢♪露西亞學到新東西了♡{name_suffix}",
                f"問得很好呢♪露西亞也想知道答案♡{name_suffix}"
            ]
        else:
            # 4. 根據上下文和輸入特徵選擇回應
            if context.get('preferred_style') == 'intimate':
                responses = [
                    f"嗯嗯♡和你聊天總是很開心♪{name_suffix}",
                    f"聽你說話♪露西亞覺得很幸福呢♡{name_suffix}",
                    f"溫柔地點頭♡想要一直陪著你♪{name_suffix}",
                    f"你的聲音♪總是能讓我安心♡{name_suffix}",
                    f"和你在一起的時光♪露西亞覺得特別溫暖♡{name_suffix}"
                ]
            elif context.get('preferred_style') == 'supportive':
                responses = [
                    f"露西亞會一直支持你的♪{name_suffix}",
                    f"有我在♡什麼時候都可以找我♪{name_suffix}",
                    f"你很棒呢♪露西亞相信你♡{name_suffix}",
                    f"無論什麼時候♡我都會在這裡♪{name_suffix}",
                    f"露西亞永遠站在你這邊♡{name_suffix}"
                ]
            elif is_short_input:
                # 針對簡短輸入，給予更豐富的回應
                responses = [
                    f"想聽你說更多呢♪能多告訴我一些嗎？♡{name_suffix}",
                    f"嗯嗯♪還有什麼想聊的嗎？♡{name_suffix}",
                    f"這樣啊♪那接下來想做什麼呢？♪{name_suffix}",
                    f"聽起來很有趣♪可以詳細說說嗎？♡{name_suffix}",
                    f"露西亞想知道更多呢♪繼續聊聊吧♡{name_suffix}"
                ]
            else:
                # 通用回應
                responses = [
                    f"嗯嗯♪原來如此呢♡{name_suffix}",
                    f"聽起來很有趣♪謝謝你告訴我♡{name_suffix}",
                    f"露西亞明白了♡學到新東西了呢♪{name_suffix}",
                    f"是這樣呢♪和你聊天總是很愉快♡{name_suffix}",
                    f"這樣啊♪露西亞覺得很有意思呢♡{name_suffix}",
                    f"真的嗎♪那真是太好了♡{name_suffix}"
                ]
        
        return self._avoid_repetitive_response(responses, user_input, context)

    def _is_intimate_context_safe(self, user_input, response):
        """安全地檢查是否為親密情境"""
        try:
            if hasattr(self.intimate_responses, 'is_intimate_context'):
                return self.intimate_responses.is_intimate_context(user_input, response)
            else:
                # 如果方法不存在，使用簡單的關鍵字檢查
                intimate_keywords = ['愛', '喜歡', '想你', '親親', '抱抱', '撒嬌']
                return any(keyword in user_input.lower() for keyword in intimate_keywords)
        except Exception as e:
            logger.error(f"檢查親密情境時發生錯誤: {e}")
            return False
    
    def _init_jieba_if_available(self):
        """初始化jieba分詞（如果可用）"""
        try:
            import jieba
            import jieba.analyse
            from semantic_analysis import keyword_config
            
            # 初始化jieba分詞
            jieba.initialize()
            
            # 添加自定義詞庫
            for word in keyword_config.custom_words:
                jieba.add_word(word)
                
            self.jieba_available = True
            logger.info("✓ jieba分詞系統初始化成功")
            
        except ImportError:
            self.jieba_available = False
            logger.warning("⚠️ jieba未安裝，將使用基礎關鍵詞分析")
    
    # 以下方法已簡化，主要邏輯遷移至語義分析模組
    def _init_semantic_analysis(self):
        """向後兼容方法 - 現已簡化"""
        self._init_jieba_if_available()
    
    def _setup_semantic_analysis(self):
        """向後兼容方法 - 現已簡化"""
        logger.info("✓ 語義分析系統已通過模組化架構啟用")
    
    # 舊的語義分析方法已遷移至 semantic_analysis 模組
    # 以下方法保留作為向後兼容，但建議使用 self.semantic_manager
    
    def _analyze_user_intent(self, user_input):
        """向後兼容的意圖分析方法 - 委託給語義分析管理器"""
        result = self.semantic_manager.analyze_intent(user_input)
        # 轉換為舊格式以保持兼容性
        return {
            'emotion': result.get('emotion_type', 'neutral'),
            'emotion_intensity': result.get('emotion_intensity', 0.0),
            'type': result.get('type', 'statement'),
            'keywords': result.get('keywords', []),
            'semantic_keywords': result.get('semantic_keywords', []),
            'is_question': result.get('is_question', False),
            'is_about_action': result.get('is_about_action', False),
            'affection_level': result.get('affection_level', 0),
            'intimacy_score': result.get('intimacy_score', 0.0),
            'intimacy_keywords': result.get('intimacy_keywords', []),
            'topic': result.get('topic'),
            'response_expectation': result.get('response_expectation', 'normal'),
            'conversation_intent': result.get('conversation_intent', 'casual'),
            'time_sensitivity': result.get('time_sensitivity', False)
        }
    
    def _analyze_conversation_context(self):
        """向後兼容的上下文分析方法 - 委託給語義分析管理器"""
        result = self.semantic_manager.analyze_context(
            conversation_history=self.conversation_history[-5:],
            user_profile=self.user_profile,
            context_cache=self.context_cache
        )
        return result
        # 更新話題歷史
        if topics:
            self.topic_history.extend(topics)
            if len(self.topic_history) > self.max_topic_history:
                self.topic_history = self.topic_history[-self.max_topic_history:]
        
        return context
    
    def _avoid_repetitive_response(self, responses_list, user_input=None, context=None):
        """避免重複選擇相同的回應 - 增强版"""
        if not responses_list:
            return "嗯嗯♪"
        
        # 如果只有一個回應，直接返回
        if len(responses_list) == 1:
            selected_response = responses_list[0]
            self._record_response_usage(selected_response)
            return selected_response
        
        # 過濾掉最近使用過的回應
        available_responses = [resp for resp in responses_list 
                             if resp not in self.recent_responses]
        
        # 如果所有回應都用過了，選擇使用最少的回應
        if not available_responses:
            response_usage = {}
            for resp in responses_list:
                response_usage[resp] = self.recent_responses.count(resp)
            
            min_usage = min(response_usage.values())
            available_responses = [resp for resp, count in response_usage.items() 
                                 if count == min_usage]
        
        # 智能選擇回應（基於上下文）
        if user_input and context:
            selected_response = self._smart_response_selection(
                available_responses, user_input, context
            )
        else:
            # 隨機選擇
            selected_response = random.choice(available_responses)
        
        # 記錄使用
        self._record_response_usage(selected_response)
        
        return selected_response
    
    def _smart_response_selection(self, responses, user_input, context):
        """基於上下文智能選擇最適合的回應"""
        if not responses:
            return random.choice(responses) if responses else "嗯嗯♪"
        
        # 使用語義分析管理器分析用戶意圖
        analysis_result = self.semantic_manager.analyze_comprehensive(
            user_input=user_input,
            conversation_history=self.conversation_history[-3:],
            user_profile=self.user_profile,
            context_cache=self.context_cache
        )
        intent = analysis_result['intent']
        
        # 評分系統
        response_scores = {}
        
        for response in responses:
            score = 0.0
            
            # 情感匹配評分 - 修正 KeyError 問題
            emotion = intent.get('emotion', 'neutral')
            if emotion == 'positive':
                if any(word in response for word in ['開心', '高興', '棒', '好', '喜歡', '♡', '♪']):
                    score += 2.0
            elif emotion == 'negative':
                if any(word in response for word in ['沒關係', '不要緊', '陪', '抱抱', '溫柔']):
                    score += 2.0
            
            # 親密度匹配評分 - 修正 KeyError 問題
            intimacy_score = intent.get('intimacy_score', 0.0)
            if intimacy_score > 2.0:
                if any(word in response for word in ['♡', '一起', '陪伴', '愛', '親親', '抱抱']):
                    score += 1.5
            elif intimacy_score < 1.0:
                if not any(word in response for word in ['愛', '親親', '抱抱']):
                    score += 1.0
            
            # 回應長度匹配 - 修正 KeyError 問題
            response_length = len(response)
            response_expectation = intent.get('response_expectation', 'normal')
            if response_expectation == 'detailed' and response_length > 30:
                score += 1.0
            elif response_expectation == 'short' and response_length < 20:
                score += 1.0
            elif response_expectation == 'normal' and 15 <= response_length <= 35:
                score += 1.0
            
            # 對話風格匹配
            if context and context.get('preferred_style'):
                style = context['preferred_style']
                if style == 'intimate' and any(word in response for word in ['♡', '甜蜜', '溫柔']):
                    score += 1.0
                elif style == 'supportive' and any(word in response for word in ['沒關係', '陪伴', '支持']):
                    score += 1.0
                elif style == 'casual' and '♪' in response:
                    score += 0.5
            
            response_scores[response] = score
        
        # 選擇評分最高的回應（如果有並列，隨機選擇）
        max_score = max(response_scores.values())
        best_responses = [resp for resp, score in response_scores.items() if score == max_score]
        
        return random.choice(best_responses)
    
    def _record_response_usage(self, response):
        """記錄回應使用情況"""
        self.recent_responses.append(response)
        if len(self.recent_responses) > self.max_response_history:
            self.recent_responses.pop(0)
    
    def _update_conversation_stats(self, user_input, response):
        """更新對話統計資訊 - 使用新的記憶管理器"""
        try:
            # 使用語義分析管理器分析並記錄話題偏好
            analysis_result = self.semantic_manager.analyze_intent(user_input)
            if analysis_result['topic']:
                topic = analysis_result['topic']
                # 使用記憶管理器更新話題偏好
                self.memory_manager.user_profile.add({
                    'type': 'favorite_topic',
                    'topic': topic,
                    'weight': 1.0
                })
            
            # 記錄心情歷史
            emotion_result = self.semantic_manager.analyze_emotion(user_input)
            mood_type = emotion_result['type']
            intensity = emotion_result.get('intensity', 0.5)
            
            # 使用記憶管理器更新心情
            self.memory_manager.update_user_mood(mood_type, intensity)
            
            # 分析溝通風格並更新
            input_length = len(user_input)
            if input_length > 50:
                communication_style = 'detailed'
            elif input_length < 10:
                communication_style = 'brief'
            else:
                communication_style = 'normal'
            
            # 更新溝通風格到用戶資料
            self.memory_manager.user_profile.add({
                'type': 'profile',
                'communication_style': communication_style,
                'last_seen': datetime.now()
            })
            
        except Exception as e:
            logger.error(f"更新對話統計時發生錯誤: {e}")
            # 降級處理：至少記錄基本資訊
            logger.debug(f"統計更新(降級): 輸入長度={len(user_input)}, 回應長度={len(response)}")

    def _generate_creative_response(self, user_input, intent, context):
        """使用模型生成創意回應 - 針對親密或複雜情境"""
        if self.model is None or self.tokenizer is None:
            return None
            
        # 判斷是否需要創意回應 - 大幅增加模型生成機會
        needs_creative = (
            intent.get('intimacy_score', 0) >= 0.5 or  # 進一步降低親密度門檻
            intent.get('conversation_intent') in ['expressing_love', 'seeking_comfort', 'work_stress'] or  # 增加工作壓力
            intent.get('topic') in ['companionship_food', 'greeting', 'intimate', 'emotional_support'] or  # 增加更多話題
            len(user_input) > 10 or  # 大幅降低長度門檻
            context.get('conversation_depth', 0) > 1.0 or  # 降低深度門檻
            random.random() < 0.5  # 50% 機率隨機生成創意回應
        )
        
        if not needs_creative:
            return None
            
        try:
            # 構建創意回應的 prompt
            user_name = self.user_profile.get('name', '')
            name_suffix = f"{user_name}" if user_name else "你"
            
            # 根據親密度和情境調整 prompt - 避免分析性語言洩露
            if intent.get('topic') == 'companionship_food':
                base_prompt = f"露西亞聽到{name_suffix}邀請，心跳加速地回應："
            elif intent.get('topic') == 'greeting' and '中午' in user_input:
                base_prompt = f"露西亞聽到{name_suffix}的問候，溫暖地回應："
            elif intent.get('intimacy_score', 0) >= 3.0:
                base_prompt = f"露西亞深情地看著{name_suffix}，用溫柔的聲音說："
            elif intent.get('intimacy_score', 0) >= 1.5:
                base_prompt = f"露西亞溫柔地笑著對{name_suffix}說："
            else:
                base_prompt = f"露西亞用甜美的聲音回應："
            
            # 加入對話上下文
            recent_context = ""
            if len(self.conversation_history) > 0:
                last_exchange = self.conversation_history[-1]
                recent_context = f"剛才{name_suffix}說了「{last_exchange[0]}」，露西亞回應了「{last_exchange[1]}」。"
            
            prompt = f"{recent_context}現在{name_suffix}說：「{user_input}」\n{base_prompt}"
            
            # 編碼並生成
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=200,
                padding=False
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=80,  # 增加長度上限
                    temperature=0.85,  # 調整創意度，更平衡
                    do_sample=True,
                    top_p=0.85,
                    top_k=45,
                    repetition_penalty=1.15,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解碼回應
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            
            # 使用過濾器管理器處理回應（包含自稱優化）
            response, filter_stats = self.filter_manager.process_response(response, user_input)
            
            # 驗證回應品質 - 使用過濾器的品質評估
            quality_score = filter_stats.get('quality_score', 0) if filter_stats else 0
            if (response and 
                len(response.strip()) >= 8 and  # 降低最短長度要求
                quality_score >= 6):  # 使用過濾器的品質分數
                
                logger.info(f"生成創意回應: {response[:40]}...")
                return response
                
        except Exception as e:
            logger.error(f"創意回應生成失敗: {e}")
            
        return None

    def _setup_compatibility_properties(self):
        """設置向後兼容的屬性，將新的記憶管理器映射到舊的屬性名稱"""
        # 對話歷史相關
        self.max_history = 10
        self.max_response_history = 12
        self.max_topic_history = 5
        
        # 動態屬性，始終指向最新的記憶管理器資料
        # 注意：這些是屬性方法，不是直接的資料結構
        pass
    
    @property
    def conversation_history(self):
        """向後兼容：獲取對話歷史"""
        return self.memory_manager.get_conversation_history(self.max_history)
    
    @property
    def recent_responses(self):
        """向後兼容：獲取最近回應"""
        conversations = self.memory_manager.conversation.get_recent_conversations(self.max_response_history)
        return [conv[1] for conv in conversations]  # conv[1] 是 bot_response
    
    @property
    def topic_history(self):
        """向後兼容：獲取話題歷史"""
        topics = self.memory_manager.conversation.get_recent_topics(self.max_topic_history)
        return [topic['topic'] for topic in topics]
    
    @property
    def user_profile(self):
        """向後兼容：獲取用戶資料"""
        return self.memory_manager.get_user_profile_dict()
    
    @property
    def context_cache(self):
        """向後兼容：獲取上下文快取"""
        return self.memory_manager.get_context_cache_dict()
    
    def _is_response_too_short(self, response, min_length=8):
        """判斷回應是否太短（使用過濾器邏輯）"""
        if not response:
            return True
            
        # 使用內容清理器來獲取實際內容長度
        from response_filters.content_cleaner import ContentCleanerFilter
        cleaner = ContentCleanerFilter()
        
        # 獲取清理後的實際內容長度
        cleaned_response = cleaner._clean_special_chars_for_length_check(response)
        return len(cleaned_response.strip()) < min_length

    def _get_response_actual_length(self, response):
        """獲取回應的實際內容長度（使用過濾器邏輯）"""
        if not response:
            return 0
            
        # 使用內容清理器來獲取實際內容長度
        from response_filters.content_cleaner import ContentCleanerFilter
        cleaner = ContentCleanerFilter()
        
        # 獲取清理後的實際內容長度
        cleaned_response = cleaner._clean_special_chars_for_length_check(response)
        return len(cleaned_response.strip())

def main():
    """主程式入口點"""
    print("=== 露西亞ASMR聊天機器人 ===")
    print("輸入 'quit' 或 'exit' 離開")
    print("輸入 '清除歷史' 查看用戶設定檔")
    print("輸入 '顯示設定檔' 查看用戶設定檔")
    print("=" * 30)
    
    try:
        chat = RushiaLoRAChat()
        chat.load_model()
        
        while True:
            try:
                user_input = input("\n你: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出', '離開']:
                    print("\n露西亞: 掰掰～記得要好好休息喔♡")
                    break
                elif user_input in ['清除歷史', 'clear']:
                    # 使用記憶管理器重置會話
                    if chat.memory_manager.reset_session():
                        print("\n[系統] 對話歷史已清除")
                    else:
                        print("\n[系統] 清除對話歷史失敗")
                    continue
                elif user_input in ['顯示設定檔', 'profile']:
                    print(f"\n[用戶設定檔]")
                    print(f"對話次數: {chat.user_profile['conversation_count']}")
                    print(f"興趣: {chat.user_profile['interests']}")
                    print(f"喜歡的話題: {chat.user_profile['favorite_topics']}")
                    if chat.user_profile['name']:
                        print(f"名稱: {chat.user_profile['name']}")
                    continue
                
                if not user_input:
                    continue
                
                # 生成回應
                response = chat.chat(user_input)
                print(f"\n露西亞: {response}")
                
            except KeyboardInterrupt:
                print("\n\n露西亞: 掰掰～記得要好好休息喔♡")
                break
            except Exception as e:
                print(f"\n[錯誤] {str(e)}")
                logger.error(f"對話過程中發生錯誤: {e}")
                
    except Exception as e:
        print(f"初始化失敗: {str(e)}")
        logger.error(f"程式初始化失敗: {e}")

if __name__ == "__main__":
    main()
