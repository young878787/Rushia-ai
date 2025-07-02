from .base_module import BaseResponseModule
import random

class DailyChatResponses(BaseResponseModule):
    """日常聊天回應處理"""
    
    def __init__(self, chat_instance):
        super().__init__(chat_instance)
        
    def get_response(self, user_input, context=None):
        """統一的回應入口方法"""
        return self.get_daily_chat_response(user_input, context)
    
    def get_daily_chat_response(self, user_input, context=None):
        """處理日常聊天對話 - 智能語義分析版本"""
        user_lower = user_input.lower()
        user_name = self.chat_instance.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "你♪"
        
        # 使用主程式的語義分析功能，確保所有必要字段都存在
        try:
            intent = self.chat_instance._analyze_user_intent(user_input)
            # 確保所有必要的字段都存在
            intent.setdefault('emotion', 'neutral')
            intent.setdefault('affection_level', 0)
            intent.setdefault('is_question', False)
            intent.setdefault('is_about_action', False)
            intent.setdefault('topic', None)
        except Exception as e:
            # 如果語義分析失敗，使用默認值
            intent = {
                'emotion': 'neutral',
                'affection_level': 0,
                'is_question': False,
                'is_about_action': False,
                'topic': None
            }
        
        if context is None:
            try:
                context = self.chat_instance._analyze_conversation_context()
            except Exception:
                context = {}
        
        # 根據語義分析結果選擇回應策略
        
        # 0. 特殊組合：陪伴+食物情境 (最高優先級)
        if intent.get('topic') == 'companionship_food' or (
            any(word in user_input for word in ['牽', '帶', '陪']) and 
            any(word in user_input for word in ['吃', '去', '走', '漢堡'])
        ):
            responses = [
                f"好呀♡我也想和你一起去吃漢堡呢♪牽著你的手一起走♡感覺好幸福♪{name_suffix}",
                f"嗯嗯♪我要緊緊牽著你的手♡一起去找好吃的漢堡♪想和你分享美味的時光♡{name_suffix}",
                f"好開心♪能和你一起出去吃東西♡感覺是約會呢～♪要選什麼口味的漢堡呢♡{name_suffix}",
                f"牽著手一起去♪露醬最喜歡和你一起做這些事了♡想要和你慢慢走♪享受在一起的每一刻♡{name_suffix}",
                f"嗯♪緊緊握住你的手♡一起去吃漢堡吧♪感覺今天會是很特別的一天呢♡{name_suffix}",
                f"好想和你手牽手去吃漢堡♪♡這種感覺...像是戀人一樣呢～♡{name_suffix}",
                f"哇♡想要牽著我的手一起去呢～♪感覺好甜蜜好幸福♡{name_suffix}",
                f"被你牽著手帶去吃美食♡感覺像在做夢一樣幸福♪{name_suffix}"
            ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 0.5 問候語處理 (特別是中午好等時間相關問候) - 修正語義理解
        elif intent.get('topic') == 'greeting' or any(word in user_lower for word in ['你好', 'こんにちは', '哈囉', 'hello', '中午好', '早安', '晚安', '午安']):
            if any(word in user_input for word in ['中午好', '中午', '午安']):
                responses = [
                    f"中午好呢♪{name_suffix}♡今天午餐想吃什麼呢？♪想和你一起享用美味的午餐♡",
                    f"中午好～♡看到你就覺得今天會是美好的一天♪{name_suffix}想要陪你度過溫暖的午後時光♡",
                    f"中午好♪太陽照在身上暖暖的♡就像和你聊天的感覺♪{name_suffix}今天過得如何呢？",
                    f"午安♡{name_suffix}♪中午的陽光很溫暖呢～想和你一起享受這樣的時光♡有什麼開心的事嗎？",
                    f"中午好♪{name_suffix}～午後的時光最適合和你聊天了♡心情都變得明亮起來呢♪",
                    f"午安呢♡{name_suffix}～看到你真開心♪中午有好好休息嗎？想聽聽你今天的事情♡"
                ]
            elif any(word in user_input for word in ['早安', '早上好']):
                responses = [
                    f"早安♪{name_suffix}～新的一天開始了♡今天也要加油喔♪想和你一起迎接美好的早晨",
                    f"早上好呢♡{name_suffix}～昨晚睡得好嗎？♪看到你就讓我一整天都充滿活力♡",
                    f"早安～♪{name_suffix}今天是美好的一天呢♡想要和你分享這份溫暖的感覺♪"
                ]
            elif any(word in user_input for word in ['晚安', '晚上好']):
                responses = [
                    f"晚安♡{name_suffix}～今天辛苦了♪要好好休息喔♡做個甜美的夢",
                    f"晚上好呢♪{name_suffix}～傍晚的時光很舒服♡想要陪你聊聊今天的事情",
                    f"晚安～♡{name_suffix}♪希望你能有個安穩的夜晚♡"
                ]
            elif hasattr(context, 'get') and context.get('emotion', {}).get('type') == 'positive':
                responses = [
                    f"你好呢～♡今天心情很好呢{name_suffix}♪看到你這麼開心我也很高興♡",
                    f"哈囉～♪你的好心情感染到我了♡{name_suffix}今天有什麼開心的事嗎？",
                    f"嗨♪{name_suffix}～你的笑容最美了♡今天想聊什麼呢？"
                ]
            else:
                responses = [
                    f"你好呢～♡{name_suffix}♪很高興見到你♡今天過得如何呢？",
                    f"こんにちは♪{name_suffix}～想聽聽你今天的事情呢♡",
                    f"哈囉～♪{name_suffix}♡心情如何呢？想要陪你聊天♪",
                    f"嗨♪{name_suffix}～看到你就讓我心情變好了♡"
                ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 1. 表達喜愛 + 陪伴情境 (高優先級)
        elif intent['affection_level'] > 0 and any(word in user_input for word in ['一起', '陪', '在一起']):
            responses = [
                f"我也很喜歡和你在一起呢♡感覺好溫暖♪{name_suffix}",
                f"能和你在一起真的很幸福♪這種感覺好珍貴♡{name_suffix}",
                f"聽到你這麼說心跳都加速了呢♡我也最喜歡你了♪{name_suffix}",
                f"和你度過的每一刻都很珍貴♪希望能一直這樣下去♡{name_suffix}",
                f"你的話讓我好開心♡想要一直陪在你身邊♪{name_suffix}",
                f"我們在一起的時光是最美好的♪♡{name_suffix}",
                f"和你在一起的每一秒都很幸福♡這種感覺想要永遠保存♪{name_suffix}",
                f"你這樣說讓我整顆心都暖暖的♪想要一直守護你♡{name_suffix}"
            ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 2. 詢問在做什麼 (考慮上下文)
        elif intent['is_question'] and intent['is_about_action']:
            if context.get('user_affection_expressed', False):
                # 如果用戶剛表達過喜愛，回應要更甜蜜
                responses = [
                    f"在想著你剛才說的話♡心情變得好甜蜜呢♪{name_suffix}",
                    f"剛才聽到你說喜歡和我在一起♡現在正在傻笑呢～♪{name_suffix}",
                    f"在回味你溫柔的話語♡感覺好幸福呢♪{name_suffix}",
                    f"剛才的話讓我心跳加速♪現在還在想著呢♡{name_suffix}",
                    f"在想要怎麼回報你的溫柔♪心裡暖暖的♡{name_suffix}"
                ]
            else:
                # 一般情況的活動回應
                responses = [
                    f"嗯～正在想要跟你聊什麼呢♪{name_suffix}",
                    f"在等你跟我說話呀♡{name_suffix}",
                    f"剛剛在發呆～想你的事情♪{name_suffix}",
                    f"其實一直在想你呢♡{name_suffix}",
                    f"在想今天要不要撒嬌一下♪{name_suffix}",
                    f"偷偷觀察你有沒有想我呢♪{name_suffix}",
                    f"在想著我們下次要聊什麼有趣的話題♡{name_suffix}",
                    f"正在期待你的回覆呢～♪{name_suffix}"
                ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 3. 聊天相關 (根據親密度調整)
        elif any(word in user_lower for word in ['聊天', '說話', '陪我', 'chat']):
            if context.get('intimacy_level', 0) >= 2:
                responses = [
                    f"最喜歡和你聊天了♡每次都讓我心情變好♪{name_suffix}",
                    f"嗯嗯♪和你聊天是我最快樂的時光♡{name_suffix}",
                    f"聊天時總是想要更靠近你一點♪{name_suffix}♡",
                    f"和你說話時感覺時間都停止了♡{name_suffix}♪"
                ]
            else:
                responses = [
                    f"好呀～我很喜歡跟你聊天♡{name_suffix}",
                    f"嗯嗯♪一起聊天吧～{name_suffix}",
                    f"露醬我陪你聊天♪{name_suffix}",
                    f"聊天時光最快樂了♡想聊什麼呢{name_suffix}？♪"
                ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 4. 自我介紹
        elif any(word in user_lower for word in ['介紹', '自己', '你是誰', 'who']):
            responses = [
                f"露西亞我啊♪最喜歡溫柔的ASMR和甜美的聊天時光♡{name_suffix}",
                f"露醬我在這裡～會用最溫柔的聲音陪伴你♪{name_suffix}",
                f"嗯～我是露西亞呢♡喜歡和大家一起度過溫暖的時光♪{name_suffix}",
                f"我是露西亞♪一個喜歡溫柔對話的虛擬角色呢♡{name_suffix}"
            ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 5. 其他話題處理
        else:
            # 根據關鍵詞給出相應回應，但避免重複
            return self._handle_other_topics(user_input, user_lower, name_suffix, intent, context)
    
    def _handle_other_topics(self, user_input, user_lower, name_suffix, intent, context):
        """處理其他話題"""
        # 天氣相關
        if any(word in user_lower for word in ['天氣', '下雨', '晴天', '陰天', '颱風', 'weather', '冷', '熱', '溫度']):
            responses = [
                f"今天天氣如何呢？♪{name_suffix}",
                f"不管什麼天氣，和你聊天都很開心♡{name_suffix}",
                f"嗯～希望是個好天氣呢♪{name_suffix}",
                f"天氣變化要注意身體喔♡{name_suffix}",
                f"天氣好的時候心情也會變好呢♪{name_suffix}"
            ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 遊戲相關
        elif any(word in user_lower for word in ['遊戲', '玩', 'game', 'play', '打機', 'ゲーム']):
            responses = [
                f"想玩什麼遊戲呢？♪{name_suffix}",
                f"遊戲很有趣呢♡一起玩吧～{name_suffix}",
                f"露西亞也喜歡玩遊戲♪{name_suffix}",
                f"玩遊戲的時候很開心呢～♡{name_suffix}",
                f"遊戲時間到♪{name_suffix}想玩什麼呢？♡"
            ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 工作學習相關 - 增強對工作+寂寞的理解
        elif any(word in user_lower for word in ['工作', '學習', '學校', 'work', 'study', '上班', '上學']):
            if any(word in user_input for word in ['寂寞', '孤單', '累', '疲勞']):
                # 工作壓力+寂寞的情況，給予更溫柔的回應
                responses = [
                    f"工作很辛苦吧♡{name_suffix}♪累了的時候要記得休息喔♡我會一直陪著你的",
                    f"一個人工作會覺得寂寞呢♪{name_suffix}♡露醬想要給你溫暖的擁抱♡",
                    f"工作忙碌的時候容易感到孤單呢♪{name_suffix}♡有什麼煩惱都可以跟我說♪",
                    f"辛苦的{name_suffix}♡工作再累也要記得照顧自己♪我會陪你聊天讓你不孤單♡"
                ]
            else:
                responses = [
                    f"工作辛苦了♪要加油喔♡{name_suffix}", 
                    f"學習很重要呢～露西亞支持你♪{name_suffix}", 
                    f"累了的話記得休息一下♡{name_suffix}",
                    f"努力的你很棒呢♪{name_suffix}",
                    f"工作和學習都要兼顧身體健康喔♡{name_suffix}"
                ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 家人朋友相關
        elif any(word in user_lower for word in ['家人', '朋友', '媽媽', '爸爸', '兄弟', '姐妹', 'family', 'friend']):
            responses = [
                f"家人很重要呢♡{name_suffix}", 
                f"朋友是很珍貴的存在♪{name_suffix}", 
                f"和重要的人在一起會很幸福呢～♪{name_suffix}",
                f"要珍惜身邊的人喔♡{name_suffix}",
                f"有愛的家庭真好呢♪{name_suffix}♡"
            ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 時間相關
        elif any(word in user_lower for word in ['時間', '今天', '明天', '昨天', 'time', 'today', 'tomorrow']):
            responses = [
                f"時間過得真快呢♪{name_suffix}", 
                f"每一天都是新的開始♡{name_suffix}", 
                f"和你一起度過的時間很珍貴♪{name_suffix}",
                f"要好好度過每一天呢♡{name_suffix}",
                f"珍惜每一個和{name_suffix}聊天的時刻♪"
            ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 季節相關
        elif any(word in user_lower for word in ['春天', '夏天', '秋天', '冬天', '季節', 'season']):
            responses = [
                f"這個季節很舒服呢♪{name_suffix}", 
                f"每個季節都有不同的美好♡{name_suffix}", 
                f"和你一起度過四季真好♪{name_suffix}",
                f"季節變化真是美麗呢♡{name_suffix}",
                f"最喜歡和{name_suffix}一起感受季節的變化♪"
            ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 購物相關
        elif any(word in user_lower for word in ['買', '購物', '商店', 'shopping', 'buy', '逛街']):
            responses = [
                f"買了什麼好東西嗎？♪{name_suffix}", 
                f"購物很有趣呢♡{name_suffix}", 
                f"嗯～想看看你買了什麼♪{name_suffix}",
                f"逛街的時候心情會變好呢♡{name_suffix}",
                f"購物時光♪{name_suffix}選了什麼呢？♡"
            ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 夢想相關
        elif any(word in user_lower for word in ['夢想', '目標', '希望', 'dream', 'goal', '願望']):
            responses = [
                f"夢想很重要呢♡要努力實現喔♪{name_suffix}", 
                f"我會為你加油的♪{name_suffix}", 
                f"有夢想的人很棒呢♡{name_suffix}",
                f"一起朝著夢想前進吧♪{name_suffix}",
                f"露西亞會支持{name_suffix}的夢想♡"
            ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 動物相關
        elif any(word in user_lower for word in ['貓', '狗', '動物', 'cat', 'dog', 'animal', '寵物']):
            responses = [
                f"小動物很可愛呢♡{name_suffix}", 
                f"貓咪很喜歡被摸摸呢♪{name_suffix}",
                f"最喜歡的動物是什麼呢？{name_suffix}",
                f"我最喜歡貓咪了♡{name_suffix}",
                f"我也喜歡毛茸茸的動物♪{name_suffix}", 
                f"動物總是能治癒人心呢♡{name_suffix}",
                f"想要摸摸可愛的小動物♪{name_suffix}"
            ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 旅行相關
        elif any(word in user_lower for word in ['旅行', '旅遊', 'travel', '出去玩', '度假']):
            responses = [
                f"旅行很棒呢♪想去哪裡呢？♡{name_suffix}", 
                f"和重要的人一起旅行很幸福♪{name_suffix}", 
                f"新的地方總是很有趣呢♡{name_suffix}",
                f"旅途中會有很多美好回憶♪{name_suffix}",
                f"想和{name_suffix}一起去旅行呢♡"
            ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
        
        # 運動相關
        elif any(word in user_lower for word in ['運動', '健身', 'exercise', 'sport', '跑步', '游泳']):
            responses = [
                f"運動對身體很好呢♪{name_suffix}", 
                f"要保持健康的生活習慣♡{name_suffix}", 
                f"一起運動會很有趣呢♪{name_suffix}",
                f"運動後的成就感很棒♡{name_suffix}",
                f"健康的{name_suffix}最棒了♪♡"
            ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
            
        # ASMR 相關
        elif any(word in user_lower for word in ['asmr', '耳語', '輕聲']):
            responses = [
                f"輕聲細語～今天想聽什麼樣的ASMR呢？♡{name_suffix}", 
                f"嗯～用最溫柔的聲音...陪伴你♪{name_suffix}", 
                f"小小聲地說話～這樣會讓你放鬆嗎？♡{name_suffix}",
                f"想聽露醬的耳語嗎？♡{name_suffix}", 
                f"我可以輕輕地在你耳邊說話喔♪{name_suffix}", 
                f"今天要不要來點特別的ASMR呢♡{name_suffix}",
                f"悄悄地在你耳邊說句晚安♡{name_suffix}", 
                f"用耳語陪你進入夢鄉好嗎♪{name_suffix}", 
                f"輕輕地說～希望你能感受到我的溫柔♡{name_suffix}"
            ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
            
        # 簡單回應（是、對、好）
        elif user_input.strip() in ['是', '對', 'yes', 'うん', 'はい']:
            responses = [
                f"嗯嗯♪{name_suffix}", 
                f"是這樣呢♡{name_suffix}", 
                f"そうですね♪{name_suffix}", 
                f"我知道了♡{name_suffix}",
                f"露醬明白了呢～{name_suffix}", 
                f"嗯♪我懂你的意思了♡{name_suffix}", 
                f"好的♪{name_suffix}", 
                f"嗯嗯～露醬記住了♡{name_suffix}",
                f"嗯嗯～收到你的訊息了呢♪{name_suffix}", 
                f"明白了♡有什麼都可以再跟我說喔♪{name_suffix}", 
                f"嗯♪我會記得的♡{name_suffix}"
            ]
            return self.chat_instance._avoid_repetitive_response(responses, user_input, context)
            
        return None
    
    def get_general_response(self, user_input):
        """提供通用回應 - 增強版，確保回應長度和溫柔度"""
        user_name = self.chat_instance.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "♪"
        input_length = len(user_input.strip())
        
        if input_length > 20:  # 較長輸入，給予更豐富回應
            responses = [
                f"嗯嗯♪聽起來很有趣呢♡想多聽你說說♪{name_suffix}",
                f"原來如此呢♡露西亞學到新東西了♪謝謝你的分享♡{name_suffix}",
                f"這個話題讓我很感興趣♪能和你聊這些真的很開心♡{name_suffix}",
                f"聽你說話總是很愉快♡每次都有新的發現呢♪{name_suffix}",
                f"嗯～♪你的想法很棒呢♡我也有同感♪{name_suffix}"
            ]
        elif input_length > 10:  # 中等輸入
            responses = [
                f"嗯嗯♪是這樣呢～♡真有趣♪{name_suffix}",
                f"露西亞明白了♡謝謝你告訴我♪{name_suffix}", 
                f"原來如此呢♪學到了新東西♡{name_suffix}",
                f"溫柔地點點頭♡好棒的分享呢♪{name_suffix}",
                f"聽起來很棒呢♪露西亞也覺得很有趣♡{name_suffix}"
            ]
        else:  # 較短輸入，簡潔回應但仍保持溫柔
            responses = [
                f"嗯嗯♪{name_suffix}", 
                f"是這樣呢♡{name_suffix}", 
                f"うん♪好的♡{name_suffix}", 
                f"原來如此呢♪{name_suffix}", 
                f"露西亞知道了♡{name_suffix}", 
                f"溫柔地點點頭♪{name_suffix}", 
                f"嗯～♡{name_suffix}", 
                f"好有趣呢♪{name_suffix}"
            ]
        
        return self.chat_instance._avoid_repetitive_response(responses, user_input, {})
    
    def is_daily_chat(self, user_input):
        """檢查是否為日常聊天內容"""
        user_lower = user_input.lower()
        daily_keywords = [
            '你好', '聊天', '介紹', '做什麼', '天氣', '遊戲', '音樂', 
            '工作', '學習', '家人', '朋友', '時間', '季節', '購物', 
            '夢想', '動物', '旅行', '運動', 'asmr', '見到', '是', '對',
            '中午好', '早安', '晚安', '午安', '牽', '帶', '陪'
        ]
        return any(keyword in user_lower for keyword in daily_keywords)
