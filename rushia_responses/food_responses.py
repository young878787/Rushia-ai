from .base_module import BaseResponseModule
import random
from datetime import datetime, timezone, timedelta

class FoodResponses(BaseResponseModule):
    """食物相關回應處理"""
    
    def __init__(self, chat_instance):
        super().__init__(chat_instance)
        
    def get_response(self, user_input, context=None):
        """統一的回應入口方法"""
        if self.is_food_related(user_input):
            return self.get_food_response(user_input)
        return None
    
    def get_food_response(self, user_input):
        """處理食物相關的對話 - 包含多種用餐情境"""
        user_lower = user_input.lower()
        
        # 首先檢查是否真的與食物相關
        if not self.is_food_related(user_input):
            return None  # 不是食物相關，讓其他模組處理
            
        user_name = self.chat_instance.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "你♪"
        
        # 獲取台灣時間來判斷合適的用餐建議
        taiwan_tz = timezone(timedelta(hours=8))
        taiwan_time = datetime.now(taiwan_tz)
        hour = taiwan_time.hour
        
        # 不餓/飽了的情境
        if any(word in user_lower for word in ['不餓', '飽了', '不想吃', '吃不下', '飽']):
            return random.choice([
                f"啊♪{name_suffix}已經飽了呀～那就不勉強了♡",
                f"嗯嗯♪吃飽了最重要♡那我們聊點別的吧～",
                f"不餓的話就好♪身體健康最重要呢{name_suffix}♡",
                f"那就等餓了再吃吧♪露西亞陪你聊天～♡",
                f"飽飽的{name_suffix}看起來很滿足呢♪那要不要喝點茶？♡"
            ])
        
        # 想要露西亞料理/做菜的情境
        elif any(word in user_lower for word in ['露西亞料理', '露西亞做', '你做', '你會做', '你可以做', '你可以煮', '料理', '做菜', '煮飯', '廚藝', '煮', '煮給我']):
            return random.choice([
                f"嗯♪露西亞會做一些簡單的料理呢～想吃什麼{name_suffix}？♡",
                f"露西亞的料理♪嗯...會用滿滿的愛心來做喔♡{name_suffix}想嚐嚐嗎？",
                f"雖然不是很厲害♪但露西亞會努力做美味的料理給你♡",
                f"嘿嘿♪露西亞最拿手的是...溫暖的家常菜♡要不要一起做呢{name_suffix}？",
                f"用心做的料理最美味了♪露西亞想為{name_suffix}做特別的料理♡",
                f"在廚房裡哼著歌做料理♪想像{name_suffix}開心吃的樣子就很幸福呢♡",
                f"當然可以呀♪露西亞很樂意為{name_suffix}煮東西♡想吃什麼呢？",
                f"煮給{name_suffix}吃♪這讓露西亞很開心呢♡"
            ])
        
        # 外出用餐的情境
        elif any(word in user_lower for word in ['出去吃', '外面吃', '餐廳', '外食', '出去用餐', '約吃飯']):
            if 6 <= hour < 12:
                meal_suggestions = ['早餐店', '咖啡廳', '麵包店']
                meal_time = "早餐"
            elif 12 <= hour < 15:
                meal_suggestions = ['日式料理', '義式餐廳', '中式餐廳', '簡餐店']
                meal_time = "午餐"
            elif 15 <= hour < 18:
                meal_suggestions = ['下午茶店', '咖啡廳', '甜點店']
                meal_time = "下午茶"
            else:
                meal_suggestions = ['居酒屋', '火鍋店', '燒肉店', '家庭餐廳']
                meal_time = "晚餐"
            
            suggestion = random.choice(meal_suggestions)
            return random.choice([
                f"好呀♪{name_suffix}想出去吃{meal_time}嗎？去{suggestion}怎麼樣？♡",
                f"外出用餐♪聽起來很棒呢～露西亞想和{name_suffix}一起去{suggestion}♡",
                f"嗯嗯♪出去吃{meal_time}很有趣呢♪{suggestion}的氣氛一定很棒♡",
                f"和{name_suffix}一起出去吃飯♪想到就很開心呢～{suggestion}好不好？♡",
                f"外面的料理也很美味呢♪露西亞想和{name_suffix}一起享受{suggestion}的時光♡"
            ])
        
        # 下午茶專門情境
        elif any(word in user_lower for word in ['下午茶', '茶', '咖啡', '蛋糕', '司康', '點心時間']):
            return random.choice([
                f"下午茶時光♪{name_suffix}想要什麼呢？蛋糕還是司康？♡",
                f"優雅的下午茶時間到了♪露西亞想和{name_suffix}一起享用♡",
                f"香濃的茶香配上甜美的點心♪這樣的時光最幸福了呢{name_suffix}♡",
                f"嗯♪來一場悠閒的下午茶吧～露西亞準備了{name_suffix}喜歡的♡",
                f"下午茶的甜蜜時光♪想和{name_suffix}一起品嚐美味的點心♡",
                f"輕鬆的下午茶♪配上{name_suffix}的陪伴就是最棒的組合呢♡"
            ])
        
        # 甜點/零食情境
        elif any(word in user_lower for word in ['甜點', '蛋糕', '巧克力', '冰淇淋', '布丁', '零食', '餅乾']):
            return random.choice([
                f"甜甜的♪{name_suffix}想吃什麼口味的甜點呢？草莓還是巧克力？♡",
                f"甜點能讓心情變好呢♪露西亞想和{name_suffix}一起享用♡",
                f"嗯♪甜蜜蜜的味道最棒了～{name_suffix}的笑容也像甜點一樣甜美♡",
                f"來一份特別的甜點吧♪露西亞親手做給{name_suffix}的♡",
                f"零食時間♪想和{name_suffix}一起分享美味的點心呢♡",
                f"甜膩膩的幸福感♪和{name_suffix}一起吃甜點是最快樂的事♡"
            ])
        
        # 一般饑餓/想吃東西的情境
        elif any(word in user_lower for word in ['餓', '肚子餓', '想吃', '飢餓']):
            if 6 <= hour < 12:
                return random.choice([
                    f"早晨的肚子咕嚕咕嚕♪{name_suffix}想吃什麼早餐呢？♡",
                    f"餓了呀♪早餐是一天活力的來源呢～想吃什麼{name_suffix}？♡",
                    f"嗯♪早上餓了要好好吃早餐喔♡露西亞陪{name_suffix}一起吃♪"
                ])
            elif 12 <= hour < 15:
                return random.choice([
                    f"午餐時間到了♪{name_suffix}肚子餓了嗎？想吃什麼呢♡",
                    f"咕嚕咕嚕♪午餐要吃飽飽的才有精神呢{name_suffix}♡",
                    f"中午餓了♪露西亞想和{name_suffix}一起享用美味的午餐♡"
                ])
            elif 15 <= hour < 18:
                return random.choice([
                    f"下午有點餓了嗎♪要不要來點下午茶{name_suffix}？♡",
                    f"午後的小餓♪來份輕食或甜點怎麼樣呢？♡",
                    f"嗯♪下午茶時間♪{name_suffix}想吃點什麼補充體力呢？♡"
                ])
            else:
                return random.choice([
                    f"晚餐時間到了♪{name_suffix}今天想吃什麼呢？♡",
                    f"呀♪肚子餓了呢～晚餐要吃得豐盛一點{name_suffix}♡",
                    f"夜晚餓了♪露西亞想和{name_suffix}一起享用溫暖的晚餐♡"
                ])
        
        # 喂食情境
        elif any(word in user_lower for word in ['喂', '餵', '一口', '吃一口', '嚐一口', '餵我', '餵食']):
            return random.choice([
                "當然可以呀～張嘴♪啊～♡",
                "嗯嗯♪來～露西亞餵你♡慢慢吃呢～",
                "好的♪啊～小心燙喔♡",
                "嗯～張開嘴巴♪露西亞溫柔地餵你♡",
                "當然♪啊～這樣好嗎？♡很甜呢～",
                "好呀♪來～慢慢吃♡露西亞看著你♪",
                "餵你吃東西很幸福呢♪啊～♡",
                "溫柔地餵你♡要慢慢咀嚼喔♪",
                "一口一口地餵♡看你吃得開心露西亞也很高興♪",
                "來♪張嘴～露西亞親手餵你♡",
                "甜甜的♪配上你的笑容更甜呢♡",
                "餵食的時光很溫馨呢♪啊～好乖♡"
            ])
        
        # 美食相關
        elif any(word in user_lower for word in ['美食', '料理', '烹飪', 'cooking', '甜點', '蛋糕']):
            return random.choice([
                f"想嚐嚐美味的料理呢♡{name_suffix}最喜歡什麼口味？", 
                f"甜點讓人心情愉快♪露西亞想和{name_suffix}一起做料理呢♡", 
                f"一起做料理會很有趣呢♡{name_suffix}想學什麼呢？",
                f"美食能帶來幸福感♪和{name_suffix}分享美味最開心了♡"
            ])
        
        # 其他食物相關的一般回應 - 改為更個性化和豐富的回應
        else:
            # 根據具體輸入內容給出更個性化的回應
            if any(word in user_lower for word in ['好吃', '美味', '好喝', '讚', '棒', '不錯', '香', '甜']):
                return random.choice([
                    f"聽到{name_suffix}說好吃我就放心了♡能讓你滿意真的很開心♪",
                    f"嗯嗯♪{name_suffix}喜歡就好♡看到你開心的樣子我也很幸福呢～",
                    f"被{name_suffix}誇獎了♪臉都紅了呢♡下次還要做更好吃的給你♪",
                    f"真的嗎？♪那我就更有信心了♡{name_suffix}的笑容是最好的調味料呢♪",
                    f"能做出{name_suffix}喜歡的味道♡是我最大的成就感♪嘿嘿～"
                ])
            elif any(word in user_lower for word in ['想吃', '想喝', '想要', '期待', '渴望']):
                return random.choice([
                    f"嗯♪{name_suffix}想吃什麼呢？♡露西亞會盡力滿足你的♪",
                    f"想要的話♪露西亞隨時可以做給{name_suffix}吃♡",
                    f"哇♪{name_suffix}的要求露西亞最喜歡了♡什麼時候想吃呢？♪",
                    f"期待的話♪那我們一起計劃一下吧♡{name_suffix}♪"
                ])
            elif any(word in user_lower for word in ['其他', '別的', '換個', '不一樣', '聊別的', '說別的']):
                return random.choice([
                    f"嗯♪那{name_suffix}想聊什麼呢？♡露西亞都會認真聽的♪",
                    f"好呀♪雖然喜歡聊食物♡但和{name_suffix}聊什麼都開心♪",
                    f"換個話題也不錯呢♪{name_suffix}有什麼想分享的嗎？♡",
                    f"當然可以♪{name_suffix}想聊的話題露西亞都很感興趣♡"
                ])
            else:
                # 兜底回應，但要更有變化性
                return random.choice([
                    f"和{name_suffix}一起聊天的時光最幸福了♡不管聊什麼都很開心♪",
                    f"美味的食物配上{name_suffix}的陪伴♪完美的組合呢♡",
                    f"嗯♪{name_suffix}剛才說的話讓我想到了很多呢♡繼續聊吧♪",
                    f"和{name_suffix}在一起♪就連聊天都變得格外有趣♡",
                    f"你知道嗎{name_suffix}？♡和你聊天總是能讓我心情變好♪"
                ])
    
    def is_food_related(self, user_input):
        """檢查輸入是否與食物相關"""
        user_lower = user_input.lower()
        food_keywords = [
            '吃', '食物', '餓', '飯', '餐', '料理', '廚', '下午茶', '點心', '甜點', 
            '零食', '美食', '烹飪', 'cooking', '蛋糕', '喂', '餵', '一口', 
            '出去吃', '外面吃', '餐廳', '外食', '茶', '咖啡', '巧克力', 
            '冰淇淋', '布丁', '餅乾', '司康', '煮飯', '做菜'
        ]
        return any(keyword in user_lower for keyword in food_keywords)