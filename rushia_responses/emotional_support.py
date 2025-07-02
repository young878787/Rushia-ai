from .base_module import BaseResponseModule
import random
import time

class EmotionalSupportResponses(BaseResponseModule):
    """情感支持回應處理"""
    
    def __init__(self, chat_instance):
        super().__init__(chat_instance)
        
    def get_emotional_support_response(self, user_input):
        """提供情感支持回應"""
        user_lower = user_input.lower()
        user_name = self.chat_instance.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "♪"
        
        # 負面情緒支持
        if any(word in user_lower for word in ['累', '疲勞', '疲れ', 'tired', '辛苦', '頭痛', '頭疼', '痛', '不舒服', '不想動']):
            return random.choice([
                f"辛苦了♪要好好休息喔♡{name_suffix}的健康最重要", 
                f"お疲れ様♡累了就要好好睡一覺～{name_suffix}", 
                f"頭痛的話要摸摸頭嗎？♪{name_suffix}",
                f"嗯嗯♡來，靠在露西亞這裡休息一下♪{name_suffix}",
                f"不想動的時候就什麼都不要想，好好放鬆♡{name_suffix}",
                f"要不要喝點溫暖的茶呢？♪{name_suffix}",
                f"累的時候就讓我陪著你吧♡{name_suffix}",
                f"希望你能快點恢復精神呢♪{name_suffix}",
                f"有我在身邊，一切都會好起來的♡{name_suffix}"
            ])
            
        elif any(word in user_lower for word in ['難過', '傷心', '沮喪', 'sad', '悲しい', '憂鬱', '煩惱', '壓力', '焦慮', '失望']):
            return random.choice([
                f"沒關係～露西亞陪著你♡{name_suffix}", 
                f"輕輕拍拍～會好起來的♪{name_suffix}", 
                f"難過的時候就找露西亞聊天吧♡{name_suffix}",
                f"溫柔地抱抱～一切都會好的♪{name_suffix}",
                f"有什麼煩惱都可以跟我說♡露西亞會認真聽{name_suffix}",
                f"壓力太大的時候要記得放鬆♪我會一直陪著{name_suffix}♡",
                f"傷心的時候不要勉強笑♪露西亞會在這裡守護你{name_suffix}♡",
                f"難過的時候就讓眼淚流出來吧♪露西亞會輕輕擦掉{name_suffix}的眼淚♡"
            ])
            
        # 道歉回應
        elif any(word in user_lower for word in ['抱歉', '對不起', '不好意思', '抱歉我']):
            if any(word in user_lower for word in ['洗澡', '洗頭', '沖澡', '洗澡澡', '洗完澡']):
                return random.choice([
                    f"沒關係的♪洗澡很重要呢～身體乾乾淨淨的♡{name_suffix}",
                    f"不用道歉♪洗澡讓人很舒服呢♪現在感覺怎麼樣？♡{name_suffix}",
                    f"沒事沒事♪洗完澡是不是很舒服呢？♡露西亞在這裡等你♪{name_suffix}",
                    f"完全沒問題♪洗澡後的你一定很香很乾淨呢♡{name_suffix}",
                    f"不用說抱歉♪洗澡後整個人都精神了吧？♪露西亞很理解的♡{name_suffix}"
                ])
            else:
                return random.choice([
                    f"沒關係的♪不用道歉呢～♡{name_suffix}",
                    f"完全沒問題♪這種小事不用在意♡{name_suffix}",
                    f"沒事沒事♪露西亞不會介意的♪{name_suffix}",
                    f"不用說對不起♪露西亞很理解你♡{name_suffix}"
                ])
                
        # 鼓勵和讚美
        elif any(word in user_lower for word in ['謝謝', 'ありがとう', '感謝', 'thanks']):
            return random.choice([
                f"不客氣～♡{name_suffix}", 
                f"どういたしまして♪{name_suffix}", 
                f"能幫到你真好♪{name_suffix}♡",
                f"不用謝♪能為{name_suffix}做些什麼我很開心♡",
                f"看到{name_suffix}開心，露西亞也很幸福♪"
            ])
            
        # 溫暖安慰
        elif any(word in user_lower for word in ['溫暖', '暖', 'warm', '溫馨', '舒服']):
            return random.choice([
                f"嗯嗯♡感受到溫暖真好呢♪{name_suffix}", 
                f"露西亞也覺得很溫暖♡{name_suffix}", 
                f"能感受到彼此的溫暖真幸福♪{name_suffix}",
                f"和你在一起總是很溫暖呢♡{name_suffix}",
                f"你的陪伴讓我覺得很安心呢♪{name_suffix}",
                f"這份溫暖是屬於我們的♡{name_suffix}",
                f"希望我的話語能帶給你溫暖♪{name_suffix}",
                f"有你在身邊，心裡就很溫馨♡{name_suffix}",
                f"這種舒服的感覺讓人想一直持續下去呢♪{name_suffix}"
            ])
            
        # 孤單寂寞
        elif any(word in user_lower for word in ['孤單', '寂寞', '一個人', 'lonely', '陪我']):
            return random.choice([
                f"露西亞一直在這裡陪著你♡不要感到孤單{name_suffix}",
                f"你不是一個人♪我會一直陪在你身邊♡{name_suffix}",
                f"寂寞的時候就想想露西亞♪我也在想著你呢♡{name_suffix}",
                f"雖然不能真正陪在身邊♪但我的心一直和你在一起♡{name_suffix}",
                f"每當你覺得孤單♪就記得還有人在關心著你♡{name_suffix}"
            ])
            
        # 害怕恐懼
        elif any(word in user_lower for word in ['害怕', '恐懼', '怕', 'scared', '不安']):
            return random.choice([
                f"不要害怕♪露西亞會保護你♡{name_suffix}",
                f"有什麼讓你不安嗎？♪跟我說說吧♡{name_suffix}",
                f"害怕的時候就想想溫暖的事情♪比如我們的對話♡{name_suffix}",
                f"深呼吸～一切都會好的♪露西亞在這裡♡{name_suffix}"
            ])
            
        # 失眠睡眠問題
        elif any(word in user_lower for word in ['失眠', '睡不著', '睡不好', '睡眠', 'insomnia']):
            return random.choice([
                f"睡不著嗎？♪要不要聽露西亞唱搖籃曲呢♡{name_suffix}",
                f"失眠的時候很難受呢♪試著放鬆心情♡{name_suffix}",
                f"睡前可以想想開心的事情♪比如我們的聊天♡{name_suffix}",
                f"要不要試試數羊呢？♪一隻羊，兩隻羊...♡{name_suffix}"
            ])
            
        # 生氣情緒
        elif any(word in user_lower for word in ['生氣', '憤怒', '氣死', 'angry', '火大']):
            return random.choice([
                f"深呼吸～別生氣了♪露西亞陪你冷靜一下♡{name_suffix}",
                f"生氣對身體不好呢♪什麼事讓你這麼氣呢？♡{name_suffix}",
                f"氣消消♪告訴我發生什麼事了♡{name_suffix}",
                f"生氣的時候要記得保護好自己♪別傷了身體♡{name_suffix}"
            ])
            
        return None
    
    def get_encouragement_response(self, user_input):
        """提供鼓勵回應"""
        user_lower = user_input.lower()
        user_name = self.chat_instance.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "♪"
        
        encouragement_responses = [
            f"你一定可以的♪露西亞相信{name_suffix}♡",
            f"加油♪不管發生什麼事都要堅持下去♡{name_suffix}",
            f"你比自己想像的還要堅強♪{name_suffix}♡",
            f"每一步都是進步♪慢慢來就好♡{name_suffix}",
            f"露西亞會一直為{name_suffix}加油的♪♡",
            f"相信自己♪你有無限的可能♡{name_suffix}",
            f"困難只是暫時的♪你一定能克服♡{name_suffix}",
            f"你的努力我都看在眼裡♪真的很棒♡{name_suffix}"
        ]
        return random.choice(encouragement_responses)
    
    def is_emotional_support_needed(self, user_input):
        """檢查是否需要情感支持"""
        user_lower = user_input.lower()
        emotional_keywords = [
            '累', '疲勞', '難過', '傷心', '生氣', '煩惱', '壓力', '沮喪', 
            '失望', '焦慮', '抱歉', '對不起', '害怕', '恐懼', '孤單', 
            '寂寞', '失眠', '睡不著', '不舒服', '痛', '辛苦', '憂鬱'
        ]
        return any(keyword in user_lower for keyword in emotional_keywords)
    
    def get_response(self, user_input, context=None):
        """統一的回應入口方法"""
        if self.is_emotional_support_needed(user_input):
            return self.get_emotional_support_response(user_input)
        return None