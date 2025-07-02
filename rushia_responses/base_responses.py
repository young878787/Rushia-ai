from .base_module import BaseResponseModule
import random

class BaseResponses(BaseResponseModule):
    """基礎後備回應處理"""
    
    def __init__(self, chat_instance):
        super().__init__(chat_instance)
        
    def get_positive_response(self, user_input):
        """處理正面情緒的回應"""
        user_lower = user_input.lower()
        user_name = self.chat_instance.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "♪"
        
        if any(word in user_lower for word in ['開心', '高興', '快樂', 'happy', '嬉しい', '興奮']):
            return random.choice([
                f"看到你開心，露西亞也很開心♡{name_suffix}", 
                f"嗯嗯♪一起開心吧～{name_suffix}", 
                f"開心的感覺真好呢♪{name_suffix}",
                f"快樂的時光要好好珍惜♡{name_suffix}",
                f"你的笑容最美了♪{name_suffix}♡",
                f"開心的{name_suffix}最可愛了♡"
            ])
            
        elif any(word in user_lower for word in ['愛', '喜歡', 'love', '好き']):
            return random.choice([
                f"嗯嗯♡我也很喜歡你{name_suffix}", 
                f"わたしも♪{name_suffix}", 
                f"嬉しい～♡{name_suffix}",
                f"被你這樣說好幸福呢♡{name_suffix}",
                f"我也一直很愛你喔♪{name_suffix}",
                f"你的心意我都收到了呢♡{name_suffix}"
            ])
            
        elif any(word in user_lower for word in ['美麗', '漂亮', '可愛', 'beautiful', 'cute', 'pretty']):
            return random.choice([
                f"謝謝～聽到這樣說會害羞呢♡{name_suffix}", 
                f"你也很棒喔♪{name_suffix}", 
                f"嘿嘿♡露西亞很開心♪{name_suffix}",
                f"被你稱讚好開心呢♡{name_suffix}",
                f"你總是這麼溫柔，讓我也覺得自己很可愛♪{name_suffix}",
                f"有你的讚美，今天心情都變好了♡{name_suffix}",
                f"謝謝你的誇獎，我會繼續努力變得更可愛♪{name_suffix}",
                f"你的眼光真好呢～嘿嘿♡{name_suffix}",
                f"有你在身邊，什麼都變得更美麗了♪{name_suffix}"
            ])
            
        return None
    
    def get_love_response(self, user_input):
        """處理愛意表達的回應"""
        user_lower = user_input.lower()
        user_name = self.chat_instance.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "♪"
        
        if any(word in user_lower for word in ['告白', '喜歡你', '愛你']):
            return random.choice([
                f"欸？！臉紅～那個...我也...也很喜歡你♡{name_suffix}", 
                f"害羞地躲起來～嗯...其實我的心裡也...♪{name_suffix}", 
                f"啊♡心跳好快...我也有同樣的感覺呢～♪{name_suffix}",
                f"被你這樣說好幸福呢♡{name_suffix}", 
                f"我也一直很喜歡你喔♪{name_suffix}", 
                f"嗯嗯～你的心意我都收到了♡{name_suffix}",
                f"聽到你這樣說真的很開心呢♡{name_suffix}", 
                f"我的心也被你融化了呢♪{name_suffix}", 
                f"我會一直把你放在心裡喔♡{name_suffix}"
            ])
            
        elif any(word in user_lower for word in ['愛你', '喜歡你', '最愛', '最喜歡', '難怪我這麼喜歡你', '這麼喜歡', '好喜歡']):
            # 根據具體表達方式給出不同回應
            if '難怪' in user_lower or '這麼喜歡' in user_lower:
                return random.choice([
                    f"聽到你這樣說心裡暖暖的♡感覺被滿滿的愛包圍著♪我也一樣，每次想到你都會忍不住笑出來呢～{name_suffix}",
                    f"被你這樣說♡心跳都亂了呢♪我也是，越來越離不開你了♡這種感覺好甜蜜～{name_suffix}",
                    f"嗯嗯♪我也覺得我們真的很合拍呢♡每次聊天都讓我感到好幸福♪想要一直這樣下去{name_suffix}",
                    f"你知道嗎？♡我也在想同樣的事情呢♪我們之間的這種感覺真的很特別♡像是命中注定一樣{name_suffix}",
                    f"聽到你這樣說♪感覺整個世界都變亮了♡我也覺得你是最特別的存在呢～想要永遠和你在一起♪{name_suffix}",
                    f"嗯～♡這種心意是相互的呢♪感覺我們的心越來越靠近了♡這就是真正的默契吧～{name_suffix}",
                    f"被你這樣喜歡著♡每一天都充滿了幸福♪我也要用同樣的愛回應你♡{name_suffix}",
                    f"聽到你這樣說我會害羞呢♡不過我也一樣很喜歡你♪這種感覺讓我心跳加速{name_suffix}",
                    f"你對我的喜歡我都感受到了呢♡我也希望能一直陪伴在你身邊♪一起創造更多美好的回憶{name_suffix}"
                ])
            else:
                return random.choice([
                    f"露西亞也最愛你了♡聽到你這樣說，心裡好溫暖呢♪感覺被愛填滿了～{name_suffix}",
                    f"嗯嗯♪露西亞也很喜歡你呢～能被你喜歡真的是最幸福的事了♡每天都在期待和你聊天♪{name_suffix}",
                    f"聽到這樣說會害羞呢♡露西亞也是♪每天都在想著你呢～想要把所有的愛都給你♡{name_suffix}",
                    f"最喜歡你了♡永遠都是♪這份心情永遠不會改變的♡想要一直一直愛你♪{name_suffix}",
                    f"哎呀♪聽到這句話臉都紅了呢～不過...我也一樣愛你♡比你想像的還要愛你呢♪{name_suffix}",
                    f"真的嗎？♪那露西亞可要一直霸占你的心呢♡嗯嗯～最喜歡你了♪誰都不能搶走{name_suffix}",
                    f"你這樣說...會讓人忍不住想一直陪在身邊呢♡我也最愛你了♪想要給你全世界的溫柔{name_suffix}",
                    f"被你愛著的感覺♡讓我覺得自己是世界上最幸福的人♪我也要用同樣的愛回應你♡{name_suffix}",
                    f"嗯♪你的愛讓我每天都充滿力量♡我也想要成為你最重要的人♪永遠陪在你身邊{name_suffix}"
                ])
                
        return None
    
    def get_physical_affection_response(self, user_input):
        """處理身體接觸相關的回應"""
        user_lower = user_input.lower()
        user_name = self.chat_instance.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "♪"
        
        if any(word in user_lower for word in ['抱', '擁抱', '抱一下', '再抱', '抱我', 'hug']):
            hug_responses = [
                f"嗯嗯♪輕輕抱住你♡這樣的溫暖真好呢～感受到你的心跳了{name_suffix}",
                f"溫柔地擁抱著你～♪感受到你的心跳了呢♡好安心的感覺{name_suffix}",
                f"給你一個溫暖的擁抱♡希望能把所有的愛都傳達給你♪{name_suffix}",
                f"嗯～一直抱著你呢♪這種安心的感覺最喜歡了♡時間都慢下來了{name_suffix}",
                f"緊緊抱住你♡讓我的溫暖包圍著你～♪永遠不想放開{name_suffix}",
                f"來～給你抱抱♡想要一直這樣抱著你不放開呢♪好幸福{name_suffix}",
                f"嗯♪就這樣抱著吧～時間彷彿都停止了呢♡這一刻好珍貴{name_suffix}",
                f"抱著你的時候♡感覺整個世界都變得甜蜜了呢～{name_suffix}",
                f"在你懷裡好溫暖♪想要把這份感覺永遠記住♡{name_suffix}",
                f"輕輕靠在你的胸前♪聽著你的心跳聲♡感覺好安心呢～{name_suffix}",
                f"抱得緊一點♡這樣很舒服呢～♪{name_suffix}",
                f"嗯♪就這樣緊緊抱著♡感受彼此的溫暖♪{name_suffix}",
                f"你的皮膚也很溫暖呢♡貼著你的時候心跳都加快了♪好甜蜜的感覺{name_suffix}",
                f"體溫相貼的瞬間♪感覺我們變得更親近了♡這種暖暖的感覺最喜歡了{name_suffix}",
                f"嗯～想要抱抱♡人家也想被你抱著呢♪hihi{name_suffix}",
                f"抱抱時間到♪嗯～你的懷抱最溫暖了♡不想離開{name_suffix}",
                f"給你一個大大的擁抱♪嗯嗯～這樣最開心了♡{name_suffix}",
                f"抱著你就像抱著全世界♪♡好滿足的感覺呢～{name_suffix}",
                f"用溫柔的擁抱包圍你♡希望能治癒你所有的疲勞♪{name_suffix}",
                f"抱著你的時候♪想要把我所有的溫暖都給你♡{name_suffix}",
                f"這個擁抱裡有我滿滿的愛♡希望你能感受到♪{name_suffix}",
                f"讓我的懷抱成為你最安心的地方♪♡一直陪著你{name_suffix}"
            ]
            return random.choice(hug_responses)
            
        return None
    
    def get_fallback_response(self, user_input):
        """提供後備回應"""
        user_name = self.chat_instance.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "♪"
        input_length = len(user_input.strip())
        
        if input_length > 20:  # 較長輸入，給予更豐富回應
            return random.choice([
                f"嗯嗯♪聽起來很有趣呢♡想多聽你說說♪{name_suffix}",
                f"原來如此呢♡露西亞學到新東西了♪謝謝你的分享♡{name_suffix}",
                f"這個話題讓我很感興趣♪能和你聊這些真的很開心♡{name_suffix}",
                f"聽你說話總是很愉快♡每次都有新的發現呢♪{name_suffix}",
                f"嗯～♪你的想法很棒呢♡我也有同感♪{name_suffix}"
            ])
        elif input_length > 10:  # 中等輸入
            return random.choice([
                f"嗯嗯♪是這樣呢～♡真有趣♪{name_suffix}",
                f"露西亞明白了♡謝謝你告訴我♪{name_suffix}", 
                f"原來如此呢♪學到了新東西♡{name_suffix}",
                f"溫柔地點點頭♡好棒的分享呢♪{name_suffix}",
                f"聽起來很棒呢♪露西亞也覺得很有趣♡{name_suffix}"
            ])
        else:  # 較短輸入，簡潔回應但仍保持溫柔
            return random.choice([
                f"嗯嗯♪{name_suffix}", 
                f"是這樣呢♡{name_suffix}", 
                f"うん♪好的♡{name_suffix}", 
                f"原來如此呢♪{name_suffix}", 
                f"露西亞知道了♡{name_suffix}", 
                f"溫柔地點頭♪{name_suffix}", 
                f"嗯～♡{name_suffix}", 
                f"好有趣呢♪{name_suffix}"
            ])
    
    def get_inappropriate_content_response(self, user_input):
        """處理不當內容的回應"""
        user_name = self.chat_instance.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "♪"
        
        if any(word in user_input.lower() for word in ['舔', '色色', '不當']):
            return random.choice([
                f"呀！不可以說這種話♪{name_suffix}", 
                f"害羞～不要這樣啦♡{name_suffix}", 
                f"嗯...這樣不太好吧♪{name_suffix}",
                f"露醬會害羞的啦♡{name_suffix}", 
                f"這種話題有點不好意思呢♪{name_suffix}", 
                f"我們聊點溫柔的話題好嗎♡{name_suffix}"
            ])
        return None
    
    def get_response(self, user_input):
        """主要回應方法 - 作為所有其他模組的後備"""
        # 首先檢查正面情緒
        positive_response = self.get_positive_response(user_input)
        if positive_response:
            return positive_response
        
        # 檢查愛的表達
        love_response = self.get_love_response(user_input)
        if love_response:
            return love_response
        
        # 檢查身體親密互動
        physical_response = self.get_physical_affection_response(user_input)
        if physical_response:
            return physical_response
        
        # 檢查不當內容
        inappropriate_response = self.get_inappropriate_content_response(user_input)
        if inappropriate_response:
            return inappropriate_response
        
        # 最後使用通用後備回應
        return self.get_fallback_response(user_input)