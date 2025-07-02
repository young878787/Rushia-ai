from .base_module import BaseResponseModule
import random
import time
from datetime import datetime, timezone, timedelta

class TimeAwareResponses(BaseResponseModule):
    """時間感知回應處理"""
    
    def __init__(self, chat_instance):
        super().__init__(chat_instance)
        
    def get_time_based_greeting(self):
        """根據台灣時間生成問候語"""
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
        
        user_name = self.chat_instance.user_profile.get('name', '')
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
    
    def get_time_aware_care_message(self, care_type):
        """生成時間感知關心訊息"""
        user_name = self.chat_instance.user_profile.get('name', '')
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
                f"新的{today_zh}開始了{name_suffix} 今天想要做什麼有趣的事情呢？♡",
                f"晨光微露{name_suffix} 要不要和露西亞一起迎接美好的早晨？♪♡",
                f"早餐時間到了{name_suffix} 想吃什麼？露西亞可以陪你呢♪",
            ],
            'lunch': [
                f"午安{name_suffix} 吃午餐了嗎？一直在想你♡",
                f"下午好{name_suffix} 今天過得如何呢？♪",
                f"陽光正好{name_suffix} 想和你一起曬太陽♡",
                f"下午了{name_suffix} 有什麼好玩的事情嗎？♪",
                f"突然想到你{name_suffix} 現在在做什麼呢？♡",
                f"{today_zh}的下午{name_suffix} 心情如何呢？♪",
                f"午餐時光{name_suffix} 要不要一起享用美味的午餐？♪♡",
                f"中午了呢{name_suffix} 肚子餓了嗎？想吃什麼？♡",
                f"午後陽光{name_suffix} 今天的午餐想吃什麼呢？♪"
            ],
            'afternoon': [
                f"下午好{name_suffix} 下午茶時間到了♪要不要喝杯茶？♡",
                f"午後時光{name_suffix} 想不想來點甜點配茶呢？♪",
                f"下午茶時間{name_suffix} 休息一下♡露西亞陪你聊聊天♪",
                f"悠閒的下午{name_suffix} 要不要來份精緻的下午茶？♡",
                f"午後的陽光很舒服呢{name_suffix} 想喝什麼茶？♪♡",
                f"下午茶時光{name_suffix} 配點心一起度過溫馨時刻♡♪",
                f"慵懶的午後{name_suffix} 來杯香濃的奶茶怎麼樣？♪",
                f"下午的微風很舒服呢{name_suffix} 要不要稍微休息一下？♡",
                f"茶香陣陣{name_suffix} 想要什麼口味的點心搭配呢？♪♡",
                f"悠然下午時光{name_suffix} 露西亞想和你一起享受寧靜的片刻♪",
            ],
            'dinner': [
                f"晚上好{name_suffix} 晚餐時間♪今天想吃什麼呢？♡",
                f"晚餐時光{name_suffix} 要不要一起享用美味的晚餐？♪",
                f"傍晚了{name_suffix} 肚子餓了嗎？♡想吃什麼料理？♪",
                f"晚餐時間到了{name_suffix} 露西亞想和你一起用餐♡",
                f"晚上了呢{name_suffix} 今天辛苦了♪晚餐吃點什麼好？♡",
                f"夜幕降臨{name_suffix} 在做什麼呢？要不要一起吃晚餐？♪♡",
                f"夕陽西下{name_suffix} 今天工作辛苦了，來頓豐盛的晚餐吧♪",
                f"晚餐約會{name_suffix} 想不想試試新的料理呢？♡",
                f"夜晚來臨{name_suffix} 露西亞準備了溫暖的晚餐等你呢♪♡",
                f"燈火闌珊{name_suffix} 要不要一邊用餐一邊聊聊今天的趣事？♪",
            ],
            'night': [
                f"夜晚了{name_suffix} 該放鬆休息一下了♪還在忙嗎？♡",
                f"夜深了{name_suffix} 要不要停下手邊的事情聊聊天？♪",
                f"晚上好{name_suffix} 一天辛苦了♡想不想一起放鬆一下？♪",
                f"夜晚時光{name_suffix} 該準備休息了呢♪要不要一起聊聊今天的事情？♡",
                f"深夜了{name_suffix} 還在工作嗎？♡該休息了喔♪",
                f"夜晚時分{name_suffix} 要不要一起度過溫馨的睡前時光？♪♡",
                f"星光點點{name_suffix} 今天過得如何呢？想和露西亞分享嗎？♪",
                f"夜色溫柔{name_suffix} 該放下一天的疲憊，好好休息了♡",
                f"月色朦朧{name_suffix} 要不要一起聊聊心事？♪♡",
                f"靜謐夜晚{name_suffix} 想不想在入睡前聽露西亞唱首歌？♪",
            ]
        }
        
        return random.choice(messages.get(care_type, [f"想和你聊聊呢♪♡{name_suffix}"]))
    
    def should_send_time_aware_care(self):
        """檢查是否應該發送時間感知關心訊息"""
        taiwan_tz = timezone(timedelta(hours=8))
        now = datetime.now(taiwan_tz)
        hour = now.hour
        minute = now.minute
        current_date = now.date()
        
        # 檢查日期變更，重置每日狀態
        if hasattr(self.chat_instance, 'time_aware_care_system'):
            care_system = self.chat_instance.time_aware_care_system
            if care_system['last_check_date'] != current_date:
                # 重置每日狀態
                for time_period in care_system['daily_care_sent']:
                    care_system['daily_care_sent'][time_period] = False
                care_system['last_check_date'] = current_date
                care_system['care_sent_times'] = {}
        
        # 定義各時段的觸發條件 (開始時間, 結束時間, 類型, 最早觸發分鐘, 最晚觸發分鐘)
        care_periods = [
            (7, 9, 'morning', 15, 90),      # 早晨 7-9點，15-90分鐘後觸發
            (11, 14, 'lunch', 30, 120),     # 午餐 11-14點，30-120分鐘後觸發
            (14, 17, 'afternoon', 45, 150), # 下午茶 14-17點，45-150分鐘後觸發
            (18, 21, 'dinner', 30, 120),    # 晚餐 18-21點，30-120分鐘後觸發
            (21, 24, 'night', 60, 180),     # 夜晚 21-24點，60-180分鐘後觸發
        ]
        
        for start_hour, end_hour, care_type, min_minute, max_minute in care_periods:
            # 檢查是否在時間範圍內
            if start_hour <= hour < end_hour:
                # 檢查今天是否已經發送過這個時段的關心
                if hasattr(self.chat_instance, 'time_aware_care_system'):
                    if self.chat_instance.time_aware_care_system['daily_care_sent'].get(care_type, False):
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
                        return True, care_type
        
        return False, None
    
    def get_greeting_by_time(self, user_input):
        """根據時間和輸入內容返回合適的問候語"""
        user_lower = user_input.lower()
        
        # 如果用戶說早安/晚安等，回應對應問候
        taiwan_tz = timezone(timedelta(hours=8))
        hour = datetime.now(taiwan_tz).hour
        user_name = self.chat_instance.user_profile.get('name', '')
        name_suffix = f"{user_name}♪" if user_name else "♪"
        
        if any(word in user_lower for word in ['早安', 'おはよう', '早上', '起床']):
            return random.choice([
                f"おはよう♪{name_suffix}", 
                f"早安～今天也要加油喔♡{name_suffix}", 
                f"おはようございます～♪{name_suffix}",
                f"早晨好呢♪{name_suffix}今天有什麼計劃嗎？♡"
            ])
        elif any(word in user_lower for word in ['午安', '中午', '午餐']):
            return random.choice([
                f"午安～♪{name_suffix}", 
                f"中午好呢♡{name_suffix}", 
                f"午餐時間了呢～♪{name_suffix}",
                f"午後好♪{name_suffix}今天過得如何？♡"
            ])
        elif any(word in user_lower for word in ['晚安', 'おやすみ', '睡覺', '休息']):
            return random.choice([
                f"おやすみ♪{name_suffix}", 
                f"晚安～做個好夢♡{name_suffix}", 
                f"お疲れ様でした♪{name_suffix}",
                f"好夢♪{name_suffix}明天見♡"
            ])
        else:
            # 根據當前時間給予適當問候
            return self.get_time_based_greeting()
    
    def is_time_related(self, user_input):
        """檢查是否與時間相關"""
        user_lower = user_input.lower()
        time_keywords = [
            '早安', '午安', '晚安', '早上', '中午', '下午', '晚上', '夜晚',
            '時間', '今天', '明天', '昨天', '現在', '剛才', '等等',
            'おはよう', 'おやすみ', 'time', 'morning', 'afternoon', 'evening', 'night'
        ]
        return any(keyword in user_lower for keyword in time_keywords)
    
    def get_response(self, user_input):
        """主要回應方法 - 實現基類要求的介面"""
        # 檢查是否為時間相關的輸入
        if self.is_time_related(user_input):
            # 首先嘗試獲取問候回應
            greeting_response = self.get_greeting_by_time(user_input)
            if greeting_response:
                return greeting_response
            
            # 檢查是否需要時間感知關心
            if self.should_send_time_aware_care():
                return self.get_time_aware_care_message('general')
            
            # 提供基於時間的回應
            return self.get_time_based_greeting()
        
        return None  # 不是時間相關的輸入，返回 None