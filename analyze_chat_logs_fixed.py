import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

class ChatLogAnalyzer:
    def __init__(self, chat_log_dir="D:/RushiaMode/chat_log"):
        self.chat_log_dir = Path(chat_log_dir)
    
    def analyze_all_chats(self):
        """分析所有聊天記錄"""
        stats = {
            "total_streams": 0,
            "total_messages": 0,
            "unique_users": set(),
            "message_patterns": Counter(),
            "time_distribution": defaultdict(int),
            "popular_phrases": Counter(),
            "emoji_usage": Counter()
        }
        
        print("開始分析聊天記錄...")
        
        for chat_dir in self.chat_log_dir.iterdir():
            if not chat_dir.is_dir():
                continue
            
            stats["total_streams"] += 1
            print(f"分析直播: {chat_dir.name}")
            
            stream_stats = self._analyze_single_stream(chat_dir)
            self._merge_stats(stats, stream_stats)
        
        # 處理統計結果
        stats["unique_users"] = len(stats["unique_users"])
        
        return stats
    
    def analyze_individual_streams(self):
        """批量分析每個直播資料夾，為每個生成獨立的JSON報告"""
        print("開始批量分析各個直播資料夾...")
        
        output_dir = Path("D:/RushiaMode/analysis/individual_reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        global_summary = {
            "analysis_date": datetime.now().isoformat(),
            "total_streams_analyzed": 0,
            "streams": []
        }
        
        for chat_dir in self.chat_log_dir.iterdir():
            if not chat_dir.is_dir():
                continue
            
            folder_name = chat_dir.name
            print(f"分析直播資料夾: {folder_name}")
            
            # 分析單個直播
            stream_stats = self._analyze_single_stream(chat_dir)
            
            # 生成美化的JSON報告
            report_data = self._generate_json_report(folder_name, stream_stats)
            
            # 保存為 chat_analysis_report_{資料夾名}.json
            safe_folder_name = self._safe_filename(folder_name)
            report_filename = f"chat_analysis_report_{safe_folder_name}.json"
            report_path = output_dir / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            print(f"  └─ 報告已保存: {report_filename}")
            
            # 添加到全局總結
            global_summary["streams"].append({
                "folder_name": folder_name,
                "report_file": report_filename,
                "total_messages": stream_stats["messages"],
                "unique_users": len(stream_stats["users"]),
                "top_phrases": dict(stream_stats["phrases"].most_common(5))
            })
            
            global_summary["total_streams_analyzed"] += 1
        
        # 保存全局總結
        summary_path = output_dir / "global_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(global_summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n批量分析完成！")
        print(f"共分析了 {global_summary['total_streams_analyzed']} 個直播資料夾")
        print(f"個別報告保存在: {output_dir}")
        print(f"全局總結保存在: {summary_path}")
        
        return global_summary
    
    def _safe_filename(self, filename):
        """將資料夾名稱轉換為安全的檔案名稱"""
        # 移除或替換不安全的字符
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
        safe_name = re.sub(r'[\s\u3000]+', '_', safe_name)  # 替換空格和全角空格
        safe_name = safe_name.strip('_')
        
        # 限制長度
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        
        return safe_name
    
    def _generate_json_report(self, folder_name, stream_stats):
        """生成美化的JSON報告數據"""
        report = {
            "stream_info": {
                "folder_name": folder_name,
                "analysis_date": datetime.now().isoformat(),
                "duration_analyzed": "未知"  # 可以後續計算
            },
            "statistics": {
                "total_messages": stream_stats["messages"],
                "unique_users": len(stream_stats["users"]),
                "average_message_length": stream_stats.get("avg_length", 0),
                "message_rate": {
                    "messages_per_minute": 0,  # 可以後續計算
                    "peak_activity_period": "未知"
                }
            },
            "user_engagement": {
                "top_chatters": dict(stream_stats["users"].most_common(10)),
                "user_participation_rate": len(stream_stats["users"]) / max(stream_stats["messages"], 1),
                "engagement_level": self._calculate_engagement_level(stream_stats)
            },
            "content_analysis": {
                "popular_phrases": dict(stream_stats["phrases"].most_common(20)),
                "emoji_usage": dict(stream_stats["emojis"].most_common(15)),
                "message_types": dict(stream_stats["types"]),
                "language_patterns": self._analyze_language_patterns(stream_stats)
            },
            "interaction_patterns": {
                "questions_to_rushia": stream_stats.get("questions", []),
                "support_messages": stream_stats.get("support", []),
                "reactions": stream_stats.get("reactions", [])
            },
            "training_recommendations": {
                "high_value_interactions": self._identify_training_data(stream_stats),
                "conversation_themes": self._extract_themes(stream_stats),
                "response_examples": self._generate_response_examples(stream_stats)
            }
        }
        
        return report
    
    def _analyze_single_stream(self, stream_dir):
        """分析單個直播的聊天記錄"""
        stream_stats = {
            "messages": 0,
            "users": Counter(),
            "phrases": Counter(),
            "emojis": Counter(),
            "types": Counter(),
            "questions": [],
            "support": [],
            "reactions": [],
            "total_length": 0
        }
        
        # 搜索所有可能的聊天記錄文件
        chat_files = list(stream_dir.glob("*.txt")) + list(stream_dir.glob("*.json")) + list(stream_dir.glob("*.log"))
        
        for chat_file in chat_files:
            try:
                if chat_file.suffix == '.json':
                    self._process_json_chat(chat_file, stream_stats)
                else:
                    self._process_text_chat(chat_file, stream_stats)
            except Exception as e:
                print(f"處理文件 {chat_file} 時出錯: {e}")
        
        # 計算平均值
        if stream_stats["messages"] > 0:
            stream_stats["avg_length"] = stream_stats["total_length"] / stream_stats["messages"]
        
        return stream_stats
    
    def _process_json_chat(self, file_path, stats):
        """處理JSON格式的聊天記錄"""
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        self._process_chat_message(item, stats)
                elif isinstance(data, dict) and 'messages' in data:
                    for msg in data['messages']:
                        self._process_chat_message(msg, stats)
            except json.JSONDecodeError:
                print(f"JSON格式錯誤: {file_path}")
    
    def _process_text_chat(self, file_path, stats):
        """處理文本格式的聊天記錄"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 簡單的文本解析，假設格式為 "用戶名: 消息"
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            username = parts[0].strip()
                            message = parts[1].strip()
                            self._process_chat_message({
                                'username': username,
                                'message': message
                            }, stats)
    
    def _process_chat_message(self, msg, stats):
        """處理單條聊天消息"""
        if not isinstance(msg, dict):
            return
        
        username = msg.get('username', msg.get('user', ''))
        message = msg.get('message', msg.get('text', ''))
        
        if not username or not message:
            return
        
        stats["messages"] += 1
        stats["users"][username] += 1
        stats["total_length"] += len(message)
        
        # 分析消息類型
        if self._is_question(message):
            stats["types"]["問題"] += 1
            stats["questions"].append({"user": username, "message": message})
        elif self._is_support(message):
            stats["types"]["支持"] += 1
            stats["support"].append({"user": username, "message": message})
        elif self._is_reaction(message):
            stats["types"]["反應"] += 1
            stats["reactions"].append({"user": username, "message": message})
        else:
            stats["types"]["一般"] += 1
        
        # 提取短語和表情符號
        phrases = self._extract_phrases(message)
        emojis = self._extract_emojis(message)
        
        stats["phrases"].update(phrases)
        stats["emojis"].update(emojis)
    
    def _is_question(self, message):
        """判斷是否為問題"""
        question_indicators = ['？', '?', '什麼', '怎麼', '為什麼', '哪裡', '誰', '什麼時候']
        return any(indicator in message for indicator in question_indicators)
    
    def _is_support(self, message):
        """判斷是否為支持消息"""
        support_indicators = ['加油', '支持', '喜歡', '愛', '♥', '❤', '頑張', 'がんばれ', 'かわいい']
        return any(indicator in message for indicator in support_indicators)
    
    def _is_reaction(self, message):
        """判斷是否為反應消息"""
        reaction_indicators = ['www', 'wwww', '草', 'lol', 'LOL', '哈哈', '笑', 'XD']
        return any(indicator in message for indicator in reaction_indicators)
    
    def _extract_phrases(self, message):
        """提取有意義的短語"""
        # 移除標點符號，分割成詞
        words = re.findall(r'[\w\u4e00-\u9fff]+', message)
        phrases = []
        
        # 提取2-4字的短語
        for i in range(len(words)):
            for length in range(2, min(5, len(words) - i + 1)):
                phrase = ''.join(words[i:i+length])
                if len(phrase) >= 2:
                    phrases.append(phrase)
        
        return phrases
    
    def _extract_emojis(self, message):
        """提取表情符號"""
        emoji_pattern = re.compile(
            '[\U0001F600-\U0001F64F'  # 表情符號
            '\U0001F300-\U0001F5FF'  # 符號和圖標
            '\U0001F680-\U0001F6FF'  # 交通和地圖符號
            '\U0001F1E0-\U0001F1FF'  # 國旗
            ']+', 
            flags=re.UNICODE
        )
        return emoji_pattern.findall(message)
    
    def _calculate_engagement_level(self, stats):
        """計算互動參與度"""
        if stats["messages"] == 0:
            return "無"
        
        unique_users = len(stats["users"])
        messages_per_user = stats["messages"] / unique_users
        
        if messages_per_user > 10:
            return "非常高"
        elif messages_per_user > 5:
            return "高"
        elif messages_per_user > 2:
            return "中等"
        else:
            return "低"
    
    def _analyze_language_patterns(self, stats):
        """分析語言模式"""
        total_messages = stats["messages"]
        if total_messages == 0:
            return {}
        
        return {
            "問題比例": (stats["types"].get("問題", 0) / total_messages) * 100,
            "支持比例": (stats["types"].get("支持", 0) / total_messages) * 100,
            "反應比例": (stats["types"].get("反應", 0) / total_messages) * 100,
            "一般比例": (stats["types"].get("一般", 0) / total_messages) * 100
        }
    
    def _identify_training_data(self, stats):
        """識別高價值的訓練數據"""
        high_value = []
        
        # 選擇高頻用戶的優質互動
        top_users = stats["users"].most_common(10)
        for user, count in top_users:
            if count >= 5:  # 至少5條消息的用戶
                high_value.append({
                    "type": "高頻用戶互動",
                    "user": user,
                    "message_count": count,
                    "training_value": "高"
                })
        
        # 添加問題和回應
        for q in stats["questions"][:10]:  # 前10個問題
            high_value.append({
                "type": "用戶問題",
                "content": q["message"],
                "training_value": "中"
            })
        
        return high_value
    
    def _extract_themes(self, stats):
        """提取對話主題"""
        themes = []
        top_phrases = stats["phrases"].most_common(20)
        
        for phrase, count in top_phrases:
            if count >= 3:  # 至少出現3次
                themes.append({
                    "theme": phrase,
                    "frequency": count,
                    "relevance": "高" if count >= 10 else "中"
                })
        
        return themes
    
    def _generate_response_examples(self, stats):
        """生成回應範例"""
        examples = []
        
        # 基於常見短語生成範例
        common_phrases = stats["phrases"].most_common(10)
        for phrase, count in common_phrases:
            if count >= 5:                examples.append({
                    "trigger": phrase,
                    "suggested_response": f"關於{phrase}，るしあ會...",
                    "frequency": count
                })
        
        return examples[:5]  # 返回前5個範例
    
    def _merge_stats(self, global_stats, stream_stats):
        """合併統計數據"""
        global_stats["total_messages"] += stream_stats["messages"]
        global_stats["unique_users"].update(stream_stats["users"].keys())
        global_stats["popular_phrases"].update(stream_stats["phrases"])
        global_stats["emoji_usage"].update(stream_stats["emojis"])
    
    def generate_report(self, stats, stream_name="全部"):
        """生成分析報告"""
        report = f"""
=== 聊天記錄分析報告 ({stream_name}) ===
分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

基本統計:
- 總直播數: {stats.get('total_streams', 1)}
- 總消息數: {stats['total_messages']}
- 獨特用戶數: {stats['unique_users']}
- 平均每直播消息數: {stats['total_messages'] // max(stats.get('total_streams', 1), 1)}

熱門短語 (前20):
"""
        
        for phrase, count in stats['popular_phrases'].most_common(20):
            report += f"- {phrase}: {count}次\n"
        
        report += "\n表情符號使用 (前10):\n"
        for emoji, count in stats['emoji_usage'].most_common(10):
            report += f"- {emoji}: {count}次\n"
        
        report += "\n=== 訓練數據建議 ===\n"
        report += self.generate_training_suggestions(stats)
        
        return report
    
    def generate_training_suggestions(self, stats):
        """生成訓練建議"""
        suggestions = []
        
        # 基於熱門短語的建議
        top_phrases = stats['popular_phrases'].most_common(10)
        suggestions.append("高頻互動短語 (建議用於訓練):")
        
        for phrase, count in top_phrases:
            if count >= 5:  # 至少出現5次才建議
                suggestion_type = "高優先級" if count >= 20 else "中優先級"
                suggestions.append(f"- {phrase} ({count}次) - {suggestion_type}")
        
        # 角色扮演建議
        suggestions.append("\n建議的角色扮演情境:")
        suggestions.append("- 回應粉絲的支持和鼓勵")
        suggestions.append("- 對遊戲失敗的可愛反應")
        suggestions.append("- 與粉絲的日常對話")
        suggestions.append("- ASMR相關的溫柔回應")
        
        return "\n".join(suggestions)

def main():
    """主程序"""
    analyzer = ChatLogAnalyzer()
    
    print("選擇分析模式:")
    print("1. 分析所有聊天記錄 (整體統計)")
    print("2. 批量分析每個直播資料夾 (生成個別JSON報告)")
    
    choice = input("請輸入選項 (1 或 2): ").strip()
    
    if choice == "1":
        # 整體分析
        stats = analyzer.analyze_all_chats()
        report = analyzer.generate_report(stats)
        
        # 保存報告
        output_path = Path("D:/RushiaMode/analysis/chat_analysis_report.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n報告已保存到: {output_path}")
        print("=" * 50)
        print(report)
    
    elif choice == "2":
        # 批量分析
        global_summary = analyzer.analyze_individual_streams()
        print(f"\n批量分析完成！共分析了 {global_summary['total_streams_analyzed']} 個直播")
    
    else:
        print("無效選項，程序退出")

if __name__ == "__main__":
    main()
