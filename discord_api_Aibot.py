#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
露西亞 Discord AI Bot - 高效能版本
基於現有的 RushiaLoRAChat 系統，提供穩定的 Discord 整合
專注於低延遲和穩定傳輸
"""

import discord
from discord.ext import commands, tasks
import asyncio
import threading
import queue
import time
import logging
import json
import os
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import traceback
import re
import concurrent.futures

# 導入新的聊天系統
try:
    from chat_asmr import RushiaLoRAChat
except ImportError:
    print("❌ 無法導入 chat_asmr 模組，請確保檔案在同一目錄")
    exit(1)

# 載入環境變數
try:
    from dotenv import load_dotenv
    load_dotenv()  # 載入 .env 檔案
except ImportError:
    print("💡 建議安裝 python-dotenv: pip install python-dotenv")
    pass

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('discord_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DiscordAIBot(commands.Bot):
    """露西亞 Discord AI Bot 主類"""
    
    def __init__(self):
        # Discord Bot 設定 - 相容所有版本的 Intents
        intents = discord.Intents.default()
        intents.message_content = True  # 讀取消息內容 (必須)
        
        # 嘗試設定各種消息相關的 intents
        try:
            intents.messages = True         # 新版本
        except AttributeError:
            pass
        
        try:
            intents.guild_messages = True   # 舊版本相容
        except AttributeError:
            pass
            
        try:
            intents.dm_messages = True      # 私訊消息
        except AttributeError:
            try:
                intents.private_messages = True  # 更舊的版本
            except AttributeError:
                pass
        
        intents.guilds = True              # 伺服器事件
        
        super().__init__(
            command_prefix='!',
            intents=intents,
            case_insensitive=True,
            strip_after_prefix=True,
            help_command=None  # 停用預設幫助命令
        )
        
        # AI 聊天系統
        self.rushia_chat = None
        self.model_loaded = False
        self.model_loading = False
        
        # 消息處理隊列 - 提升效能
        self.message_queue = queue.Queue(maxsize=100)
        self.processing_messages = {}
        
        # 個人專用模式 - 白名單用戶
        self.owner_id = None  # 將在啟動時從環境變數或輸入獲取
        self.personal_mode = True  # 個人專用模式
        
        # 移除冷卻系統 - 個人使用不需要限制
        # self.user_cooldowns = {}
        # self.cooldown_duration = 2.0
        
        # 主動訊息系統
        self.proactive_channel = None  # 主動發送訊息的頻道（私訊或指定頻道）
        self.last_interaction_time = None  # 最後互動時間
        self.proactive_enabled = True  # 是否啟用主動訊息
        
        # 背景任務狀態
        self.background_tasks_started = False
        
        # 伺服器配置
        self.server_configs = {}
        self.default_config = {
            'enabled_channels': [],  # 空白表示所有頻道
            'blacklisted_users': [],
            'max_message_length': 500,
            'response_chance': 1.0,  # 回應機率
            'auto_reply_mentions': True,
            'auto_reply_dms': True
        }
        
        # 統計資料
        self.stats = {
            'messages_processed': 0,
            'responses_generated': 0,
            'uptime_start': time.time(),
            'errors': 0,
            'messages_sent': 0
        }
        
        # 背景任務變數
        self.message_processor_task = None
        self.proactive_checker_task = None
        self.cleanup_task = None
    
    async def setup_hook(self):
        """Bot 初始化設定"""
        logger.info("🚀 Discord AI Bot 正在啟動...")
        
        try:
            # 設定擁有者 ID
            self.setup_owner_sync()
            
            # 載入配置
            self.load_configurations_sync()
            
            # 延遲啟動消息處理器，等待 Bot 完全連線
            logger.info("✅ Bot 基本設定完成，等待連線...")
            
            # 在背景載入 AI 模型 - 不等待完成
            asyncio.create_task(self.load_ai_model())
            
            # 確保 setup_hook 快速完成
            logger.info("✅ Bot 初始化完成，AI 模型正在背景載入...")
            logger.info("⏳ 等待 Discord 連接...")
            
        except Exception as e:
            logger.error(f"❌ setup_hook 發生錯誤: {e}")
            logger.error(traceback.format_exc())
    
    def setup_owner_sync(self):
        """設定 Bot 擁有者 - 同步版本"""
        # 從環境變數讀取擁有者 ID
        owner_id_str = os.getenv('DISCORD_OWNER_ID')
        
        if owner_id_str:
            try:
                self.owner_id = int(owner_id_str)
                logger.info(f"✅ 從環境變數設定擁有者 ID: {self.owner_id}")
                return
            except ValueError:
                logger.error("❌ DISCORD_OWNER_ID 格式錯誤")
        
        if not self.owner_id:
            logger.warning("⚠️ 未設定擁有者 ID")
            logger.info("💡 請在 .env 檔案中設定: DISCORD_OWNER_ID=你的Discord用戶ID")
            logger.info("💡 或使用 !setowner 命令設定擁有者")
            logger.info("⚠️ 暫時允許所有用戶使用 Bot")
            self.personal_mode = False
    
    def load_configurations_sync(self):
        """載入伺服器配置 - 同步版本"""
        config_file = 'bot_config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.server_configs = json.load(f)
                logger.info(f"✅ 已載入 {len(self.server_configs)} 個伺服器配置")
            except Exception as e:
                logger.error(f"❌ 載入配置失敗: {e}")
    
    async def setup_owner(self):
        """設定 Bot 擁有者"""
        # 從環境變數讀取擁有者 ID
        owner_id_str = os.getenv('DISCORD_OWNER_ID')
        
        if owner_id_str:
            try:
                self.owner_id = int(owner_id_str)
                logger.info(f"✅ 從環境變數設定擁有者 ID: {self.owner_id}")
                return
            except ValueError:
                logger.error("❌ DISCORD_OWNER_ID 格式錯誤")
        
        if not self.owner_id:
            logger.warning("⚠️ 未設定擁有者 ID")
            logger.info("💡 請在 .env 檔案中設定: DISCORD_OWNER_ID=你的Discord用戶ID")
            logger.info("💡 或使用 !setowner 命令設定擁有者")
            logger.info("⚠️ 暫時允許所有用戶使用 Bot")
            self.personal_mode = False
    
    async def load_configurations(self):
        """載入伺服器配置"""
        config_file = 'bot_config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.server_configs = json.load(f)
                logger.info(f"✅ 已載入 {len(self.server_configs)} 個伺服器配置")
            except Exception as e:
                logger.error(f"❌ 載入配置失敗: {e}")
    
    async def save_configurations(self):
        """儲存伺服器配置"""
        try:
            with open('bot_config.json', 'w', encoding='utf-8') as f:
                json.dump(self.server_configs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ 儲存配置失敗: {e}")
    
    def get_server_config(self, guild_id: int) -> Dict:
        """獲取伺服器配置"""
        return self.server_configs.get(str(guild_id), self.default_config.copy())
    
    async def load_ai_model(self):
        """在背景線程中載入 AI 模型"""
        self.model_loading = True
        logger.info("⏳ 開始載入露西亞 AI 模型...")
        
        def load_model():
            try:
                logger.info("🤖 正在初始化 RushiaLoRAChat...")
                self.rushia_chat = RushiaLoRAChat()
                logger.info("🤖 正在載入模型檔案...")
                success = self.rushia_chat.load_model()
                
                if success:
                    self.model_loaded = True
                    logger.info("✅ 露西亞 AI 模型載入成功!")
                    
                    # 如果已設定擁有者，初始化主動訊息系統
                    if self.owner_id:
                        # 設定初始互動時間
                        self.last_interaction_time = time.time()
                        if hasattr(self.rushia_chat, 'update_message_timing'):
                            self.rushia_chat.update_message_timing(is_user_message=True)
                        logger.info("✅ 主動訊息系統已初始化")
                else:
                    logger.error("❌ AI 模型載入失敗")
                    
            except Exception as e:
                logger.error(f"❌ AI 模型載入異常: {e}")
                logger.error(traceback.format_exc())
            finally:
                self.model_loading = False
        
        # 使用 ThreadPoolExecutor 確保完全背景執行
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # 提交任務但不等待完成
                future = executor.submit(load_model)
                logger.info("⏳ AI 模型正在背景載入中...")
                # 立即返回，不等待完成
                return
        except Exception as e:
            logger.error(f"❌ 啟動模型載入任務失敗: {e}")
            self.model_loading = False
    
    def check_user_permission(self, user_id: int) -> bool:
        """檢查用戶是否有權限使用 Bot"""
        if not self.personal_mode:
            return True  # 非個人模式，允許所有用戶
        
        if not self.owner_id:
            return True  # 未設定擁有者，允許所有用戶
        
        is_owner = user_id == self.owner_id
        if not is_owner:
            logger.info(f"🚫 用戶 {user_id} 不在白名單中，拒絕回應")
        
        return is_owner
    
    async def should_respond(self, message: discord.Message) -> bool:
        """判斷是否應該回應此消息"""
        # 基本檢查
        if message.author.bot:
            return False
        
        if not self.model_loaded:
            return False
        
        # 個人專用模式 - 檢查用戶權限
        if not self.check_user_permission(message.author.id):
            return False
        
        should_respond = False
        
        # DM 自動回應 - 私訊必定回應 (僅限白名單用戶)
        if isinstance(message.channel, discord.DMChannel):
            logger.info(f"✅ 收到擁有者私訊: {message.content[:30]}...")
            should_respond = True
        
        # 被提及時自動回應 (僅限白名單用戶)
        elif self.user.mentioned_in(message):
            logger.info(f"✅ 被擁有者提及於 {message.guild.name if message.guild else 'DM'}")
            should_respond = True
        
        # 如果需要回應，則更新 proactive system 狀態（用戶發送了正常訊息）
        if should_respond and self.rushia_chat and hasattr(self.rushia_chat, 'proactive_system'):
            current_time = time.time()
            ps = self.rushia_chat.proactive_system
            
            # 更新用戶訊息時間
            ps['last_user_message_time'] = current_time
            ps['last_message_time'] = current_time
            
            # 如果在等待回應，則重置等待狀態
            if ps['waiting_for_response']:
                ps['waiting_for_response'] = False
                ps['reminder_count'] = 0
                logger.info("✅ 用戶回應了，重置催促系統")
        
        return should_respond
    
    async def on_ready(self):
        """Bot 就緒事件"""
        logger.info(f"✅ {self.user} 已連線到 Discord!")
        logger.info(f"📊 已連接到 {len(self.guilds)} 個伺服器")
        
        # 顯示個人專用模式狀態
        if self.personal_mode and self.owner_id:
            logger.info(f"🔒 個人專用模式已啟用 - 擁有者 ID: {self.owner_id}")
            try:
                owner_user = await self.fetch_user(self.owner_id)
                logger.info(f"👤 擁有者: {owner_user.name}#{owner_user.discriminator}")
            except discord.NotFound:
                logger.warning(f"⚠️ 無法找到擁有者用戶 (ID: {self.owner_id})")
            except Exception as e:
                logger.warning(f"⚠️ 獲取擁有者資訊時發生錯誤: {e}")
        elif not self.personal_mode:
            logger.info("🌐 公開模式 - 所有用戶可使用")
        else:
            logger.warning("⚠️ 個人專用模式但未設定擁有者，請使用 !setowner 設定")
        
        # 設定 Bot 狀態
        activity_name = "專屬聊天♪" if self.personal_mode else "溫柔的聊天♪"
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name=activity_name
            ),
            status=discord.Status.online
        )
        
        # 啟動模型載入狀態檢查
        if not self.model_loaded and self.model_loading:
            asyncio.create_task(self.check_model_loading())
        
        # 啟動背景任務
        if not self.background_tasks_started:
            self.start_background_tasks()
            self.background_tasks_started = True
    
    async def check_model_loading(self):
        """檢查模型載入狀態"""
        check_count = 0
        while self.model_loading and check_count < 120:  # 最多等待2分鐘
            await asyncio.sleep(1)
            check_count += 1
            
            if check_count % 10 == 0:  # 每10秒報告一次
                logger.info(f"⏳ AI 模型載入中... ({check_count}s)")
        
        if self.model_loaded:
            logger.info("🎉 AI 模型已準備就緒！")
        elif not self.model_loading:
            logger.warning("⚠️ AI 模型載入已結束，但狀態未知")
        else:
            logger.error("❌ AI 模型載入超時，請檢查模型檔案")
    
    def start_background_tasks(self):
        """啟動所有背景任務"""
        logger.info("🔄 啟動背景任務...")
        
        # 檢查 Bot 是否準備就緒
        if not self.is_ready():
            logger.warning("⚠️ Bot 尚未準備就緒，延遲啟動背景任務...")
            # 延遲啟動
            asyncio.create_task(self._delayed_start_background_tasks())
            return
        
        # 啟動消息處理器
        self.message_processor_task = asyncio.create_task(self.message_processor())
        logger.info("✅ 消息處理器已啟動")
        
        # 啟動主動訊息檢查器
        if self.proactive_enabled:
            self.proactive_checker_task = asyncio.create_task(self.proactive_message_checker())
            logger.info("✅ 主動訊息檢查器已啟動")
            
    async def _delayed_start_background_tasks(self):
        """延遲啟動背景任務"""
        # 等待 Bot 準備就緒
        retries = 0
        while not self.is_ready() and retries < 30:  # 最多等待30秒
            await asyncio.sleep(1)
            retries += 1
            
        if self.is_ready():
            logger.info("✅ Bot 已準備就緒，啟動背景任務...")
            
            # 啟動消息處理器
            self.message_processor_task = asyncio.create_task(self.message_processor())
            logger.info("✅ 消息處理器已啟動")
            
            # 啟動主動訊息檢查器
            if self.proactive_enabled:
                self.proactive_checker_task = asyncio.create_task(self.proactive_message_checker())
                logger.info("✅ 主動訊息檢查器已啟動")
        else:
            logger.error("❌ Bot 未能在預期時間內準備就緒，背景任務啟動失敗")
    
    async def proactive_message_checker(self):
        """主動訊息檢查器 - 背景任務"""
        logger.info("🤖 主動訊息檢查器已啟動")
        
        while not self.is_closed():
            try:
                # 檢查 Bot 連線狀態
                if not self.is_ready():
                    logger.warning("⚠️ Discord 連線中斷，暫停主動訊息檢查...")
                    await asyncio.sleep(30)  # 等待重連
                    continue
                
                # 每15秒檢查一次（提高催促訊息的及時性）
                await asyncio.sleep(15)
                
                # 等待模型載入完成
                if not self.model_loaded:
                    logger.debug("⏳ 等待模型載入完成...")
                    continue
                
                # 只有當為個人專用模式且有擁有者時才檢查
                if not self.personal_mode or not self.owner_id:
                    logger.debug("⏸️ 非個人模式或未設定擁有者，跳過主動訊息檢查")
                    continue
                
                # 檢查是否需要發送主動訊息或催促訊息
                if self.rushia_chat:
                    message, message_type = self.rushia_chat.get_proactive_message_if_needed()
                    
                    if message and message_type:
                        logger.info(f"🔔 準備發送{message_type}訊息: {message[:50]}...")
                        await self.send_proactive_message(message, message_type)
                    else:
                        # 每3分鐘記錄一次詳細狀態用於調試
                        if hasattr(self, '_last_debug_log'):
                            if time.time() - self._last_debug_log > 180:  # 改為3分鐘
                                self._log_proactive_system_debug()
                                self._last_debug_log = time.time()
                        else:
                            self._last_debug_log = time.time()
                            self._log_proactive_system_debug()
                        
                        # 詳細日誌監控：顯示當前狀態和催促進度
                        if hasattr(self.rushia_chat, 'proactive_system'):
                            ps = self.rushia_chat.proactive_system
                            waiting = ps.get('waiting_for_response', False)
                            reminder_count = ps.get('reminder_count', 0)
                            last_proactive = ps.get('last_proactive_message_time')
                            last_user = ps.get('last_user_message_time')
                            current_time = time.time()
                            
                            # 詳細的催促系統狀態監控
                            if waiting and last_proactive:
                                wait_minutes = (current_time - last_proactive) / 60
                                reminder_intervals = [5, 15, 30, 60]
                                
                                # 檢查每個催促時間點的狀態
                                status_info = []
                                for i, interval in enumerate(reminder_intervals):
                                    if i == reminder_count:
                                        if wait_minutes >= interval:
                                            logger.warning(f"🔔 催促檢查: 第{i+1}次催促時間已到 ({interval}分) - 應該發送但未觸發！")
                                            # 檢查為什麼沒有觸發
                                            can_send = self.rushia_chat.should_send_reminder()
                                            logger.info(f"🔍 催促檢查: should_send_reminder() = {can_send}")
                                        else:
                                            remaining = interval - wait_minutes
                                            logger.debug(f"⏳ 催促檢查: 第{i+1}次催促 ({interval}分) 還需{remaining:.1f}分鐘")
                                        break
                                    elif i < reminder_count:
                                        logger.debug(f"✅ 催促檢查: 第{i+1}次催促 ({interval}分) 已發送")
                                
                                # 每分鐘記錄一次等待狀態
                                if int(wait_minutes) % 1 == 0:  # 每分鐘整數時記錄
                                    logger.info(f"📊 催促狀態: 等待回應 {wait_minutes:.1f}分鐘, 已催促{reminder_count}次")
                            
                            elif not waiting:
                                # 檢查主動訊息觸發條件
                                if last_user:
                                    silence_minutes = (current_time - last_user) / 60
                                    should_send = self.rushia_chat.should_send_proactive_message()
                                    if silence_minutes > 30:  # 超過30分鐘沉默時記錄
                                        logger.debug(f"📊 主動訊息檢查: 沉默{silence_minutes:.1f}分鐘, should_send={should_send}")
                        
            except discord.HTTPException as e:
                logger.error(f"❌ Discord HTTP 錯誤: {e}")
                await asyncio.sleep(30)  # HTTP 錯誤時等待30秒
            except discord.ConnectionClosed as e:
                logger.error(f"❌ Discord 連線關閉: {e}")
                await asyncio.sleep(60)  # 連線錯誤時等待更久
            except Exception as e:
                logger.error(f"❌ 主動訊息檢查器錯誤: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)  # 其他錯誤時等待更久
    
    async def send_proactive_message(self, message: str, message_type: str):
        """發送主動訊息"""
        try:
            target_channel = None
            
            # 確定發送目標
            if self.proactive_channel:
                # 使用設定的頻道
                target_channel = self.proactive_channel
            elif self.owner_id:
                # 發送私訊給擁有者
                try:
                    owner = await self.fetch_user(self.owner_id)
                    target_channel = owner
                except discord.NotFound:
                    logger.warning(f"⚠️ 無法找到擁有者 (ID: {self.owner_id})")
                    return
            
            if target_channel:
                # 發送訊息
                await target_channel.send(message)
                
                # 更新主動訊息系統狀態
                if self.rushia_chat and hasattr(self.rushia_chat, 'proactive_system'):
                    if message_type == "proactive":
                        # 主動訊息的狀態已在 generate_proactive_message 中設定，這裡只需記錄日誌
                        logger.debug(f"✅ 主動訊息發送完成，催促系統已啟動")
                    elif message_type == "reminder":
                        # 催促訊息不需要重新設定等待狀態
                        logger.debug(f"✅ 催促訊息發送完成")
                    elif message_type == "time_aware":
                        # 時間感知關心訊息
                        logger.info(f"🕐 時間感知關心訊息發送完成")
                        # 時間感知關心不會觸發催促系統，它是獨立的關心機制
                        logger.debug(f"✅ 催促訊息發送完成，催促次數: {self.rushia_chat.proactive_system.get('reminder_count', 0)}")
                
                # 紀錄日誌
                message_type_text = "催促訊息" if message_type == "reminder" else "主動訊息"
                logger.info(f"📤 已發送{message_type_text}: {message[:50]}...")
                
                # 更新統計
                self.stats['messages_sent'] += 1
                
        except discord.Forbidden:
            logger.warning("⚠️ 無權限發送主動訊息")
        except discord.HTTPException as e:
            logger.error(f"❌ 發送主動訊息失敗: {e}")
        except Exception as e:
            logger.error(f"❌ 主動訊息發送錯誤: {e}")
    
    def update_interaction_time(self, user_id: int):
        """更新用戶互動時間"""
        if self.personal_mode and user_id == self.owner_id:
            self.last_interaction_time = time.time()
            
            # 更新聊天系統的時間記錄
            if self.rushia_chat and hasattr(self.rushia_chat, 'update_message_timing'):
                self.rushia_chat.update_message_timing(is_user_message=True)
                logger.debug(f"✅ 用戶訊息時間更新完成，重置催促系統")
    
    async def on_message(self, message: discord.Message):
        """消息接收事件"""
        try:
            # 檢查是否為指令
            ctx = await self.get_context(message)
            if ctx.valid:
                # 是有效指令，處理並返回，不進入 AI 回應流程
                await self.process_commands(message)
                return
            
            # 檢查是否需要 AI 回應
            if await self.should_respond(message):
                # 更新互動時間
                self.update_interaction_time(message.author.id)
                
                # 將消息加入處理隊列
                if not self.message_queue.full():
                    self.message_queue.put_nowait({
                        'message': message,
                        'timestamp': time.time()
                    })
                    self.stats['messages_processed'] += 1
                else:
                    logger.warning("⚠️ 消息處理隊列已滿，跳過此消息")
                    
        except Exception as e:
            logger.error(f"❌ 處理消息時發生錯誤: {e}")
            self.stats['errors'] += 1
    
    async def message_processor(self):
        """消息處理器 - 背景任務"""
        logger.info("🔄 消息處理器已啟動")
        
        while not self.is_closed():
            try:
                # 檢查 Bot 連線狀態
                if not self.is_ready():
                    logger.warning("⚠️ Discord 連線中斷，暫停消息處理...")
                    await asyncio.sleep(5)  # 等待重連
                    continue
                
                # 從隊列取得消息 - 使用異步等待避免阻塞
                try:
                    await asyncio.sleep(0.1)  # 短暫休息，避免忙等待
                    if self.message_queue.empty():
                        continue
                    
                    message_data = self.message_queue.get_nowait()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"❌ 從隊列取得消息時發生錯誤: {e}")
                    continue
                
                message = message_data['message']
                message_id = message.id
                
                # 防止重複處理
                if message_id in self.processing_messages:
                    continue
                
                self.processing_messages[message_id] = time.time()
                
                # 處理消息
                try:
                    await self.process_ai_message(message)
                except discord.HTTPException as e:
                    logger.error(f"❌ Discord HTTP 錯誤: {e}")
                except discord.ConnectionClosed as e:
                    logger.error(f"❌ Discord 連線關閉: {e}")
                    await asyncio.sleep(5)  # 等待重連
                except Exception as e:
                    logger.error(f"❌ 處理消息時發生未預期錯誤: {e}")
                    logger.error(traceback.format_exc())
                
                # 清理處理記錄
                if message_id in self.processing_messages:
                    del self.processing_messages[message_id]
                
            except Exception as e:
                logger.error(f"❌ 消息處理器發生錯誤: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(1)
    
    async def process_ai_message(self, message: discord.Message):
        """處理 AI 消息生成"""
        try:
            # 提取用戶訊息
            user_message = message.content
            
            # 移除 Bot 提及
            if self.user.mentioned_in(message):
                user_message = user_message.replace(f'<@{self.user.id}>', '').strip()
                user_message = user_message.replace(f'<@!{self.user.id}>', '').strip()
            
            if not user_message:
                user_message = "你好"
            
            # 檢查消息長度
            max_length = 500
            if message.guild:
                config = self.get_server_config(message.guild.id)
                max_length = config.get('max_message_length', 500)
            
            if len(user_message) > max_length:
                await message.reply("消息太長了呢～請稍微簡短一些♪", mention_author=False)
                return
            
            # 顯示輸入指示器
            async with message.channel.typing():
                # 生成 AI 回應
                ai_response = await self.generate_ai_response(user_message)
                
                # 額外的回應品質檢查（Discord端）
                if ai_response:
                    ai_response = self.clean_discord_response(ai_response)
                    
                    # 進一步檢查回應品質
                    if ai_response and len(ai_response.strip()) < 2:
                        ai_response = None
                    
                    # 如果回應仍然有問題，記錄並提供備用回應
                    if not ai_response:
                        logger.warning(f"⚠️ AI 回應品質檢查失敗，提供備用回應")
                        if "草莓牛奶" in user_message or "喝" in user_message:
                            ai_response = "好呀好呀～我這就去準備草莓牛奶給你喝！要加很多草莓哦♪"
                        elif "聊天" in user_message:
                            ai_response = "當然願意陪你聊天呀～有什麼想說的嗎？♪"
                        elif "下午好" in user_message or "中午好" in user_message:
                            ai_response = "下午好呀～今天過得怎麼樣？要不要一起聊聊天呢♪"
                        else:
                            ai_response = "嗯嗯～我在聽哦，繼續和我說說話吧♪"
                
                if ai_response:
                    # 分割長消息
                    if len(ai_response) > 2000:
                        chunks = [ai_response[i:i+2000] for i in range(0, len(ai_response), 2000)]
                        for chunk in chunks:
                            await message.reply(chunk, mention_author=False)
                    else:
                        await message.reply(ai_response, mention_author=False)
                    
                    # 更新AI回應時間記錄 - 只更新last_message_time，不重置等待狀態
                    if self.rushia_chat and hasattr(self.rushia_chat, 'proactive_system'):
                        # AI 對用戶訊息的回應，只更新時間，保持催促系統狀態不變
                        self.rushia_chat.proactive_system['last_message_time'] = time.time()
                        logger.debug(f"✅ AI回應時間已更新，保持催促系統狀態")
                    
                    self.stats['responses_generated'] += 1
                    logger.info(f"✅ 已回應用戶 {message.author.name}: {user_message[:50]}...")
                else:
                    await message.reply("抱歉，露西亞現在想不出要說什麼呢～♪", mention_author=False)
                    
        except discord.errors.Forbidden:
            logger.warning(f"⚠️ 沒有權限回覆消息 - 頻道: {message.channel}")
        except discord.errors.HTTPException as e:
            logger.error(f"❌ Discord HTTP 錯誤: {e}")
        except Exception as e:
            logger.error(f"❌ 處理 AI 消息時發生錯誤: {e}")
            logger.error(traceback.format_exc())
    
    async def generate_ai_response(self, user_message: str) -> Optional[str]:
        """生成 AI 回應"""
        if not self.model_loaded or not self.rushia_chat:
            logger.debug("❌ 模型未載入或 RushiaChat 未初始化")
            return None
        
        try:
            loop = asyncio.get_event_loop()
            # 使用新的 chat_asmr.py 的 chat 方法
            response = await loop.run_in_executor(
                None,
                self.rushia_chat.chat,
                user_message
            )
            
            # 確保回應是字符串類型
            if response and isinstance(response, str):
                return response.strip() if response.strip() else None
            else:
                logger.warning(f"⚠️ AI 回應不是字符串類型: {type(response)}")
                return None
            
        except Exception as e:
            logger.error(f"❌ AI 生成回應時發生錯誤: {e}")
            logger.error(traceback.format_exc())
            return None
    
    @tasks.loop(minutes=30)
    async def cleanup_task_loop(self):
        """定期清理任務"""
        try:
            current_time = time.time()
            
            # 清理過期的處理記錄
            expired_keys = [
                key for key, timestamp in self.processing_messages.items()
                if current_time - timestamp > 300  # 5分鐘
            ]
            for key in expired_keys:
                del self.processing_messages[key]
            
            # GPU 顯存清理
            if self.rushia_chat and hasattr(self.rushia_chat, 'device'):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            logger.debug("🧹 定期清理完成")
            
        except Exception as e:
            logger.error(f"❌ 清理任務發生錯誤: {e}")
    
    def clean_discord_response(self, response):
        """Discord 端額外的回應清理 - 強化版本"""
        if not response or not isinstance(response, str):
            return None
            
        # 移除殘留的不當內容和格式問題
        try:
            # 移除不必要的引號和對話格式
            response = re.sub(r'^[「『"\']*', '', response)  # 移除開頭引號
            response = re.sub(r'[」』"\']*$', '', response)  # 移除結尾引號
            
            # 移除對話格式殘留
            response = re.sub(r'露西[亞亚][:：]\s*', '', response)
            response = re.sub(r'露醬[:：]\s*', '', response)
            response = re.sub(r'^.*?[:：]\s*', '', response)  # 移除其他對話格式
            
            # 移除"回應"等元資訊
            response = re.sub(r'\s*回應\s*$', '', response)
            response = re.sub(r'\s*露醬回應\s*$', '', response)
            response = re.sub(r'\s*露醬\s*$', '', response)  # 移除結尾的"露醬"
            response = re.sub(r'」\s*露醬.*?$', '」', response)  # 處理引號後的露醬
            
            # 清理不當內容
            unwanted_patterns = [
                r'燻肉.*?味',
                r'の香',
                r'≧ω≦',
                r'๑•̀ㅂ•́و✧',
                r'輸入.*?".*?$',
            ]
            
            for pattern in unwanted_patterns:
                response = re.sub(pattern, '', response, flags=re.IGNORECASE)
            
            # 簡轉繁處理（額外保險）
            simple_to_traditional = {
                '亚': '亞', '说': '說', '话': '話', '觉': '覺', '认': '認',
                '为': '為', '爱': '愛', '开': '開', '关': '關', '听': '聽',
                '谢': '謝', '应': '應', '该': '該', '过': '過', '这': '這',
                '那': '那', '会': '會', '还': '還', '没': '沒', '来': '來',
                '时': '時', '间': '間', '让': '讓', '给': '給', '帮': '幫',
                '问': '問', '题': '題', '样': '樣', '门': '門', '办': '辦',
                '经': '經', '历': '歷', '现': '現', '实': '實', '试': '試',
                '验': '驗', '刚': '剛', '才': '才', '呢': '呢', '吗': '嗎',
                '啊': '啊', '呀': '呀', '喔': '喔', '哦': '哦'
            }
            
            for simple, traditional in simple_to_traditional.items():
                response = response.replace(simple, traditional)
            
            # 修正常見的句子截斷問題
            if response.endswith('我你'):
                response = response[:-2] + '♪'
            elif response.endswith('我你有什麼'):
                response = response[:-5] + '有什麼想聊的嗎？♪'
            elif '一個小，' in response:  # 修正"做了一個小，"這種截斷
                response = response.replace('一個小，', '一個小專案，')
            
            # 確保回應不為空且合理
            response = response.strip()
            if len(response) < 2:
                return None
                
            # 檢查是否包含過多奇怪符號
            weird_symbols = len(re.findall(r'[≧≦ω๑ㅂו✧]', response))
            if weird_symbols > 2:
                return None
            
            # 確保回應以合適的標點結尾
            if response and not re.search(r'[♪♡～。！？!?.]$', response):
                if '草莓牛奶' in response or '喝' in response:
                    response += '♪'
                elif '聊天' in response or '陪' in response:
                    response += '～'
                else:
                    response += '♪'
                
            return response
        except Exception as e:
            logger.error(f"❌ 清理回應時發生錯誤: {e}")
            return None
    
    # Bot 命令
    @commands.command(name='status')
    async def status_command(self, ctx):
        """顯示 Bot 狀態"""
        uptime = time.time() - self.stats['uptime_start']
        uptime_str = str(timedelta(seconds=int(uptime)))
        
        embed = discord.Embed(
            title="🤖 露西亞 Bot 狀態",
            color=0xFF69B4,
            timestamp=datetime.utcnow()
        )
        
        embed.add_field(
            name="🔧 模型狀態",
            value="✅ 已載入" if self.model_loaded else ("🔄 載入中..." if self.model_loading else "❌ 未載入"),
            inline=True
        )
        
        embed.add_field(
            name="⏰ 運行時間",
            value=uptime_str,
            inline=True
        )
        
        embed.add_field(
            name="📊 統計資料",
            value=f"處理消息: {self.stats['messages_processed']}\n"
                  f"生成回應: {self.stats['responses_generated']}\n"
                  f"錯誤次數: {self.stats['errors']}",
            inline=True
        )
        
        embed.add_field(
            name="🌐 連接狀態",
            value=f"伺服器: {len(self.guilds)}\n"
                  f"延遲: {round(self.latency * 1000)}ms",
            inline=True
        )
        
        embed.add_field(
            name="👤 個人專用模式",
            value=f"{'✅ 啟用' if self.personal_mode else '❌ 停用'}\n"
                  f"擁有者 ID: {self.owner_id if self.owner_id else '未設定'}",
            inline=True
        )
        
        embed.add_field(
            name="💬 使用說明",
            value="• 直接私訊我來聊天 (僅限擁有者)\n• 在伺服器中 @我 來聊天 (僅限擁有者)\n• 無冷卻限制，隨時對話",
            inline=False
        )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='reload')
    @commands.has_permissions(administrator=True)
    async def reload_model(self, ctx):
        """重新載入 AI 模型 (僅管理員)"""
        if self.model_loading:
            await ctx.send("❌ 模型正在載入中，請稍候...")
            return
        
        await ctx.send("🔄 開始重新載入 AI 模型...")
        
        # 重置狀態
        self.model_loaded = False
        self.rushia_chat = None
        
        # 重新載入
        await self.load_ai_model()
        
        if self.model_loaded:
            await ctx.send("✅ AI 模型重新載入成功!")
        else:
            await ctx.send("❌ AI 模型重新載入失敗!")
    
    @commands.command(name='setowner')
    async def set_owner_command(self, ctx, user_id: int = None):
        """設定 Bot 擁有者 (僅當前擁有者或未設定時可用)"""
        # 檢查權限
        if self.owner_id and ctx.author.id != self.owner_id:
            await ctx.send("❌ 只有當前擁有者可以更改擁有者設定!")
            return
        
        if user_id is None:
            user_id = ctx.author.id
        
        self.owner_id = user_id
        self.personal_mode = True
        
        # 設定主動訊息頻道
        if isinstance(ctx.channel, discord.DMChannel):
            self.proactive_channel = await self.fetch_user(user_id)
        else:
            self.proactive_channel = ctx.channel
        
        try:
            user = await self.fetch_user(user_id)
            await ctx.send(f"✅ 已設定擁有者為: {user.name} (ID: {user_id})")
            logger.info(f"✅ 擁有者已更改為: {user_id}")
            
            # 初始化互動時間
            self.update_interaction_time(user_id)
        except discord.NotFound:
            await ctx.send(f"✅ 已設定擁有者 ID: {user_id} (無法取得用戶名稱)")
    
    @commands.command(name='togglemode')
    async def toggle_personal_mode(self, ctx):
        """切換個人專用模式 (僅擁有者可用)"""
        if self.owner_id and ctx.author.id != self.owner_id:
            await ctx.send("❌ 只有擁有者可以切換模式!")
            return
        
        self.personal_mode = not self.personal_mode
        mode_text = "個人專用模式" if self.personal_mode else "公開模式"
        await ctx.send(f"✅ 已切換為: {mode_text}")
        logger.info(f"✅ 模式已切換為: {mode_text}")
    
    @commands.command(name='proactive')
    async def toggle_proactive(self, ctx, action: str = "status"):
        """管理主動訊息功能 (僅擁有者可用)
        
        用法:
        !proactive status - 顯示主動訊息狀態
        !proactive on/off - 開啟/關閉主動訊息
        !proactive setchannel - 設定主動訊息頻道為當前頻道
        !proactive setdm - 設定主動訊息為私訊
        """
        if self.owner_id and ctx.author.id != self.owner_id:
            await ctx.send("❌ 只有擁有者可以管理主動訊息功能!")
            return
        
        action = action.lower()
        
        if action == "status":
            # 顯示主動訊息狀態
            embed = discord.Embed(
                title="🤖 主動訊息系統狀態",
                color=0x00FF00 if self.proactive_enabled else 0xFF0000
            )
            
            # 基本狀態
            embed.add_field(
                name="📊 系統狀態",
                value=f"主動訊息: {'✅' if self.proactive_enabled else '❌'}\n"
                      f"個人模式: {'✅' if self.personal_mode else '❌'}\n"
                      f"模型狀態: {'✅' if self.model_loaded else '❌'}",
                inline=False
            )
            
            # 頻道設定
            if self.proactive_channel:
                if isinstance(self.proactive_channel, discord.User):
                    channel_text = f"私訊 ({self.proactive_channel.name})"
                else:
                    channel_text = f"#{self.proactive_channel.name}"
            else:
                channel_text = "未設定"
            
            embed.add_field(
                name="📍 發送目標",
                value=channel_text,
                inline=True
            )
            
            # 統計資訊
            if self.rushia_chat:
                daily_count = self.rushia_chat.proactive_system.get('daily_proactive_count', 0)
                last_time = self.rushia_chat.proactive_system.get('last_proactive_message_time')
                waiting = self.rushia_chat.proactive_system.get('waiting_for_response', False)
                
                time_text = "從未發送"
                if last_time:
                    time_diff = time.time() - last_time
                    if time_diff < 3600:
                        time_text = f"{int(time_diff/60)} 分鐘前"
                    else:
                        time_text = f"{int(time_diff/3600)} 小時前"
                
                embed.add_field(
                    name="📈 今日統計",
                    value=f"主動訊息: {daily_count}/5\n"
                          f"最後發送: {time_text}\n"
                          f"等待回應: {'是' if waiting else '否'}",
                    inline=True
                )
            
            await ctx.send(embed=embed)
            
        elif action in ["on", "enable", "開啟"]:
            self.proactive_enabled = True
            await ctx.send("✅ 主動訊息功能已開啟")
            logger.info("✅ 主動訊息功能已開啟")
            
        elif action in ["off", "disable", "關閉"]:
            self.proactive_enabled = False
            await ctx.send("❌ 主動訊息功能已關閉")
            logger.info("❌ 主動訊息功能已關閉")
            
        elif action in ["setchannel", "設定頻道"]:
            if isinstance(ctx.channel, discord.DMChannel):
                await ctx.send("❌ 無法在私訊中設定頻道！請在伺服器頻道中使用此命令。")
                return
            
            self.proactive_channel = ctx.channel
            await ctx.send(f"✅ 主動訊息頻道已設定為: #{ctx.channel.name}")
            logger.info(f"✅ 主動訊息頻道設定為: {ctx.channel.id}")
            
        elif action in ["setdm", "私訊", "dm"]:
            if self.owner_id:
                try:
                    owner = await self.fetch_user(self.owner_id)
                    self.proactive_channel = owner
                    await ctx.send("✅ 主動訊息已設定為私訊模式")
                    logger.info("✅ 主動訊息設定為私訊模式")
                except discord.NotFound:
                    await ctx.send("❌ 無法找到擁有者用戶")
            else:
                await ctx.send("❌ 未設定擁有者，無法設定私訊模式")
                
        else:
            await ctx.send("❌ 未知操作！請使用: `status`, `on`, `off`, `setchannel`, `setdm`")
    
    @commands.command(name='testproactive')
    async def test_proactive(self, ctx):
        """測試主動訊息功能 (僅擁有者可用)"""
        if self.owner_id and ctx.author.id != self.owner_id:
            await ctx.send("❌ 只有擁有者可以測試主動訊息!")
            return
        
        if not self.model_loaded or not self.rushia_chat:
            await ctx.send("❌ AI 模型尚未載入，無法測試主動訊息")
            return
        
        # 生成測試用主動訊息
        test_message = self.rushia_chat.generate_proactive_message()
        
        # 發送測試訊息
        await ctx.send(f"🧪 **測試主動訊息:**\n{test_message}")
        logger.info(f"🧪 發送測試主動訊息: {test_message[:50]}...")
        
        # 也測試催促訊息
        reminder_message = self.rushia_chat.generate_reminder_message()
        await ctx.send(f"🧪 **測試催促訊息:**\n{reminder_message}")
        logger.info(f"🧪 發送測試催促訊息: {reminder_message[:50]}...")
    
    @commands.command(name='forcemessage')
    async def force_proactive(self, ctx):
        """強制發送一條主動訊息 (僅擁有者可用)"""
        if self.owner_id and ctx.author.id != self.owner_id:
            await ctx.send("❌ 只有擁有者可以強制發送主動訊息!")
            return
        
        if not self.model_loaded or not self.rushia_chat:
            await ctx.send("❌ AI 模型尚未載入，無法發送主動訊息")
            return
        
        # 強制生成並發送主動訊息
        message = self.rushia_chat.generate_proactive_message()
        
        # 更新主動訊息系統狀態
        self.rushia_chat.proactive_system['last_proactive_message_time'] = time.time()
        self.rushia_chat.proactive_system['waiting_for_response'] = True
        self.rushia_chat.proactive_system['reminder_count'] = 0  # 重置催促次數
        self.rushia_chat.proactive_system['daily_proactive_count'] += 1
        
        await self.send_proactive_message(message, "proactive")
        await ctx.send(f"✅ 已強制發送主動訊息，催促系統已啟動")
        logger.info(f"👨‍💼 手動觸發主動訊息: {message[:50]}...")
    
    @commands.command(name='config')
    @commands.has_permissions(manage_guild=True)
    async def config_command(self, ctx, action: str = "show", setting: str = None, *, value: str = None):
        """配置 Bot 設定 (僅伺服器管理員) - 個人模式下此功能受限"""
        if self.personal_mode and ctx.author.id != self.owner_id:
            await ctx.send("❌ 個人專用模式下，只有擁有者可以配置設定!")
            return
            
        if not ctx.guild:
            await ctx.send("❌ 此命令只能在伺服器中使用!")
            return
        
        guild_id = str(ctx.guild.id)
        config = self.get_server_config(ctx.guild.id)
        
        if action.lower() == "show":
            embed = discord.Embed(
                title=f"⚙️ {ctx.guild.name} 的 Bot 配置",
                color=0x00FF00
            )
            
            embed.add_field(
                name="🔊 啟用頻道",
                value="所有頻道" if not config.get('enabled_channels') else 
                      ', '.join([f"<#{ch}>" for ch in config.get('enabled_channels', [])]),
                inline=False
            )
            
            embed.add_field(
                name="📝 設定選項",
                value=f"最大消息長度: {config.get('max_message_length', 500)}\n"
                      f"回應機率: {config.get('response_chance', 1.0) * 100:.0f}%\n"
                      f"自動回覆提及: {'✅' if config.get('auto_reply_mentions', True) else '❌'}\n"
                      f"自動回覆私訊: {'✅' if config.get('auto_reply_dms', True) else '❌'}",
                inline=False
            )
            
            if self.personal_mode:
                embed.add_field(
                    name="⚠️ 注意",
                    value="個人專用模式已啟用，Bot 只會回應擁有者的訊息",
                    inline=False
                )
            
            await ctx.send(embed=embed)
        
        elif action.lower() == "set" and setting and value:
            if setting == "max_length":
                try:
                    max_len = int(value)
                    if 50 <= max_len <= 2000:
                        config['max_message_length'] = max_len
                        self.server_configs[guild_id] = config
                        await self.save_configurations()
                        await ctx.send(f"✅ 最大消息長度設為: {max_len}")
                    else:
                        await ctx.send("❌ 長度必須在 50-2000 之間")
                except ValueError:
                    await ctx.send("❌ 請輸入有效數字")
            
            elif setting == "response_chance":
                try:
                    chance = float(value)
                    if 0.0 <= chance <= 1.0:
                        config['response_chance'] = chance
                        self.server_configs[guild_id] = config
                        await self.save_configurations()
                        await ctx.send(f"✅ 回應機率設為: {chance * 100:.0f}%")
                    else:
                        await ctx.send("❌ 機率必須在 0.0-1.0 之間")
                except ValueError:
                    await ctx.send("❌ 請輸入有效數字")
            
            else:
                await ctx.send("❌ 未知設定項目")
        
        else:
            await ctx.send("❌ 用法: `!config show` 或 `!config set <設定> <值>`")
    
    async def on_command_error(self, ctx, error):
        """命令錯誤處理"""
        if isinstance(error, commands.MissingPermissions):
            await ctx.send("❌ 你沒有權限使用此命令!")
        elif isinstance(error, commands.CommandNotFound):
            pass  # 忽略未知命令
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send("❌ 缺少必要參數!")
        else:
            logger.error(f"❌ 命令錯誤: {error}")
            await ctx.send("❌ 執行命令時發生錯誤!")
    
    @commands.command(name='reminderinfo')
    async def reminder_info(self, ctx):
        """顯示催促系統狀態 (僅擁有者可用)"""
        if self.owner_id and ctx.author.id != self.owner_id:
            await ctx.send("❌ 只有擁有者可以查看催促系統狀態!")
            return
        
        if not self.model_loaded or not self.rushia_chat:
            await ctx.send("❌ AI 模型尚未載入")
            return
        
        # 獲取催促系統狀態
        ps = self.rushia_chat.proactive_system
        current_time = time.time()
        
        embed = discord.Embed(
            title="🔔 催促系統狀態",
            color=0xFF69B4,
            timestamp=datetime.utcnow()
        )
        
        # 基本狀態
        embed.add_field(
            name="📊 系統狀態",
            value=f"等待回應: {'✅' if ps['waiting_for_response'] else '❌'}\n"
                  f"催促次數: {ps['reminder_count']}\n"
                  f"今日主動訊息: {ps['daily_proactive_count']}/5",
            inline=True
        )
        
        # 時間信息
        time_info = []
        if ps['last_user_message_time']:
            user_ago = (current_time - ps['last_user_message_time']) / 60
            time_info.append(f"用戶訊息: {user_ago:.1f}分鐘前")
        else:
            time_info.append("用戶訊息: 無記錄")
            
        if ps['last_proactive_message_time']:
            proactive_ago = (current_time - ps['last_proactive_message_time']) / 60
            time_info.append(f"主動訊息: {proactive_ago:.1f}分鐘前")
        else:
            time_info.append("主動訊息: 無記錄")
        
        embed.add_field(
            name="⏰ 時間記錄",
            value='\n'.join(time_info),
            inline=True
        )
        
        # 催促邏輯檢查
        if ps['waiting_for_response'] and ps['last_proactive_message_time']:
            wait_minutes = (current_time - ps['last_proactive_message_time']) / 60
            reminder_intervals = [5, 15, 30, 60]
            reminder_count = ps['reminder_count']
            
            status_info = []
            for i, interval in enumerate(reminder_intervals):
                if i == reminder_count:
                    if wait_minutes >= interval:
                        status_info.append(f"⏰ 第{i+1}次催促 ({interval}分): 準備發送")
                    else:
                        remaining = interval - wait_minutes
                        status_info.append(f"⏳ 第{i+1}次催促 ({interval}分): 還需{remaining:.1f}分鐘")
                    break
                elif i < reminder_count:
                    status_info.append(f"✅ 第{i+1}次催促 ({interval}分): 已發送")
                else:
                    status_info.append(f"⏸️ 第{i+1}次催促 ({interval}分): 未到時間")
            
            embed.add_field(
                name="🔔 催促狀態",
                value='\n'.join(status_info) if status_info else "無催促排程",
                inline=False
            )
        else:
            embed.add_field(
                name="🔔 催促狀態",
                value="無活動催促 (需要先發送主動訊息)",
                inline=False
            )
        
        await ctx.send(embed=embed)

    def _log_proactive_system_debug(self):
        """記錄主動訊息系統的詳細調試信息"""
        if not self.rushia_chat or not hasattr(self.rushia_chat, 'proactive_system'):
            return
            
        ps = self.rushia_chat.proactive_system
        current_time = time.time()
        
        # 基本狀態
        waiting = ps.get('waiting_for_response', False)
        reminder_count = ps.get('reminder_count', 0)
        daily_count = ps.get('daily_proactive_count', 0)
        
        # 時間信息
        last_user = ps.get('last_user_message_time')
        last_proactive = ps.get('last_proactive_message_time')
        
        user_silence = "無記錄"
        if last_user:
            user_silence_minutes = (current_time - last_user) / 60
            user_silence = f"{user_silence_minutes:.1f}分鐘"
        
        proactive_ago = "從未發送"
        if last_proactive:
            proactive_minutes = (current_time - last_proactive) / 60
            proactive_ago = f"{proactive_minutes:.1f}分鐘前"
        
        # 檢查觸發條件
        should_proactive = self.rushia_chat.should_send_proactive_message() if hasattr(self.rushia_chat, 'should_send_proactive_message') else False
        should_reminder = self.rushia_chat.should_send_reminder() if hasattr(self.rushia_chat, 'should_send_reminder') else False
        
        # 時間感知關心系統狀態
        time_care_status = "未啟用"
        if hasattr(self.rushia_chat, 'time_aware_care_system'):
            care_system = self.rushia_chat.time_aware_care_system
            if care_system.get('enabled', False):
                today_sent = care_system.get('daily_care_sent', {})
                sent_count = sum(1 for sent in today_sent.values() if sent)
                total_periods = len(today_sent)
                time_care_status = f"已發送 {sent_count}/{total_periods} 時段"
                
                # 檢查是否可以發送時間感知關心
                should_time_care, care_type = False, None
                if hasattr(self.rushia_chat, 'should_send_time_aware_care'):
                    should_time_care, care_type = self.rushia_chat.should_send_time_aware_care()
                
                if should_time_care and care_type:
                    time_care_status += f"，可發送 {care_type} 時段關心"
        
        logger.info(f"🔍 主動訊息系統狀態 - 等待回應:{waiting}, 催促次數:{reminder_count}, 今日主動:{daily_count}/5")
        logger.info(f"🔍 時間狀態 - 用戶沉默:{user_silence}, 上次主動:{proactive_ago}")
        logger.info(f"🔍 時間感知關心 - {time_care_status}")
        logger.info(f"🔍 觸發檢查 - 應發主動:{should_proactive}, 應發催促:{should_reminder}")
        
        # 如果等待回應，顯示催促進度
        if waiting and last_proactive:
            wait_minutes = (current_time - last_proactive) / 60
            reminder_intervals = [5, 15, 30, 60]
            
            if reminder_count < len(reminder_intervals):
                next_interval = reminder_intervals[reminder_count]
                remaining = next_interval - wait_minutes
                if remaining > 0:
                    logger.info(f"🔍 催促進度 - 第{reminder_count+1}次催促({next_interval}分)還需{remaining:.1f}分鐘")
                else:
                    logger.warning(f"🔍 催促進度 - 第{reminder_count+1}次催促({next_interval}分)已超時{abs(remaining):.1f}分鐘！")

    async def close(self):
        """關閉 Bot 並清理資源"""
        logger.info("🔄 正在關閉 Discord Bot...")
        
        # 取消背景任務
        if hasattr(self, 'message_processor_task') and self.message_processor_task:
            self.message_processor_task.cancel()
            try:
                await self.message_processor_task
            except asyncio.CancelledError:
                pass
            logger.info("✅ 消息處理器已停止")
        
        if hasattr(self, 'proactive_checker_task') and self.proactive_checker_task:
            self.proactive_checker_task.cancel()
            try:
                await self.proactive_checker_task
            except asyncio.CancelledError:
                pass
            logger.info("✅ 主動訊息檢查器已停止")
        
        # 調用父類的 close 方法
        await super().close()
        logger.info("✅ Discord Bot 已關閉")

def main():
    """主程式入口點"""
    print("🌸 露西亞 Discord AI Bot - 個人專用版 🌸")
    print("📋 個人專用功能:")
    print("   • 只回應指定擁有者的私訊和提及")
    print("   • 無冷卻限制，隨時對話")
    print("   • 主動發送訊息與催促回應")
    print("   • 其他用戶的訊息會被忽略")
    print("📜 主要命令:")
    print("   • !status - 查看 Bot 狀態")
    print("   • !setowner - 設定擁有者")
    print("   • !togglemode - 切換個人/公開模式")
    print("   • !proactive - 管理主動訊息功能")
    print("   • !testproactive - 測試主動訊息")
    print("="*50)
    
    # 檢查 .env 檔案 - 檢查多個可能位置
    env_locations = ['.env', '../.env', 'd:/RushiaMode/.env']
    env_found = False
    
    for env_path in env_locations:
        if os.path.exists(env_path):
            print(f"✅ 找到 .env 配置檔案: {env_path}")
            env_found = True
            break
    
    if not env_found:
        print("⚠️ 未找到 .env 檔案，建議創建一個")
    
    # 從環境變數讀取 Token
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    owner_id = os.getenv('DISCORD_OWNER_ID')
    
    if not bot_token:
        print("❌ 找不到 Discord Bot Token!")
        print("💡 請編輯 .env 檔案，設定:")
        print("   DISCORD_BOT_TOKEN=你的_bot_token")
        print("   DISCORD_OWNER_ID=你的_用戶_id")
        print("="*50)
        return
    
    if not owner_id:
        print("⚠️ 未設定擁有者 ID，Bot 將以公開模式啟動")
        print("💡 建議在 .env 檔案中設定: DISCORD_OWNER_ID=你的用戶ID")
        print("💡 或啟動後使用 !setowner 命令設定")
    else:
        print(f"✅ 擁有者 ID: {owner_id}")
    
    try:
        # 建立並執行 Bot
        print("🚀 個人專用 Bot 啟動中...")
        bot = DiscordAIBot()
        
        # 設定更詳細的日誌等級
        discord_logger = logging.getLogger('discord')
        discord_logger.setLevel(logging.WARNING)  # 減少 Discord 內部日誌
        
        bot.run(bot_token, log_handler=None)  # 使用我們自定義的日誌處理
        
    except discord.LoginFailure:
        logger.error("❌ Discord Bot Token 無效!")
        print("💡 請檢查你的 Bot Token 是否正確")
    except discord.HTTPException as e:
        logger.error(f"❌ Discord HTTP 錯誤: {e}")
        print("💡 可能是網路問題或 Discord API 限制")
        print("💡 如果是連線問題，請檢查網路連線後重試")
    except discord.ConnectionClosed as e:
        logger.error(f"❌ Discord 連線被關閉: {e}")
        print("💡 連線被中斷，可能是網路不穩定")
    except asyncio.TimeoutError:
        logger.error("❌ Discord 連線超時")
        print("💡 連線超時，請檢查網路連線後重試")
    except AttributeError as e:
        if "'NoneType' object has no attribute 'sequence'" in str(e):
            logger.error("❌ Discord WebSocket 連線錯誤 (sequence)")
            print("💡 WebSocket 連線錯誤，這通常是臨時的網路問題")
            print("💡 請稍等片刻後重新啟動")
        else:
            logger.error(f"❌ 屬性錯誤: {e}")
            logger.error(traceback.format_exc())
            print(f"❌ 發生屬性錯誤: {e}")
    except KeyboardInterrupt:
        logger.info("👋 Bot 已手動停止")
        print("\n👋 再見！")
    except Exception as e:
        logger.error(f"❌ Bot 執行時發生錯誤: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ 發生錯誤: {e}")
        print("💡 請檢查錯誤訊息並重新啟動")

if __name__ == "__main__":
    main()
