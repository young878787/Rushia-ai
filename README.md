# Rushia-ai 🌸 露西亞 AI 聊天系統

- 前情提要 以下都是Ai生的使用手冊可能會有小問題 有問題都可以上DC問我 
- 我的群組: https://discord.gg/UxwTqpvepr 
- threeds:https://www.threads.com/@young20160124
- email:young20160124@gmail.com
- 以下是我的感想/更新紀錄
- 2025/07/02 
第一次發布 一個以rushia問答訓練的聊天AI機器人  透過ragtag上面直播檔分析製作出問題集 用類似Qlora的方式訓練 改成8int而已 
bug還不少 就是題詞或是有些字詞沒有過濾乾淨(大概用7-8輪對話就會有)
-有點小煩人 設定成8-15 30 45 60 都會定時有訊息的機器人 還有固定週期 早 中 下午 晚上 都會有特定主動訊息(附圖放最下
訓練上我只附上ragtag腳本和翻譯的範例 我這裡整理聊天紀錄跟直播的程式太分散 哪天整合一個出來
感謝qwen3 8B和sakura qwen2.5 14B幫我完成底模設計和翻譯大量文本 
目前我還只是在放暑假的屁孩 對於這種AI設計有一定的執著 但你有好的建議或想一起開發都可以私訊我 歡迎各位大佬
## 一些範例圖檔
![image](https://github.com/user-attachments/assets/f24ded5d-8809-4ade-84fc-950fb6cedff3)
![image](https://github.com/user-attachments/assets/d080b9aa-65af-4477-a092-e3bd27032d76)

### 🤖 核心 AI 系統
- **LoRA 微調模型**：基於 Qwen 等大型語言模型進行露西亞專用訓練
- **多層過濾系統**：確保回應品質和角色一致性
- **情感分析引擎**：理解用戶情緒並做出適當回應
- **記憶管理系統**：維護對話歷史和用戶資料

### 💬 Discord 整合
- **即時聊天**：與 Discord 無縫整合，支援私訊和群組對話
- **主動訊息**：特定時間觸發關懷訊息和互動
- **權限控制**：個人專用模式，只回應指定用戶
- **命令系統**：豐富的管理指令和狀態查詢

### 🎯 智能回應系統
- **情境感知**：根據時間、情緒、對話歷史調整回應
- **多種回應類型**：親密對話、情感支持、日常聊天、食物話題
- **品質控制**：自動過濾不當內容和重複回應
- **個性化**：根據用戶偏好調整對話風格

### 🛠️ 輔助工具
- **RagTag 爬蟲**：自動下載直播影片和聊天記錄
- **聊天記錄分析**：分析觀眾互動模式，生成訓練建議
- **模型管理器**：管理多個 LoRA 模型版本

## 📋 系統需求

### 硬體需求
- **GPU**：RTX 3060 以上（至少12G VRAM 4070) 我目前用5070 推理+訓練
- **記憶體**：32GB RAM 以上
- **儲存空間**：50GB 以上可用空間

### 軟體環境
- Python 3.10+
- CUDA 11.8+ 或 12.x
- Windows 10/11

## 🚀 快速開始

### 1. 環境設置

```bash
# 克隆專案
git clone https://github.com/your-username/rushia-ai.git
cd rushia-ai

# 安裝依賴套件
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft datasets
pip install discord.py python-dotenv
pip install opencc jieba requests beautifulsoup4
pip install selenium yt-dlp
pip install accelerate optimum
pip install bitsandbytes
```

### 2. 配置文件設置

**重要：請根據您的系統路徑修改以下配置文件**

#### `config_llm.json` - LLM 模型配置
```json
{
  "llm_model_path": "d:/RushiaMode/models/qwen2.5-8b-instruct-q4_k_m.gguf",
  "llm_prompt_template": "你是一位可愛的Vtuber，請根據下列NMT初稿進行個性化潤色：{input}",
  "max_tokens": 512,
  "threads": 12,
  "gpu_layers": 60,
  "temperature": 0.7
}
```

#### `config_nmt.json` - 翻譯模型配置
```json
{
  "model_path": "d:/RushiaMode/models/sakura-14b-qwen2.5-v1.0-iq4xs.gguf",
  "llama_cpp_path": "d:/RushiaMode/scripts/llama.cpp",
  "threads": 12,
  "max_tokens": 1024,
  "temperature": 0.4,
  "sakura_api_url": "http://localhost:8000/v1/chat/completions"
}
```

#### `.env` - Discord 配置
```env
DISCORD_TOKEN=your_discord_bot_token_here #token
OWNER_ID=your_discord_user_id_here #目前設計以單人私訊為主 還沒有對應頻道的設計
```

### 3. 目錄結構設置

請確保以下目錄存在並調整為您的路徑：
```
D:/RushiaMode/
├── models/          # 存放 AI 模型
├── data/           # 訓練數據
├── chat_log/       # 聊天記錄
├── audio/          # 音頻文件
└── scripts/        # 腳本文件
```

## 🎮 使用方法

### 基本聊天系統

```python
from chat_asmr import RushiaLoRAChat

# 初始化聊天系統
chat = RushiaLoRAChat()

# 載入模型（首次使用會比較慢）
chat.load_model()

# 開始對話
while True:
    user_input = input("你: ")
    if user_input.lower() in ['quit', 'exit', '退出']:
        break
    
    response = chat.chat(user_input)
    print(f"露西亞: {response}")
```

### Discord 機器人部署

1. **設置 Discord Bot**
   - 前往 [Discord Developer Portal](https://discord.com/developers/applications)
   - 創建新應用程式和 Bot
   - 複製 Token 到 `.env` 文件
   - 目前是以個人私訊為主 有設計限制回覆功能(你顯卡夠好要讓多人用也可以 但目前測試都是私訊 我也很好奇多人會怎麼樣)

2. **啟動 Bot**
```bash
python discord_api_Aibot.py
```

3. **Discord 指令**
```
!status          # 查看 Bot 狀態
!setowner        # 設定擁有者
!togglemode      # 切換個人/公開模式
!proactive       # 管理主動訊息功能
!testproactive   # 測試主動訊息
!config          # 查看配置
```

## 🏋️ 模型訓練

### 準備訓練數據

1. **整理對話數據**
```python
# 數據格式示例
training_data = [
    {
        "input": "今天心情不好",
        "output": "怎麼了嗎？露西亞在這裡陪著你呢～"
    },
    # ... 更多對話數據
]
```

2. **使用訓練腳本**
```bash
python train_rushia_lora_advanced.py
```

### 訓練參數調整

適用於 RTX 5070 的最佳配置：
```python
# RTX 5070 優化設置
- batch_size: 4-8
- gradient_accumulation_steps: 4
- learning_rate: 2e-4
- lora_r: 16
- lora_alpha: 32
- 使用 bfloat16 精度
- 啟用 gradient_checkpointing
```

## 🔧 核心模組介紹

### 1. 主聊天系統 (`chat_asmr.py`)
- **核心功能**：處理用戶輸入，生成個性化回應
- **特色功能**：
  - 時間感知回應（早安、晚安等）
  - 情緒檢測和相應回應
  - 主動關懷訊息
  - 對話歷史記憶

### 2. Discord 整合 (`discord_api_Aibot.py`)
- **核心功能**：Discord Bot 整合，支援即時聊天
- **特色功能**：
  - 個人專用模式（只回應指定用戶）
  - 定時主動訊息
  - 豐富的管理命令
  - 訊息佇列處理

### 3. 回應過濾系統 (`response_filters/`)
- **`content_cleaner.py`**：清理回應內容，移除重複詞語
- **`character_confusion_filter.py`**：防止角色混淆
- **`dialogue_format_filter.py`**：格式化對話結構
- **`quality_validator.py`**：驗證回應品質
- **`sweetness_enhancer.py`**：增強親暱度表達

### 4. 語義分析系統 (`semantic_analysis/`)
- **`emotion_analyzer.py`**：分析用戶情緒狀態
- **`intent_recognizer.py`**：識別用戶意圖
- **`intimacy_calculator.py`**：計算親密度等級
- **`context_analyzer.py`**：分析對話上下文

### 5. 回應模組系統 (`rushia_responses/`)
- **`intimate_responses.py`**：親密對話回應
- **`food_responses.py`**：食物相關話題
- **`emotional_support.py`**：情感支持回應
- **`daily_chat.py`**：日常閒聊
- **`time_aware_responses.py`**：時間感知回應

### 6. 記憶管理 (`memory_management/`)
- **`conversation_history.py`**：對話歷史管理
- **`user_profile.py`**：用戶資料管理
- **`context_cache.py`**：上下文快取

## 🛠️ 輔助工具

### RagTag 爬蟲工具 (`ragtag.py`)
用於從 RagTag 網站自動下載露西亞的直播影片和聊天記錄

**使用方法：**
```bash
python ragtag.py
# 輸入頻道 ID 和頁面範圍
# 自動下載影片音頻和聊天記錄
```

**功能特色：**
- 批量下載指定頁面範圍的影片
- 自動提取聊天記錄
- 安全的檔案命名處理
- 支援斷點續傳

### 聊天記錄分析 (`analyze_chat_logs_fixed.py`)
分析直播聊天記錄，提供訓練數據建議

**使用方法：**
```bash
python analyze_chat_logs_fixed.py
# 選擇分析模式：
# 1. 整體統計分析
# 2. 個別直播分析
```

**分析內容：**
- 觀眾互動模式
- 熱門詞彙統計
- 情緒分佈分析
- 訓練數據建議

### ASMR 翻譯工具 (`asmr_nmt_translate.py`)
將日文 ASMR 內容翻譯成中文

**使用方法：**
```bash
python asmr_nmt_translate.py
# 或測試模式
python asmr_nmt_translate.py --test
```

### 模型管理器 (`model_manager.py`)
管理多個 LoRA 模型版本

**功能：**
- 列出所有模型
- 備份重要模型
- 刪除過期模型
- 顯示模型資訊

## ⚙️ 重要配置說明

### 🚨 路徑配置（必須修改）
**請務必修改以下文件中的路徑以符合您的系統：**

1. **`config_llm.json`** 和 **`config_nmt.json`**
   - 修改模型路徑 `model_path`
   - 修改 llama.cpp 路徑（如果使用）
   - 確保模型文件確實存在於指定路徑

2. **各 Python 文件中的硬編碼路徑**
   - `ragtag.py`：修改 `base_dir` 和 `driver_path`
   - `analyze_chat_logs_fixed.py`：修改 `chat_log_dir`
   - `asmr_nmt_translate.py`：修改 `INPUT_DIR` 和 `OUT_DIR`
   - `discord_api_Aibot.py`：檢查模型路徑設定
   - `chat_asmr.py`：確認模型配置路徑

3. **環境變數設定**
   - 創建 `.env` 文件並設定 Discord Token
   - 確保 `DISCORD_TOKEN` 和 `OWNER_ID` 正確填入

**⚠️ 常見錯誤：**
- 路徑使用反斜線 `\` 而非正斜線 `/`
- 模型文件不存在或路徑錯誤
- 忘記修改配置文件中的絕對路徑

### 🔥 必備模型下載
**重要：使用前必須先下載以下模型**

#### 基礎模型下載：
- **底模型**：Qwen3-8B
- 下載連結：[HuggingFace](https://huggingface.co/Qwen/Qwen3-8B)
  
- **翻譯模型**：Sakura-14B-Qwen2.5-v1.0 (GGUF 格式)
  - 下載連結：[HuggingFace](https://huggingface.co/SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF)
  - 檔案名稱：`sakura-14b-qwen2.5-v1.0-iq4xs.gguf`

#### 露西亞專用 LoRA 模型：
- **Rushia LoRA 適配器**：需要向作者索取
  - 📧 **聯絡作者獲取 LoRA 模型文件**
  - 檔案格式：`.safetensors` 或 `.bin`
  - 此 LoRA 基於 Qwen3-8B 訓練，包含露西亞的個性和語言風格

**模型放置位置：**
```
D:/RushiaMode/models/
├── qwen3-8b #資料夾
├── sakura-14b-qwen2.5-v1.0-iq4xs.gguf   # 翻譯模型
└── rushia-qwen3-8b-lora-asmr-8bit #lora
```

**模型大小說明：**
- Qwen3-8B-Instruct (Q4_K_M)：約 4.5GB
- Sakura-14B-Qwen2.5-v1.0 (IQ4_XS)：約 8.2GB
- Rushia LoRA 適配器：約 4.5GB
- 總計需要約 20-30GB 儲存空間

## 🎯 高級功能

### 特定時間觸發
- **日常問候** 8 15 30 45 60 都會主動關心(如果沒回應對談)
- **早安問候**：每日 6-10 點自動發送
- **晚安關懷**：每日 22-24 點主動關心
- **用餐提醒**：用餐時間主動詢問 中餐11-12 下午茶2-4 晚餐 5-7
- **長時間無互動提醒**：超過設定時間自動發送關懷訊息

### 智能過濾
- **重複詞語過濾**：避免回應中出現重複內容
- **角色一致性檢查**：確保回應符合露西亞的人設
- **品質驗證**：自動檢測和改善低品質回應
- **不當內容過濾**：過濾可能的不適當內容

### 個性化回應
- **情緒適配**：根據用戶情緒調整回應風格
- **親密度系統**：隨互動增加親密度，影響回應內容
- **記憶系統**：記住用戶偏好和重要對話
- **上下文理解**：結合對話歷史產生連貫回應

## 📝 開發和貢獻
感謝我自己 還有sakura模型提供翻譯模型

### 添加新的回應模組
1. 在 `rushia_responses/` 目錄創建新模組
2. 繼承 `BaseModule` 類
3. 在 `__init__.py` 中註冊模組

### 添加新的過濾器
1. 在 `response_filters/` 目錄創建新過濾器
2. 繼承 `BaseResponseFilter` 類
3. 在 `FilterManager` 中註冊

## ⚠️ 注意事項

1. **首次啟動較慢**：模型載入需要時間，請耐心等待
2. **記憶體需求**：確保有足夠的 GPU 記憶體
3. **路徑配置**：務必修改所有配置文件中的路徑
4. **Discord Token**：妥善保管 Discord Bot Token
5. **模型下載**：某些模型檔案較大，請確保網路穩定

## 🔗 相關連結

- [Discord Developer Portal](https://discord.com/developers/applications)
- [HuggingFace Models](https://huggingface.co/models)
- [RagTag Archive](https://archive.ragtag.moe)

## 📄 授權

本專案僅供學習和研究使用，請遵守相關的使用條款和版權規定。

---

**享受與露西亞的對話時光！** 🌸✨
