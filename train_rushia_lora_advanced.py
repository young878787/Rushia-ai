import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import json
from pathlib import Path
import os
import gc
import random
import re
import warnings

# 全局 bfloat16 優化設置
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 強制全局使用 bfloat16（RTX 5070 最佳配置）
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    torch.set_default_dtype(torch.bfloat16)
    GLOBAL_DTYPE = torch.bfloat16
    print("✅ 全局設置為 bfloat16（RTX 5070 優化）")
else:
    torch.set_default_dtype(torch.float16)
    GLOBAL_DTYPE = torch.float16
    print("⚠️ 降級為 float16（兼容模式）")

# 量化警告配置（設為 False 以顯示所有警告）
SUPPRESS_QUANTIZATION_WARNINGS = False
if SUPPRESS_QUANTIZATION_WARNINGS:
    warnings.filterwarnings("ignore", message="MatMul8bitLt")
    print("🔇 已抑制 INT8 量化警告")
else:
    print("🔊 顯示所有量化警告（幫助監控訓練過程）")

# RTX 5070 特定優化建議
print("🚀 RTX 5070 優化建議：INT8量化 + gradient_checkpointing + paged_adamw_32bit + 智能精度選擇")

# 智能精度配置檢測
def detect_best_precision():
    """檢測 RTX 5070 的最佳精度配置"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        bf16_supported = torch.cuda.is_bf16_supported()
        
        print(f"🔍 GPU: {device_name}")
        print(f"🔍 計算能力: {compute_capability}")
        print(f"🔍 bfloat16 支援: {bf16_supported}")
        
        # RTX 5070 優化決策 - 強制使用 bfloat16
        if bf16_supported:
            print("✅ 使用 bfloat16 → int8 → bfloat16 流程")
            return {
                "use_bf16": True,
                "use_fp16": False,
                "torch_dtype": torch.bfloat16,
                "compute_dtype": torch.bfloat16  # 強制 bfloat16
            }
        else:
            print("⚠️ 降級到 float16 兼容模式")
            return {
                "use_bf16": False,
                "use_fp16": True,
                "torch_dtype": torch.float16,
                "compute_dtype": torch.float16
            }
    else:
        return {
            "use_bf16": False,
            "use_fp16": False,
            "torch_dtype": torch.float32,
            "compute_dtype": torch.float32
        }

# 全局精度配置
PRECISION_CONFIG = detect_best_precision()
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import json
from pathlib import Path
import os
import gc
import random
import re

# Windows 兼容性檢查
import platform
IS_WINDOWS = platform.system() == "Windows"

# Flash Attention 在 Windows 上不支援，直接跳過
print("🪟 使用標準 attention（Windows 兼容）")

# 檢查記憶體優化功能
OPTIMUM_AVAILABLE = False
ACCELERATE_AVAILABLE = False

try:
    import optimum
    OPTIMUM_AVAILABLE = True 
    print("✅ Optimum 可用，將啟用額外記憶體優化")
except ImportError:
    print("💡 建議安裝 optimum: pip install optimum")

try:
    import accelerate
    ACCELERATE_AVAILABLE = True
    print("✅ Accelerate 可用")
except ImportError:
    print("💡 建議安裝 accelerate: pip install accelerate")

# RTX 5070 特定優化建議
print("� RTX 5070 優化建議：gradient_checkpointing + paged_adamw_32bit + bfloat16優化器狀態")

class RushiaLLMTrainer:
    def __init__(self, 
                 model_path="D:/RushiaMode/models/Qwen3-8B",
                 resume_from_checkpoint=None,
                 data_category="all",
                 enable_data_augmentation=True):
        self.model_path = model_path
        self.resume_from_checkpoint = resume_from_checkpoint
        self.data_category = data_category  # "chat", "asmr", "roleplay", "all"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_data_augmentation = enable_data_augmentation
        
        # 初始化資料增強器
        if self.enable_data_augmentation:
            self.data_augmenter = DataAugmentation()
            print("✅ 資料增強已啟用")
        
    def load_model_and_tokenizer(self):
        """載入模型和分詞器 - bfloat16→8bit量化版本"""
        print("載入模型和分詞器（bfloat16→8bit量化版本）...")
        
        # 載入分詞器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True,
            cache_dir=self.model_path
        )
        
        # 設置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # 配置8bit量化 - bfloat16 → int8 → bfloat16 流程
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,                # LLM.int8() 閾值
            llm_int8_has_fp16_weight=False,        # 不保留 fp16 權重
            llm_int8_enable_fp32_cpu_offload=False, # 不使用 CPU offload
            # 強制使用 bfloat16 作為計算精度（RTX 5070 優化）
            # 注意：bitsandbytes 可能仍會在內部轉換，但我們指定 bfloat16
        )
        
        # 載入模型 - RTX 5070 智能配置（Qwen3-8B + 8bit量化）
        model_kwargs = {
            "trust_remote_code": True,
            "quantization_config": bnb_config,
            "torch_dtype": PRECISION_CONFIG["torch_dtype"],  # 智能精度選擇
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "max_memory": {0: "10GB"}  # 8bit量化記憶體配置
        }
        
        print("🚀 RTX 5070 環境：bfloat16 → INT8 → bfloat16 完整流程")
        
        # 訓練時的通用優化
        model_kwargs["use_cache"] = False  # 訓練時禁用KV cache節省顯存
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs
        )
        
        # 準備模型進行bfloat16→8bit訓練 - 關鍵步驟！
        self.model = prepare_model_for_kbit_training(
            self.model, 
            use_gradient_checkpointing=True  # 明確啟用梯度檢查點
        )
        
        # 確保模型使用正確的精度
        print(f"🎯 模型權重精度: {next(self.model.parameters()).dtype}")
        
        # 啟用梯度檢查點來節省顯存
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("✅ 梯度檢查點已啟用")
        
        # 標準attention配置完成
        print("✅ bfloat16 → INT8 → bfloat16 流程配置完成")
        
        print("模型載入完成（bfloat16 → 8bit量化 → bfloat16 輸出）！")
        
    def setup_lora(self, existing_adapter_path=None):
        """設置LoRA配置 - 8bit量化版本"""
        if existing_adapter_path and os.path.exists(existing_adapter_path):
            print(f"載入現有LoRA適配器: {existing_adapter_path}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, existing_adapter_path)
        else:
            print("創建新的LoRA適配器")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=64,  # 官方QLoRA論文推薦64
                lora_alpha=16,  # alpha通常設為r/4
                lora_dropout=0.1,  # 官方推薦0.1
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                              "gate_proj", "up_proj", "down_proj"],
                bias="none",
                modules_to_save=None,
                use_rslora=False,  # 可選：啟用RSLoRA
                use_dora=False,    # 可選：啟用DoRA
            )
            
            self.model = get_peft_model(self.model, lora_config)
        
        # 啟用訓練模式
        self.model.train()
        self.model.print_trainable_parameters()
        
    def load_training_data(self, data_file=None, max_samples=1000, shuffle=True):
        """載入訓練數據 - 支持分類別數據文件，資料混洗和增強"""
        print(f"載入訓練數據 - 類別: {self.data_category}")
        
        # 根據類別自動選擇對應的數據文件
        if data_file is None:
            if self.data_category == "asmr":
                data_file = "D:/RushiaMode/training_data/by_category/rushia_ASMR_training.jsonl"
            elif self.data_category == "chat":
                data_file = "D:/RushiaMode/training_data/by_category/rushia_雜談_training.jsonl"
            elif self.data_category == "roleplay":
                data_file = "D:/RushiaMode/training_data/by_category/rushia_roleplay_training.jsonl"
            else:  # all or other categories
                data_file = "D:/RushiaMode/training_data/rushia_mixed_training.jsonl"
        
        print(f"使用數據文件: {data_file}")
        
        # 檢查文件是否存在
        if not os.path.exists(data_file):
            print(f"錯誤: 數據文件不存在 - {data_file}")
            print("請先運行 prepare_training_data_enhanced.py 生成訓練數據")
            return []
        
        conversations = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_count, line in enumerate(f):
                if line_count >= max_samples:
                    break
                data = json.loads(line)
                conversations.append(data)
        
        print(f"載入數據量: {len(conversations)} (限制: {max_samples})")
        
        # 資料混洗
        if shuffle:
            random.shuffle(conversations)
            print("✅ 資料已混洗")
        
        # 轉換為訓練格式並應用資料增強
        texts = []
        augmented_count = 0
        
        for conv in conversations:
            text = ""
            for turn in conv["conversations"]:
                if turn["from"] == "human":
                    user_text = turn['value']
                    # 對用戶輸入進行增強
                    if self.enable_data_augmentation:
                        user_text = self.data_augmenter.augment_text(user_text)
                    text += f"用戶: {user_text}\n"
                else:
                    assistant_text = turn['value']
                    # 對助理回應進行增強
                    if self.enable_data_augmentation:
                        original_text = assistant_text
                        assistant_text = self.data_augmenter.augment_text(assistant_text)
                        if assistant_text != original_text:
                            augmented_count += 1
                    text += f"るしあ: {assistant_text}\n"
            text += self.tokenizer.eos_token
            texts.append(text)
        
        if self.enable_data_augmentation:
            print(f"✅ 資料增強完成，增強了 {augmented_count} 條回應")
        
        return texts    
    def tokenize_data(self, texts, max_length=512):
        """分詞處理 - 支援動態 padding"""
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,  # 關鍵：不在這裡 padding，讓 DataCollator 處理
                max_length=max_length,
                return_tensors=None
            )
            # 對於語言模型，labels就是input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        dataset = Dataset.from_dict({"text": texts})
        
        # 啟用資料混洗
        dataset = dataset.shuffle(seed=42)
        print("✅ Dataset 層級混洗已啟用")
        
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=["text"],
            batch_size=50  # 批處理大小
        )
        
        print(f"✅ 分詞完成，序列最大長度: {max_length}")
        return tokenized_dataset
    
    def train(self, output_dir=None, epochs=5, max_samples=2000, max_seq_length=512):
        """開始訓練 - INT8量化優化版本，支援動態padding和資料混洄"""
        if output_dir is None:
            output_dir = f"D:/RushiaMode/models/rushia-qwen3-8b-lora-{self.data_category}-8bit"
        
        print(f"🚀 開始INT8量化訓練 - 類別: {self.data_category}")
        print(f"📁 輸出目錄: {output_dir}")
        print(f"📏 最大序列長度: {max_seq_length}")
        print(f"🎯 精度配置: {PRECISION_CONFIG}")
        
        # GPU 和記憶體診斷信息
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 GPU: {gpu_name}")
            print(f"💾 總記憶體: {gpu_memory:.1f}GB")
            print(f"💾 當前使用: {torch.cuda.memory_allocated(0) / 1024**3:.1f}GB")
        
        # 載入數據（限制樣本數，啟用混洗）
        texts = self.load_training_data(max_samples=max_samples, shuffle=True)
        if not texts:
            print("❌ 沒有載入到訓練數據，退出訓練")
            return None
            
        dataset = self.tokenize_data(texts, max_length=max_seq_length)
        
        # 訓練參數（Windows 優化配置）
        training_args_dict = {
            "output_dir": output_dir,
            "overwrite_output_dir": False,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": 2,  
            "gradient_accumulation_steps": 8,  
            "warmup_steps": 100,               # 更多warmup步數
            "logging_steps": 10,
            "save_steps": 500,
            "save_total_limit": 2,
            "learning_rate": 1e-4,
            "bf16": PRECISION_CONFIG["use_bf16"],      # 智能 bfloat16 選擇
            "fp16": PRECISION_CONFIG["use_fp16"],      # 智能 float16 選擇
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
            "gradient_checkpointing": True,    # 強制啟用梯度檢查點
            "dataloader_num_workers": 4,       # RTX 5070 可以使用更多 workers
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "max_grad_norm": 0.3,             # 官方推薦0.3
            "logging_dir": f"{output_dir}/logs",
            "report_to": [],
            "push_to_hub": False,
            "label_names": ["labels"],
            "optim": "paged_adamw_32bit",     # Paged optimizer for memory efficiency
            "lr_scheduler_type": "cosine",    # 官方推薦cosine調度器
            # 資料混洗設置
            "dataloader_drop_last": False,    # 不丟棄最後一個不完整的batch
            "group_by_length": False,         # 關閉按長度分組，讓動態padding發揮作用
            # 記憶體優化相關設置
            "ddp_find_unused_parameters": False,  # 避免DDP問題
            "save_safetensors": True,             # 使用更安全的保存格式
            # 進階記憶體管理
            "max_steps": -1,                      # 讓epoch控制訓練
            "eval_steps": None,                   # 不進行評估節省時間
            "prediction_loss_only": True,        # 只計算loss節省計算
        }
        
        # 只有在多線程時才添加 prefetch_factor
        if training_args_dict["dataloader_num_workers"] > 0:
            training_args_dict["dataloader_prefetch_factor"] = 2
        
        training_args = TrainingArguments(**training_args_dict)
        
        # 使用動態資料整理器
        data_collator = DynamicDataCollator(
            tokenizer=self.tokenizer,
            mlm=False,
            max_length=max_seq_length
        )
        print("✅ 動態 padding 資料整理器已啟用")
        
        # 自定義訓練器支援每個 epoch 前混洗（簡化版本）
        class ShufflingTrainer(Trainer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.epoch_count = 0
                
            def get_train_dataloader(self):
                """每個 epoch 前重新混洗資料"""
                if self.epoch_count > 0:  # 第一個epoch已經在dataset.shuffle()中混洗過了
                    print(f"🔀 Epoch {self.epoch_count + 1}: 重新混洗訓練資料")
                    self.train_dataset = self.train_dataset.shuffle(seed=42 + self.epoch_count)
                self.epoch_count += 1
                return super().get_train_dataloader()
            
            # 移除複雜的 training_step 覆蓋，使用原生實現
            # Transformers 會自動處理 autocast 和精度管理
        
        # 訓練器
        trainer = ShufflingTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # 開始訓練
        import time
        start_time = time.time()
        print("🚀 開始訓練...")
        print(f"⏰ 開始時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("✅ 每個 epoch 前自動混洗已啟用")
        print("✅ 動態 padding 已啟用（根據 batch 最大長度）")
        print("⚠️ 量化警告顯示已啟用（監控精度轉換）")
        if self.enable_data_augmentation:
            print("✅ 資料增強已啟用")
        
        # 訓練前記憶體狀態
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated(0) / 1024**3
            print(f"💾 訓練前記憶體使用: {memory_before:.1f}GB")
            torch.cuda.empty_cache()
            gc.collect()
            memory_after_cleanup = torch.cuda.memory_allocated(0) / 1024**3
            print(f"💾 清理後記憶體使用: {memory_after_cleanup:.1f}GB")
            print("🧹 GPU記憶體已清理")
        
        trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        
        # 訓練完成統計
        end_time = time.time()
        training_duration = end_time - start_time
        hours = int(training_duration // 3600)
        minutes = int((training_duration % 3600) // 60)
        seconds = int(training_duration % 60)
        
        print(f"⏱️ 訓練耗時: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"⏰ 結束時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 最終記憶體狀態
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(0) / 1024**3
            max_memory = torch.cuda.max_memory_allocated(0) / 1024**3
            print(f"💾 最終記憶體使用: {final_memory:.1f}GB")
            print(f"💾 峰值記憶體使用: {max_memory:.1f}GB")
        
        # 保存最終模型
        print("💾 保存模型...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ 訓練完成！模型已保存到: {output_dir}")
        return output_dir

class DataAugmentation:
    """資料增強類別"""
    
    def __init__(self):
        # 同義詞替換字典（ASMR和聊天相關）
        self.synonyms = {
            "好的": ["好~", "好呢", "好呀", "嗯嗯", "ok", "OK"],
            "聲音": ["音色", "嗓音", "voice"],
            "可愛": ["kawaii", "卡哇伊", "超萌", "好萌"],
            "舒服": ["放鬆", "療癒", "comfort", "relax"],
            "睡覺": ["睡眠", "休息", "sleep", "入眠"],
            "耳朵": ["ears", "耳朵朵", "小耳朵"],
            "輕柔": ["溫柔", "gentle", "soft", "輕軟"],
            "安靜": ["quiet", "寧靜", "peaceful"],
            "謝謝": ["感謝", "thank you", "3Q", "thx"],
            "晚安": ["good night", "おやすみ", "睡個好覺"],
            "呢": ["呀", "哦", "喔", "~"],
            "很": ["非常", "超", "好", "特別"],
            "真的": ["確實", "的確", "really", "真是"],
            "一起": ["together", "共同", "一同"],
            "喜歡": ["愛", "like", "love", "鍾愛"],
        }
        
        # 表情符號增強
        self.emoticons = ["~", "♡", "♪", "☆", "◇", "○", "△", "▽", "♬", "♫"]
        
    def augment_text(self, text, augment_prob=0.3):
        """對文本進行增強"""
        if random.random() > augment_prob:
            return text
            
        # 1. 同義詞替換
        text = self._synonym_replacement(text)
        
        # 2. 隨機插入表情符號
        text = self._add_emoticons(text)
        
        # 3. 標點符號變化
        text = self._punctuation_variation(text)
        
        return text
    
    def _synonym_replacement(self, text):
        """同義詞替換"""
        for word, synonyms in self.synonyms.items():
            if word in text and random.random() < 0.3:
                replacement = random.choice(synonyms)
                text = text.replace(word, replacement, 1)  # 只替換第一個
        return text
    
    def _add_emoticons(self, text):
        """添加表情符號"""
        if random.random() < 0.2:  # 20% 機率添加表情
            emoticon = random.choice(self.emoticons)
            if text.endswith(('。', '！', '？', '~', '♡')):
                text = text + emoticon
            else:
                text = text + emoticon
        return text
    
    def _punctuation_variation(self, text):
        """標點符號變化"""
        # 隨機將句號改為其他符號
        if random.random() < 0.2:
            text = text.replace('。', random.choice(['~', '♡', '呢~', '哦~']))
        return text

class DynamicDataCollator(DataCollatorForLanguageModeling):
    """動態 padding 的資料整理器 - bfloat16 優化版本"""
    
    def __init__(self, tokenizer, mlm=False, max_length=None):
        super().__init__(tokenizer, mlm)
        self.max_length = max_length
        
    def __call__(self, examples):
        # 找出 batch 中最長的序列
        if self.max_length:
            max_len = min(max([len(ex['input_ids']) for ex in examples]), self.max_length)
        else:
            max_len = max([len(ex['input_ids']) for ex in examples])
        
        # 動態 padding 到 batch 最大長度
        batch = []
        for example in examples:
            input_ids = example['input_ids'][:max_len]  # 截斷
            attention_mask = example['attention_mask'][:max_len]
            labels = example['labels'][:max_len]
            
            # Padding 到 batch 最大長度
            pad_length = max_len - len(input_ids)
            if pad_length > 0:
                input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
                labels.extend([-100] * pad_length)
            
            batch.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })
        
        # 轉換為 tensor，並確保使用全局 DTYPE
        result = {
            'input_ids': torch.tensor([ex['input_ids'] for ex in batch], dtype=torch.long),
            'attention_mask': torch.tensor([ex['attention_mask'] for ex in batch], dtype=torch.long),
            'labels': torch.tensor([ex['labels'] for ex in batch], dtype=torch.long)
        }
        
        # 對於需要浮點運算的張量，確保使用 bfloat16
        # input_ids, attention_mask, labels 保持 long 類型
        # 其他計算會自動使用全局設定的 bfloat16
        
        return result

def check_gpu_memory():
    """檢查GPU顯存情況"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # GB
            print(f"GPU {i}: {props.name}")
            print(f"  總顯存: {total_memory:.1f} GB")
            
            if torch.cuda.is_initialized():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"  已分配: {allocated:.1f} GB")
                print(f"  已緩存: {cached:.1f} GB")
                print(f"  可用: {total_memory - cached:.1f} GB")
            
            return total_memory
    else:
        print("未檢測到CUDA GPU")
        return 0

def optimize_memory_settings():
    """優化記憶體設置"""
    if torch.cuda.is_available():
        # 啟用記憶體分片
        torch.cuda.empty_cache()
        
        # 設置記憶體分配策略
        if hasattr(torch.cuda, 'set_memory_fraction'):
            torch.cuda.set_memory_fraction(0.9)  # 使用90%顯存
            
        # 啟用cudnn benchmark
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # 優化 CUDNN
        torch.backends.cuda.matmul.allow_tf32 = True  # 啟用 TF32（RTX 5070 支援）
        
        print("✅ 記憶體優化設置完成")
        print("� 使用標準attention + gradient_checkpointing + paged_adamw_32bit 優化顯存")
            
        return True
    return False

def optimize_for_rtx5070():
    """RTX 5070 專用優化設置"""
    print("� 應用 RTX 5070 專用優化...")
    
    # RTX 5070 記憶體管理優化
    if torch.cuda.is_available():
        # 清理 GPU 記憶體
        torch.cuda.empty_cache()
        
        # 設置記憶體分配策略（RTX 5070 12GB 優化）
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
            
        # Windows 下關閉一些可能有問題的優化
        torch.backends.cudnn.benchmark = False  # Windows下可能不穩定
        torch.backends.cudnn.deterministic = True
        
        # 設置更保守的線程數
        torch.set_num_threads(min(4, torch.get_num_threads()))
        
        print("✅ Windows 優化設置完成")
    else:
        print("🐧 Linux/Unix 環境：啟用高性能設置")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def train_asmr_lora(epochs=3, max_samples=1000):
    """專門訓練ASMR LoRA模型"""
    print("🎧 開始訓練ASMR專用LoRA模型 (4bit量化)")
    print("="*60)
    
    trainer = RushiaLLMTrainer(data_category="asmr")
    
    try:
        # 載入模型（4bit量化）
        trainer.load_model_and_tokenizer()
        
        # 設置LoRA
        trainer.setup_lora()
        
        # 開始訓練
        model_path = trainer.train(epochs=epochs, max_samples=max_samples)
        
        if model_path:
            print(f"\n🎉 ASMR LoRA模型訓練完成！")
            print(f"📁 模型保存在: {model_path}")
            return model_path
        else:
            print("❌ 訓練失敗")
            return None
            
    except Exception as e:
        print(f"❌ 訓練過程中出現錯誤: {e}")
        return None

def train_chat_lora(epochs=3, max_samples=1000):
    """專門訓練聊天LoRA模型"""
    print("💬 開始訓練聊天專用LoRA模型 (4bit量化)")
    print("="*60)
    
    trainer = RushiaLLMTrainer(data_category="chat")
    
    try:
        # 載入模型（4bit量化）
        trainer.load_model_and_tokenizer()
        
        # 設置LoRA
        trainer.setup_lora()
        
        # 開始訓練
        model_path = trainer.train(epochs=epochs, max_samples=max_samples)
        
        if model_path:
            print(f"\n🎉 聊天LoRA模型訓練完成！")
            print(f"📁 模型保存在: {model_path}")
            return model_path
        else:
            print("❌ 訓練失敗")
            return None
            
    except Exception as e:
        print(f"❌ 訓練過程中出現錯誤: {e}")
        return None

def train_all_available_categories():
    """訓練所有可用的分類數據 - 4bit量化版本"""
    # 檢查可用的分類數據文件
    category_files = {
        "asmr": "D:/RushiaMode/training_data/by_category/rushia_ASMR_training.jsonl",
        "chat": "D:/RushiaMode/training_data/by_category/rushia_雜談_training.jsonl",
        "mixed": "D:/RushiaMode/training_data/rushia_mixed_training.jsonl"
    }
    
    available_categories = []
    for cat, file_path in category_files.items():
        if os.path.exists(file_path):
            available_categories.append(cat)
            print(f"✅ 發現 {cat} 訓練數據: {file_path}")
        else:
            print(f"❌ 未找到 {cat} 訓練數據: {file_path}")
    
    if not available_categories:
        print("❌ 沒有找到任何訓練數據文件！")
        print("請先運行 prepare_training_data_enhanced.py 生成訓練數據")
        return
    
    print(f"\n發現 {len(available_categories)} 個可用類別: {', '.join(available_categories)}")
    
    # 訓練每個可用類別
    trained_models = []
    for category in available_categories:
        print(f"\n{'='*60}")
        print(f"開始訓練: {category.upper()}")
        print(f"{'='*60}")
        
        try:
            if category == "asmr":
                model_path = train_asmr_lora(epochs=2, max_samples=800)
            elif category == "chat":
                model_path = train_chat_lora(epochs=2, max_samples=800)
            else:  # mixed
                trainer = RushiaLLMTrainer(data_category="all")
                trainer.load_model_and_tokenizer()
                trainer.setup_lora()
                model_path = trainer.train(epochs=2, max_samples=800)
            
            if model_path:
                trained_models.append((category, model_path))
                print(f"✅ {category} 訓練完成")
            else:
                print(f"❌ {category} 訓練失敗")
                
        except Exception as e:
            print(f"❌ {category} 訓練過程中出現錯誤: {e}")
    
    print(f"\n🎉 全部訓練完成！共訓練了 {len(trained_models)} 個模型:")
    for cat, path in trained_models:
        print(f"  📁 {cat}: {path}")

def train_all_available_categories_enhanced(enable_augmentation=True):
    """訓練所有可用的分類數據 - 支援資料增強版本"""
    # 檢查可用的分類數據文件
    category_files = {
        "asmr": "D:/RushiaMode/training_data/by_category/rushia_ASMR_training.jsonl",
        "chat": "D:/RushiaMode/training_data/by_category/rushia_雜談_training.jsonl",
        "mixed": "D:/RushiaMode/training_data/rushia_mixed_training.jsonl"
    }
    
    available_categories = []
    for cat, file_path in category_files.items():
        if os.path.exists(file_path):
            available_categories.append(cat)
            print(f"✅ 發現 {cat} 訓練數據: {file_path}")
        else:
            print(f"❌ 未找到 {cat} 訓練數據: {file_path}")
    
    if not available_categories:
        print("❌ 沒有找到任何訓練數據文件！")
        print("請先運行 prepare_training_data_enhanced.py 生成訓練數據")
        return
    
    print(f"\n發現 {len(available_categories)} 個可用類別: {', '.join(available_categories)}")
    print(f"資料增強: {'啟用' if enable_augmentation else '關閉'}")
    
    # 訓練每個可用類別
    trained_models = []
    for category in available_categories:
        print(f"\n{'='*60}")
        print(f"開始訓練: {category.upper()}")
        print(f"{'='*60}")
        
        try:
            trainer = RushiaLLMTrainer(
                data_category=category if category != "mixed" else "all",
                enable_data_augmentation=enable_augmentation
            )
            trainer.load_model_and_tokenizer()
            trainer.setup_lora()
            model_path = trainer.train(epochs=2, max_samples=800, max_seq_length=384)
            
            if model_path:
                trained_models.append((category, model_path))
                print(f"✅ {category} 訓練完成")
            else:
                print(f"❌ {category} 訓練失敗")
                
        except Exception as e:
            print(f"❌ {category} 訓練過程中出現錯誤: {e}")
    
    print(f"\n🎉 全部訓練完成！共訓練了 {len(trained_models)} 個模型:")
    for cat, path in trained_models:
        print(f"  📁 {cat}: {path}")

def main():
    """主函數"""
    print("🤖 潤羽露西亞 LoRA 訓練系統 (Windows 兼容版)")
    print("="*70)
    
    # Windows 專用優化
    optimize_for_rtx5070()
    print()
    
    # 優化記憶體設置
    print("🔧 優化記憶體設置...")
    optimize_memory_settings()
    print()
    
    # 檢查GPU顯存
    print("檢查系統資源...")
    gpu_memory = check_gpu_memory()
    print()
    
    if gpu_memory < 8:
        print("⚠️  警告: 顯存不足8GB，將使用保守的訓練參數")
        print()
    
    print("選擇訓練模式:")
    print("1. 🎧 ASMR專用LoRA訓練 (推薦)")
    print("2. 💬 聊天專用LoRA訓練")
    print("3. 🚀 自動訓練所有可用分類")
    print("4. 🛠️ 自定義訓練")
    
    choice = input("\n請選擇 (1/2/3/4): ").strip()
    
    # 詢問是否啟用資料增強
    augment_choice = input("\n是否啟用資料增強？(Y/n): ").strip().lower()
    enable_augmentation = augment_choice != 'n'
    
    if choice == "1":
        epochs = int(input("請輸入訓練輪數 (建議2-3): ") or "3")
        max_samples = int(input("請輸入最大樣本數 (建議500-1000): ") or "800")
        max_seq_length = int(input("請輸入訓練時最大序列長度 (建議256-512): ") or "384")
        
        trainer = RushiaLLMTrainer(data_category="asmr", enable_data_augmentation=enable_augmentation)
        trainer.load_model_and_tokenizer()
        trainer.setup_lora()
        trainer.train(epochs=epochs, max_samples=max_samples, max_seq_length=max_seq_length)
        
    elif choice == "2":
        epochs = int(input("請輸入訓練輪數 (建議2-3): ") or "3")
        max_samples = int(input("請輸入最大樣本數 (建議500-1000): ") or "800")
        max_seq_length = int(input("請輸入訓練時最大序列長度 (建議256-512): ") or "384")
        
        trainer = RushiaLLMTrainer(data_category="chat", enable_data_augmentation=enable_augmentation)
        trainer.load_model_and_tokenizer()
        trainer.setup_lora()
        trainer.train(epochs=epochs, max_samples=max_samples, max_seq_length=max_seq_length)
        
    elif choice == "3":
        # 自動訓練所有分類時，默認啟用增強
        print("🚀 自動訓練模式，啟用所有優化功能")
        train_all_available_categories_enhanced(enable_augmentation)
        
    elif choice == "4":
        print("\n可用分類:")
        print("- asmr: ASMR專用模型")
        print("- chat: 日常聊天模型")
        print("- all: 混合數據模型")
        
        category = input("請選擇分類 (asmr/chat/all): ").strip().lower()
        if category in ["asmr", "chat", "all"]:
            epochs = int(input("請輸入訓練輪數 (建議2-5): ") or "3")
            max_samples = int(input("請輸入最大樣本數 (建議500-1500): ") or "1000")
            max_seq_length = int(input("請輸入訓練時最大序列長度 (建議256-512): ") or "384")
            
            trainer = RushiaLLMTrainer(data_category=category, enable_data_augmentation=enable_augmentation)
            trainer.load_model_and_tokenizer()
            trainer.setup_lora()
            trainer.train(epochs=epochs, max_samples=max_samples, max_seq_length=max_seq_length)
        else:
            print("無效的分類選擇")
            
    else:
        print("無效選擇")

if __name__ == "__main__":
    main()
