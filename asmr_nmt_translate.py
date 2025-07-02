import os
import json
import subprocess
from pathlib import Path
import argparse
import requests
import re

# 讀取 NMT 設定
with open(r'd:/RushiaMode/scripts/config_nmt.json', encoding='utf-8') as f:
    config = json.load(f)

MODEL_PATH = config['model_path']
LLAMA_CPP_PATH = config['llama_cpp_path']
TEMPERATURE = str(config['temperature'])
PROMPT_TEMPLATE = config['prompt_template']
GPU_LAYERS = str(config.get('gpu_layers', 32))  # 預設 32 層
MAX_TOKENS = str(config['max_tokens'])

# 目錄設定
INPUT_DIR = r'd:/RushiaMode/text/雜談'
OUT_DIR = r'd:/RushiaMode/data/nmt/雜談'
os.makedirs(OUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='只翻譯單一測試檔案前三條')
args = parser.parse_args()

SAKURA_API_URL = config.get('sakura_api_url')

# 載入詞彙庫
VOCAB_PATH = r'd:/RushiaMode/rushia_wiki/rushia_ja_zh_vocab.json'
if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, encoding='utf-8') as f:
        vocab_list = json.load(f)
else:
    vocab_list = []

def build_gpt_dict_text():
    gpt_dict_text_list = []
    for v in vocab_list:
        ja = v.get('ja', '').strip()
        zh = v.get('zh', '').strip()
        if ja and zh:
            gpt_dict_text_list.append(f"{ja}->{zh}")
    return '\n'.join(gpt_dict_text_list)

def log_external_api(payload):
    log_path = r'd:/RushiaMode/scripts/api_external.log'
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2) + '\n---\n')

def postprocess_translation(text):
    # 去除所有連續重複詞語（2字以上，最多保留1次）
    text = re.sub(r'((.{2,10}?))\1{1,}', r'\1', text)
    # 針對「想讓你開心」等常見詞語再去重
    text = re.sub(r'(想让你开心){2,}', r'想让你开心', text)
    text = re.sub(r'(男朋友の?){2,}', r'男朋友', text)
    # 針對「謝謝」等詞語
    text = re.sub(r'(謝謝[你妳大家]*){2,}', r'\1', text)
    # 針對標點
    text = re.sub(r'(，){2,}', '，', text)
    text = re.sub(r'(。){2,}', '。', text)
    return text

def sakura_translate(ja):
    system_prompt = (
        "你是一个輕小說翻譯模型，請流暢通順地以日本輕小說的風格將日文翻譯成繁體中文。"
        "請避免重複同一句話或詞語，遇到重複詞語只需翻譯一次，請精簡且自然地翻譯。"
    )
    gpt_dict_raw_text = build_gpt_dict_text()
    user_prompt = f"根据以下术语表（可以为空）：\n{gpt_dict_raw_text}\n将下面的日文文本根据对应关系和备注翻译成中文：{ja}"
    if SAKURA_API_URL:
        payload = {
            "model": "sakura",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": float(TEMPERATURE),
            "max_tokens": int(MAX_TOKENS)
        }
        log_external_api(payload)
        try:
            resp = requests.post(SAKURA_API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            zh = resp.json()['choices'][0]['message']['content'].strip()
            zh = postprocess_translation(zh)
        except Exception as e:
            zh = f"[API ERROR] {e}"
    else:
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        result = subprocess.run([
            os.path.join(LLAMA_CPP_PATH, 'llama-cli.exe'),
            '-m', MODEL_PATH,
            '--gpu-layers', '999',
            '--n-gpu-layers', '999',
            '--temp', TEMPERATURE,
            '--n-predict', str(MAX_TOKENS),
            '-p', prompt
        ], capture_output=True, encoding='utf-8')
        zh = result.stdout.strip().replace('\r', '').replace('\n', '')
        zh = postprocess_translation(zh)
    return zh

if args.test:
    # 測試模式
    test_file = os.path.join(INPUT_DIR, '！自己紹介！ (2).json')
    out_file = os.path.join(OUT_DIR, 'test_！自己紹介！.json')
    with open(test_file, encoding='utf-8') as f:
        data = json.load(f)
    out_data = []
    total = len(data[:3])
    for idx, seg in enumerate(data[:3]):
        ja = seg.get('transcript', '')
        emotion = seg.get('emotion', '')
        starttime = seg.get('starttime', None)
        endtime = seg.get('endtime', None)
        audio_file = seg.get('audio_file', None)
        chunk_file = seg.get('chunk_file', None)
        if not ja:
            continue
        zh = sakura_translate(ja)
        out_data.append({
            'ja': ja,
            'zh_hant': zh,
            'emotion': emotion,
            'starttime': starttime,
            'endtime': endtime,
            'audio_file': audio_file,
            'chunk_file': chunk_file
        })
        # 分段儲存
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        percent = (idx + 1) / total * 100
        print(f'進度: {idx+1}/{total} ({percent:.1f}%)', end='\r', flush=True)
    print('\n測試翻譯完成，結果已輸出到', out_file)
else:
    # 批次模式
    for file in os.listdir(INPUT_DIR):
        if not file.endswith('.json'):
            continue
        in_path = os.path.join(INPUT_DIR, file)
        out_path = os.path.join(OUT_DIR, file)
        # 若已存在且已完整處理，直接跳過
        if os.path.exists(out_path):
            try:
                with open(out_path, encoding='utf-8') as f:
                    out_data = json.load(f)
                with open(in_path, encoding='utf-8') as f:
                    in_data = json.load(f)
                if len(out_data) == len(in_data):
                    print(f'{file} 已完整處理，跳過')
                    continue
                else:
                    print(f'{file} 檔案不完整，將繼續處理')
            except Exception:
                print(f'{file} 輸出檔案損壞，將重新處理')
        with open(in_path, encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        # 若有部分已處理，繼續接續
        out_data = []
        start_idx = 0
        if os.path.exists(out_path):
            try:
                with open(out_path, encoding='utf-8') as f:
                    out_data = json.load(f)
                start_idx = len(out_data)
            except Exception:
                out_data = []
                start_idx = 0
        total = len(data)
        for idx in range(start_idx, total):
            seg = data[idx]
            ja = seg.get('transcript', '')
            emotion = seg.get('emotion', '')
            starttime = seg.get('starttime', None)
            endtime = seg.get('endtime', None)
            audio_file = seg.get('audio_file', None)
            chunk_file = seg.get('chunk_file', None)
            if not ja:
                continue
            zh = sakura_translate(ja)
            out_data.append({
                'ja': ja,
                'zh_hant': zh,
                'emotion': emotion,
                'starttime': starttime,
                'endtime': endtime,
                'audio_file': audio_file,
                'chunk_file': chunk_file
            })
            # 分段儲存
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(out_data, f, ensure_ascii=False, indent=2)
            percent = (idx + 1) / total * 100
            print(f'{file} 進度: {idx+1}/{total} ({percent:.1f}%)', end='\r', flush=True)
        print(f'\n{file} 翻譯完成，結果已輸出到 {out_path}')
    print('全部翻譯完成！')
print('Done.')
