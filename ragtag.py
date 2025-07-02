import requests
from bs4 import BeautifulSoup
import subprocess
import os
from chatlog import fetch_chatlog_json_url_selenium
import sys

BASE_URL = "https://archive.ragtag.moe"

def fetch_video_links(channel_id, start_page=1, end_page=1):
    all_video_urls = set()

    for page in range(start_page, end_page + 1):
        url = f"{BASE_URL}/channel/{channel_id}?page={page}"
        print(f"抓取第 {page} 頁: {url}")
        try:
            res = requests.get(url)
            res.raise_for_status()
        except Exception as e:
            print(f"錯誤：無法讀取 {url} - {e}")
            continue

        soup = BeautifulSoup(res.text, "html.parser")
        # 先抓 a 標籤，然後用 href 過濾
        links = soup.find_all("a", href=True)
        for a in links:
            href = a['href']
            if href.startswith("/watch") or href.startswith("https://archive.ragtag.moe/watch"):
                if href.startswith("/watch"):
                    full_url = BASE_URL + href
                else:
                    full_url = href
                all_video_urls.add(full_url)

    return sorted(all_video_urls)

def save_links_to_file(links, filename="links.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for url in links:
            f.write(url + "\n")
    print(f"已儲存 {len(links)} 筆連結到 {filename}")

def download_audio_from_links(links_file="links.txt", output_dir="audio"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(links_file, "r", encoding="utf-8") as f:
        links = [line.strip() for line in f if line.strip()]
    for url in links:
        print(f"正在下載音頻: {url}")
        # 使用 yt-dlp 僅下載單一音訊檔案（避免 playlist）
        cmd = [
            "python", "-m", "yt_dlp",
            "-f", "bestaudio",
            "--no-playlist",
            "-o", f"{output_dir}/%(title)s.%(ext)s",
            url
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"下載失敗: {url} - {e}")
    print("全部音頻下載完成！")

def fetch_video_title(url):
    # 用 yt-dlp 取得影片標題
    import yt_dlp
    ydl_opts = {'quiet': True, 'skip_download': True, 'forcejson': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get('title', 'unknown_title')

def process_video(url, driver_path, base_dir):
    print(f"處理影片: {url}")
    # 取得影片標題
    try:
        title = fetch_video_title(url)
    except Exception as e:
        print(f"取得標題失敗: {e}")
        title = 'unknown_title'
    safe_title = ''.join(c for c in title if c not in '\/:*?"<>|')
    out_dir = os.path.join(base_dir, safe_title)
    os.makedirs(out_dir, exist_ok=True)
    # 下載 chat log，最多重試 3 次
    chatlog_url = None
    chatlog_filename = None
    chatlog_path = None
    for attempt in range(3):
        try:
            chatlog_url = fetch_chatlog_json_url_selenium(url, driver_path=driver_path)
            chatlog_filename = os.path.basename(chatlog_url) if chatlog_url else None
            chatlog_path = os.path.join(out_dir, chatlog_filename) if chatlog_filename else None
            break
        except Exception as e:
            print(f"[重試 {attempt+1}/3] chatlog 取得失敗: {e}")
            import time; time.sleep(3)
    need_download_chatlog = chatlog_url and (not chatlog_path or not os.path.exists(chatlog_path))
    if chatlog_url:
        if need_download_chatlog:
            print(f"下載 chat log: {chatlog_url}")
            try:
                resp = requests.get(chatlog_url)
                resp.raise_for_status()
                with open(chatlog_path, 'wb') as f:
                    f.write(resp.content)
                print(f"已儲存 chat log: {chatlog_path}")
            except Exception as e:
                print(f"chat log 下載失敗: {e}")
        else:
            print(f"chat log 已存在: {chatlog_path}")
    else:
        print("找不到 chat log 連結。")
    # 下載音訊
    audio_glob1 = os.path.join(out_dir, '*.m4a')
    audio_glob2 = os.path.join(out_dir, '*.webm')
    import glob
    audio_files = glob.glob(audio_glob1) + glob.glob(audio_glob2)
    need_download_audio = not audio_files
    audio_path = os.path.join(out_dir, '%(title)s.%(ext)s')
    if need_download_audio:
        python_exe = sys.executable
        cmd = [
            python_exe, "-m", "yt_dlp",
            "-f", "bestaudio",
            "--no-playlist",
            "-o", audio_path,
            url
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"音訊下載失敗: {e}")
    else:
        print(f"音訊已存在: {audio_files}")

def main():
    channel_id = input("請輸入頻道 ID（例如 UCl_gCybOJRIgOXw6Qb4qJzQ）: ").strip()
    start_page = int(input("開始頁碼（例如 1）: "))
    end_page = int(input("結束頁碼（例如 3）: "))
    driver_path = r"d:\RushiaMode\chromedriver-win64\chromedriver.exe"
    base_dir = r"D:\RushiaMode\audio"
    links = fetch_video_links(channel_id, start_page, end_page)
    save_links_to_file(links)
    for url in links:
        process_video(url, driver_path, base_dir)
    print("全部影片與 chat log 處理完成！")

if __name__ == "__main__":
    main()
