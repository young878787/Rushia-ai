import os
import json
import shutil
from pathlib import Path
from datetime import datetime

class RushiaModelManager:
    def __init__(self, models_dir="D:/RushiaMode/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def list_models(self):
        """列出所有可用的模型"""
        models = []
        
        for model_path in self.models_dir.iterdir():
            if model_path.is_dir() and "rushia" in model_path.name.lower():
                # 獲取模型信息
                config_file = model_path / "adapter_config.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    model_info = {
                        "name": model_path.name,
                        "path": str(model_path),
                        "size": self._get_folder_size(model_path),
                        "created": datetime.fromtimestamp(model_path.stat().st_ctime),
                        "category": self._extract_category(model_path.name),
                        "config": config
                    }
                    models.append(model_info)
        
        return sorted(models, key=lambda x: x["created"], reverse=True)
    
    def _get_folder_size(self, folder_path):
        """計算資料夾大小"""
        total_size = 0
        for file_path in folder_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def _extract_category(self, model_name):
        """從模型名稱提取類別"""
        if "chat" in model_name:
            return "chat"
        elif "asmr" in model_name:
            return "asmr"
        elif "roleplay" in model_name:
            return "roleplay"
        else:
            return "all"
    
    def backup_model(self, model_name, backup_dir=None):
        """備份模型"""
        if backup_dir is None:
            backup_dir = self.models_dir / "backups"
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(exist_ok=True)
        
        model_path = self.models_dir / model_name
        if not model_path.exists():
            print(f"模型不存在: {model_name}")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{model_name}_backup_{timestamp}"
        backup_path = backup_dir / backup_name
        
        print(f"正在備份模型: {model_name} -> {backup_name}")
        shutil.copytree(model_path, backup_path)
        print(f"備份完成: {backup_path}")
        
        return backup_path
    
    def merge_models(self, model_list, output_name):
        """合併多個LoRA模型"""
        print(f"合併模型: {model_list} -> {output_name}")
        
        # 這裡需要實現模型合併邏輯
        # 由於LoRA模型合併比較複雜，這裡提供框架
        
        output_path = self.models_dir / output_name
        output_path.mkdir(exist_ok=True)
        
        # TODO: 實現實際的模型合併邏輯
        print("模型合併功能開發中...")
        
        return output_path
    
    def delete_model(self, model_name, confirm=True):
        """刪除模型"""
        model_path = self.models_dir / model_name
        if not model_path.exists():
            print(f"模型不存在: {model_name}")
            return False
        
        if confirm:
            response = input(f"確定要刪除模型 {model_name}？(y/N): ")
            if response.lower() != 'y':
                print("取消刪除")
                return False
        
        print(f"正在刪除模型: {model_name}")
        shutil.rmtree(model_path)
        print("刪除完成")
        
        return True
    
    def create_model_info(self, model_name, description="", tags=None):
        """為模型創建信息文件"""
        if tags is None:
            tags = []
        
        model_path = self.models_dir / model_name
        if not model_path.exists():
            print(f"模型不存在: {model_name}")
            return False
        
        info = {
            "name": model_name,
            "description": description,
            "tags": tags,
            "created": datetime.now().isoformat(),
            "category": self._extract_category(model_name)
        }
        
        info_file = model_path / "model_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        print(f"模型信息已保存: {info_file}")
        return True
    
    def show_model_info(self, model_name):
        """顯示模型詳細信息"""
        models = self.list_models()
        model = next((m for m in models if m["name"] == model_name), None)
        
        if not model:
            print(f"模型不存在: {model_name}")
            return
        
        print(f"\n{'='*50}")
        print(f"模型名稱: {model['name']}")
        print(f"路徑: {model['path']}")
        print(f"大小: {model['size'] / 1024**2:.2f} MB")
        print(f"創建時間: {model['created']}")
        print(f"類別: {model['category']}")
        print(f"{'='*50}")
        
        # 顯示額外信息
        info_file = Path(model['path']) / "model_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
            print(f"描述: {info.get('description', 'N/A')}")
            print(f"標籤: {', '.join(info.get('tags', []))}")

def main():
    """主選單"""
    manager = RushiaModelManager()
    
    while True:
        print("\n=== Rushia 模型管理器 ===")
        print("1. 列出所有模型")
        print("2. 顯示模型詳情")
        print("3. 備份模型")
        print("4. 刪除模型")
        print("5. 創建模型信息")
        print("0. 退出")
        
        choice = input("\n請選擇: ").strip()
        
        if choice == "1":
            models = manager.list_models()
            if not models:
                print("沒有找到模型")
                continue
            
            print(f"\n找到 {len(models)} 個模型:")
            for i, model in enumerate(models, 1):
                print(f"{i}. {model['name']} ({model['category']}) - {model['size']/1024**2:.1f}MB")
        
        elif choice == "2":
            model_name = input("請輸入模型名稱: ").strip()
            manager.show_model_info(model_name)
        
        elif choice == "3":
            model_name = input("請輸入要備份的模型名稱: ").strip()
            manager.backup_model(model_name)
        
        elif choice == "4":
            model_name = input("請輸入要刪除的模型名稱: ").strip()
            manager.delete_model(model_name)
        
        elif choice == "5":
            model_name = input("請輸入模型名稱: ").strip()
            description = input("請輸入模型描述: ").strip()
            tags = input("請輸入標籤 (用逗號分隔): ").strip().split(',')
            tags = [tag.strip() for tag in tags if tag.strip()]
            manager.create_model_info(model_name, description, tags)
        
        elif choice == "0":
            print("再見！")
            break
        
        else:
            print("無效選擇")

if __name__ == "__main__":
    main()
