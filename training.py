from ultralytics import YOLO
import shutil
from pathlib import Path

if __name__ == "__main__":
    # ============= 設定區 =============
    SAVE_DIR = "my_models"                    # 👈 儲存目錄
    MODEL_NAME = "20251106m_model.pt"         # 👈 模型檔名（可以自己改）
    # ==================================
    
    # 選你要的模型（n / s / m / l / x）
    model = YOLO("yolo11m.pt")

    # 開始訓練
    results = model.train(
        data="data.yaml",   # 你的資料集設定
        epochs=100,
        imgsz=640,
        device=0,           # 用 GPU
        batch=-1,           # 自動尋找最大 batch
        workers=0,          # Windows 必須設為 0 避免 multiprocessing 錯誤
        cache=True,
        patience=30         # 早停
    )

    # 建立自訂目錄
    save_path = Path(SAVE_DIR)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 找到訓練好的最佳模型（best.pt）
    best_model_path = str(model.trainer.save_dir) + "/weights/best.pt"
    
    # 複製到你指定的位置和檔名
    final_model_path = SAVE_DIR + "/" + MODEL_NAME
    shutil.copy2(best_model_path, final_model_path)
    
    # 訓練完成後，自動顯示儲存路徑
    print("\n✅ 訓練完成！模型已自動儲存。")
    print(f"📁 原始訓練輸出：{model.trainer.save_dir}/weights/")
    print(f"📂 你的模型位置：{final_model_path}")
    print(f"✨ 模型檔名：{MODEL_NAME}")
