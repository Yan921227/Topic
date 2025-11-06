"""
簡易版影片辨識程式
直接修改下面的設定即可使用
"""
from ultralytics import YOLO
import cv2
from pathlib import Path
import torch

# ============= 設定區 =============
VIDEO_PATH = "C:\\Users\\User\\Desktop\\IMG_1159.mp4"              # 輸入影片路徑
MODEL_PATH = "runs\\detect\\train3\\weights\\best.pt"  # 模型路徑
OUTPUT_DIR = "output_frames\\20251106m_model"                # 輸出目錄

# 統一信心度（用於初步過濾，建議設低一點）
BASE_CONFIDENCE = 0.25

# 針對不同類別設定各自的信心度閾值
CLASS_CONFIDENCE = {
    0: 0.85,  # backswing（後擺動作）
    1: 0.62,  # impact（擊球瞬間）
}

SAVE_INTERVAL = 1                           # 每隔幾幀存一張 (1=每幀都存)
# ==================================

if __name__ == "__main__":
    # 建立輸出目錄
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 載入模型
    print(f"🔄 載入模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # 開啟影片
    print(f"🔄 開啟影片: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("❌ 無法開啟影片檔案！請檢查路徑是否正確")
        exit()
    
    # 取得影片資訊
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📊 影片資訊:")
    print(f"   總幀數: {total_frames}")
    print(f"   FPS: {fps}")
    print(f"   解析度: {width}x{height}")
    print(f"   預計輸出: {total_frames // SAVE_INTERVAL} 張圖片")
    print(f"\n🚀 開始辨識...\n")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 根據間隔決定是否處理
        if frame_count % SAVE_INTERVAL != 0:
            continue
        
        # 進行辨識（使用較低的基礎信心度）
        results = model(frame, conf=BASE_CONFIDENCE, verbose=False)
        
        # 根據各類別的信心度閾值過濾結果
        boxes = results[0].boxes
        
        if boxes is not None and len(boxes) > 0:
            filtered_indices = []
            
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])  # 類別ID
                confidence = float(box.conf[0])  # 信心度
                
                # 檢查這個類別是否有設定專屬閾值
                if class_id in CLASS_CONFIDENCE:
                    threshold = CLASS_CONFIDENCE[class_id]
                else:
                    threshold = BASE_CONFIDENCE  # 沒設定就用基礎閾值
                
                # 只保留符合該類別閾值的偵測結果
                if confidence >= threshold:
                    filtered_indices.append(i)
            
            # 如果有符合條件的框，保留它們；否則清空
            if filtered_indices:
                # 保留符合條件的 boxes
                import torch
                boxes.data = boxes.data[filtered_indices]
            else:
                # 清空所有 boxes
                boxes.data = boxes.data[:0]
        
        # 繪製結果
        annotated_frame = results[0].plot()
        
        # 儲存圖片
        output_filename = output_path / f"frame_{frame_count:06d}.jpg"
        cv2.imwrite(str(output_filename), annotated_frame)
        saved_count += 1
        
        # 顯示進度
        if saved_count % 10 == 0:
            progress = (frame_count / total_frames) * 100
            detections = len(results[0].boxes)
            print(f"進度: {progress:.1f}% | 已儲存: {saved_count} 張 | 偵測到: {detections} 個物件")
    
    cap.release()
    
    print(f"\n✅ 完成！")
    print(f"📁 已儲存 {saved_count} 張圖片到: {output_path.absolute()}")

