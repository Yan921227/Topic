"""
快速查看模型的類別資訊
"""
from ultralytics import YOLO

# 載入你的模型
model = YOLO("runs/detect/train/weights/best.pt")

# 顯示類別資訊
print("📋 模型類別資訊：")
print(f"類別數量：{len(model.names)}")
print("\n類別對照表：")
print("-" * 40)
for idx, name in model.names.items():
    print(f"類別 {idx}: {name}")
print("-" * 40)
