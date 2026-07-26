# 🥤 YOLO Coca-Cola & Pepsi Detection

YOLO11n fine-tuned model for detecting Coca-Cola and Pepsi cans.

## 📊 Results

| Metric | Value |
|--------|-------|
| mAP50 | 89.1% |
| mAP50-95 | 64.1% |
| Precision | 71.4% |
| Recall | 90.4% |

| Class | mAP50 | mAP50-95 |
|-------|-------|----------|
| COCACOLA | 99.5% | 83.6% |
| PEPSI | 78.7% | 44.6% |

## 🚀 Installation

```bash
pip install ultralytics opencv-python
```

## 📖 Usage

### Image Detection
```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(source="image.jpg", save=True, conf=0.3)
```

### Video Detection
```python
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, conf=0.3, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🏋️ Training

> The published `best.pt` was trained on Google Colab (GPU, 200 epochs). `train.py` in this repo is a lighter local/CPU config for quick iteration.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="data.yaml", epochs=200, imgsz=640, device=0)
```

## 📁 Dataset Structure

```
dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

## 📦 Export to ONNX

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="onnx")
```

## 🛠️ Requirements

- Python 3.8+
- ultralytics
- opencv-python
- yt-dlp (for YouTube video detection)

## 📝 License

MIT
