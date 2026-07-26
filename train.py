from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model.train(data='data.yaml', epochs=5, imgsz=640, batch=16, device="cpu",workers=1)






