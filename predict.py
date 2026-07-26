import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.8, verbose=False)
    annotated = results[0].plot()

    h, w = annotated.shape[:2]
    scale = min(900 / w, 700 / h)
    resized = cv2.resize(annotated, (int(w * scale), int(h * scale)))

    cv2.imshow("YOLO Detection", resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
