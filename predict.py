import cv2
from ultralytics import YOLO


def load_model(path="best.pt"):
    return YOLO(path)


def detect_frame(model, frame, conf=0.1):
    return model.predict(source=frame, conf=conf, verbose=False)


def draw_results(results, max_w=900, max_h=700):
    annotated = results[0].plot()
    h, w = annotated.shape[:2]
    scale = min(max_w / w, max_h / h)
    return cv2.resize(annotated, (int(w * scale), int(h * scale)))


def main():
    model = load_model()
    cap = cv2.VideoCapture("video.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_frame(model, frame)
        display = draw_results(results)

        cv2.imshow("YOLO Detection", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
