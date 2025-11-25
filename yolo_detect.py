# yolo_detect.py
from ultralytics import YOLO
import cv2
import os

# choose a model: yolov8n.pt is small (nano)
MODEL = "yolov8n.pt"

def yolo_infer(image_path, conf=0.25, save=True):
    model = YOLO(MODEL)
    results = model.predict(source=image_path, conf=conf, save=False)  # returns list
    # results[0].boxes gives bounding boxes
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    annotated = img.copy()
    boxes = []
    r = results[0]
    if r.boxes is not None:
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1,y1,x2,y2 = xyxy
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            boxes.append((x1,y1,x2-x1,y2-y1, conf, cls))
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(annotated, f"{conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    out_path = os.path.splitext(image_path)[0] + "_yolo_out.jpg"
    if save:
        cv2.imwrite(out_path, annotated)
    return out_path, boxes

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python yolo_detect.py PATH_TO_IMAGE")
        sys.exit(0)
    out, boxes = yolo_infer(sys.argv[1])
    print("Saved:", out)
    print("Boxes:", boxes)
