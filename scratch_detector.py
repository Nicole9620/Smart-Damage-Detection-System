# scratch_detector.py
import cv2
import numpy as np

def detect_scratches(image_path, min_length=30):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # High-pass filter for thin marks
    edges = cv2.Canny(blur, 20, 80)

    # Dilate to strengthen hairline scratches
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = img.copy()
    boxes = []

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if max(w,h) < min_length:  # ensure it's a long thin scratch
            continue

        aspect_ratio = max(w/h, h/w)
        if aspect_ratio < 2:      # keep only long/linear shapes
            continue

        boxes.append((x,y,w,h))
        cv2.rectangle(annotated, (x,y), (x+w, y+h), (0,255,255), 2)

    return annotated, boxes
