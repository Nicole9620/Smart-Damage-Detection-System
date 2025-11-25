# damage_detector.py — FINAL bulletproof dent detector
import cv2
import numpy as np
import math

def preprocess(img, scale=1400):
    h, w = img.shape[:2]
    if max(h, w) > scale:
        if h > w:
            new_h = scale
            new_w = int(w * scale / h)
        else:
            new_w = scale
            new_h = int(h * scale / w)
        img = cv2.resize(img, (new_w, new_h))
    return img

def detect_damage_opencv(image_path, min_area=50, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    img = preprocess(img)
    h, w = img.shape[:2]
    orig = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Strong CLAHE
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    g = clahe.apply(gray)

    # REALLY STRONG BLACKHAT (captures dents reliably)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (101,101))
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)

    # NORMALIZE
    bh = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    bh = cv2.GaussianBlur(bh, (7,7), 0)

    # ⭐ NO ADAPTIVE THRESHOLD (it was deleting the dent)
    # Simple + Otsu threshold (never wipes shallow dents)
    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Strong closing
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (41,41))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_k)

    # Slight open
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_k)

    # FIND CONTOURS
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w1, h1 = cv2.boundingRect(c)
        boxes.append((x,y,w1,h1))

    # fallback: pick largest blob if nothing detected
    if len(boxes) == 0 and len(cnts) > 0:
        biggest = max(cnts, key=cv2.contourArea)
        x,y,w1,h1 = cv2.boundingRect(biggest)
        boxes = [(x,y,w1,h1)]

    # DRAW
    for (x,y,w1,h1) in boxes:
        cv2.rectangle(orig, (x,y), (x+w1,y+h1), (0,255,0), 3)
        cv2.putText(orig, "dent", (x, max(10,y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    debug_imgs = {"blackhat": blackhat, "thresh": th, "closed": closed} if debug else {}

    return orig, boxes, debug_imgs
