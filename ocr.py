import cv2
import easyocr

def extract_dimensions(img):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(img, text_threshold=0.4)

    dims = []

    for bbox, text, conf in results:
        if conf < 0.42:
            continue

        if not any(c.isdigit() for c in text):
            continue

        dims.append({
            "text": text,
            "confidence": float(conf),
            "bbox": [[float(p[0]), float(p[1])] for p in bbox]
        })

    return dims
