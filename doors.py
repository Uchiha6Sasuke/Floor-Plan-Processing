import os
from ultralytics import YOLO

def detect_doors(img, model_path):
    if not os.path.exists(model_path):
        return []

    model = YOLO(model_path)
    results = model(img, conf=0.4, iou=0.5, verbose=False)

    doors = []

    for r in results:
        if r.boxes is None:
            continue

        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            conf = float(b.conf[0])

            if conf < 0.6:
                continue

            doors.append({
                "id": f"d{len(doors)+1}",
                "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                "confidence": conf
            })

    return doors
