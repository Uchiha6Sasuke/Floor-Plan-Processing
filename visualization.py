import cv2
import numpy as np

def visualize(img, walls, dims, doors, out_path):
    vis = img.copy()

    for w in walls:
        pts = np.array(w["polygon"], np.int32)
        cv2.polylines(vis, [pts], False, (0, 0, 255), 2)

    for d in dims:
        box = np.array(d["bbox"], dtype=np.int32)
        cv2.polylines(vis, [box], True, (255, 0, 0), 1)

    for door in doors:
        x, y, w, h = door["bbox"]
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite(out_path, vis)
