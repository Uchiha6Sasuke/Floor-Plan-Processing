import cv2
import numpy as np
from config import MIN_WALL_AREA, MIN_COMPONENT_AREA

def detect_wall_mask(binary):
    clean = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thick = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
    thick = cv2.dilate(thick, kernel, iterations=1)
    thick = cv2.morphologyEx(
        thick,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
        iterations=2
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thick)
    wall_mask = np.zeros_like(thick)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
            continue

        component = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(wall_mask, [contours[0]], -1, 255, -1)

    return cv2.erode(
        wall_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )


def extract_wall_contours(wall_mask):
    contours, _ = cv2.findContours(
        wall_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    walls = []
    wid = 1

    for c in contours:
        if cv2.contourArea(c) < MIN_WALL_AREA:
            continue

        approx = cv2.approxPolyDP(
            c,
            0.0005 * cv2.arcLength(c, True),
            True
        ).squeeze()

        if len(approx) < 4:
            continue

        walls.append({
            "id": f"w{wid}",
            "polygon": approx.tolist()
        })
        wid += 1

    return walls
