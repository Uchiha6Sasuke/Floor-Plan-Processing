import os
import json
import cv2
import torch

from config import INPUT_DIR, OUTPUT_DIR, VIS_DIR, DOOR_MODEL_PATH
from preprocessing import preprocess
from walls import detect_wall_mask, extract_wall_contours
from doors import detect_doors
from ocr import extract_dimensions
from visualization import visualize


def process_image(path):
    img = cv2.imread(path)
    name = os.path.basename(path)

    binary = preprocess(img)
    wall_mask = detect_wall_mask(binary)

    walls = extract_wall_contours(wall_mask)
    dims = extract_dimensions(img)
    doors = detect_doors(img, DOOR_MODEL_PATH)

    visualize(
        img,
        walls,
        dims,
        doors,
        os.path.join(VIS_DIR, name.replace(".", "_vis."))
    )

    return {
        "meta": {"source": name},
        "walls": walls,
        "dimensions": dims,
        "doors": doors
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    results = []

    for f in os.listdir(INPUT_DIR):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            results.append(process_image(os.path.join(INPUT_DIR, f)))

    with open(f"{OUTPUT_DIR}/result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("GPU" if torch.cuda.is_available() else "CPU")
    main()
