import json
import cv2


def load_preds_from_item0_json(path: str):
    with open(path, "r") as f:
        item = json.load(f)
    return item["predictions"]["predictions"]


def draw_roboflow_boxes(image_path: str, preds, output_path: str = "boxed.jpg"):
    img = cv2.imread(image_path)

    for p in preds:
        x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
        x0, y0 = int(x - w / 2), int(y - h / 2)
        x1, y1 = int(x + w / 2), int(y + h / 2)

        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

    cv2.imwrite(output_path, img)
    return output_path


if __name__ == "__main__":
    preds = load_preds_from_item0_json("roboflow_item0.json")
    draw_roboflow_boxes("test-1.jpg", preds, "boxed.jpg")
