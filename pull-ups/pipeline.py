import os
import json
from dotenv import load_dotenv

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

load_dotenv()

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)

# ---- Static config ----
workspace_name = "sportsapp"
workflow_id = "find-faces-left-hands-right-hands-and-pull-up-bars"

process_fps = 4

save_debug_json = False
debug_every_k_processed = 30

HAND_CLASS_NAMES = {"left_hand", "right_hand"}
BAR_CLASS_NAMES = {"pull_up_bar"}
HEAD_CLASS_NAMES = {"head", "face"}

MIN_HAND_CONF = 0.20
MIN_BAR_CONF = 0.30
MIN_HEAD_CONF = 0.25

COOLDOWN_SECONDS = 1.5
HEAD_BAR_THRESHOLD_Y = 0.15
HAND_BAR_PROXIMITY = 0.12
# -----------------------


def xywh_to_xyxy(p):
    x, y, bw, bh = p["x"], p["y"], p["width"], p["height"]
    x1, y1 = x - bw / 2, y - bh / 2
    x2, y2 = x + bw / 2, y + bh / 2
    return x1, y1, x2, y2


def get_candidates(preds, class_names, min_conf):
    return [
        p
        for p in preds
        if p.get("class", "") in class_names and p.get("confidence", 0.0) >= min_conf
    ]


def find_gripped_bar(hands, bars):
    if len(hands) < 2 or not bars:
        return None

    hand_centroids = [(h["x"], h["y"]) for h in hands]

    best_bar = None
    best_close_hands = 0

    for bar in bars:
        bx1, by1, bx2, by2 = xywh_to_xyxy(bar)
        bar_cx = (bx1 + bx2) / 2
        bar_cy = (by1 + by2) / 2
        bar_w = bx2 - bx1

        close_hands = 0
        for hx, hy in hand_centroids:
            dist = np.hypot(hx - bar_cx, hy - bar_cy)
            if dist < HAND_BAR_PROXIMITY * bar_w:
                close_hands += 1

        if close_hands >= 2 and close_hands > best_close_hands:
            best_close_hands = close_hands
            best_bar = bar

    return best_bar


def head_above_bar(head_y, bar_y1, bar_y2, frame_height):
    bar_center_y = (bar_y1 + bar_y2) / 2
    head_threshold = bar_center_y - HEAD_BAR_THRESHOLD_Y * frame_height
    return head_y < head_threshold


def run_frame(frame_bgr):
    result = client.run_workflow(
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        images={"image": frame_bgr},
        use_cache=True,
    )
    return result[0]["predictions"]["predictions"]


def run(video_path: str, output_path: str, task: str) -> dict:
    """
    Entry point for main.py.

    Currently supports:
      - task == "count_reps": counts pull-ups based on head crossing bar.
    """
    if task != "count_reps":
        raise ValueError(f"Unsupported pullups task: {task}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps != src_fps:
        src_fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))

    stride = max(1, int(round(src_fps / process_fps)))

    pullups = 0
    prev_head_above = False
    last_score_frame = -10**9
    cooldown_frames = int(COOLDOWN_SECONDS * src_fps)

    last_preds = []
    frame_idx = 0
    processed_idx = 0

    last_gripped_bar_xyxy = None
    last_head = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        scored_this_frame = False

        if frame_idx % stride == 0:
            preds = run_frame(frame)
            last_preds = preds
            processed_idx += 1

            if save_debug_json and (processed_idx % debug_every_k_processed == 0):
                with open(
                    f"roboflow_response_frame_{frame_idx}.json", "w"
                ) as f:
                    json.dump(preds, f, indent=2)

            hands = get_candidates(last_preds, HAND_CLASS_NAMES, MIN_HAND_CONF)
            bars = get_candidates(last_preds, BAR_CLASS_NAMES, MIN_BAR_CONF)
            heads = get_candidates(last_preds, HEAD_CLASS_NAMES, MIN_HEAD_CONF)

            gripped_bar = find_gripped_bar(hands, bars)
            if gripped_bar is not None:
                last_gripped_bar_xyxy = xywh_to_xyxy(gripped_bar)
                last_head = (
                    max(heads, key=lambda p: p.get("confidence", 0.0))
                    if heads
                    else None
                )
            else:
                last_gripped_bar_xyxy = None
                last_head = None

            if (
                last_gripped_bar_xyxy is not None
                and last_head is not None
                and (frame_idx - last_score_frame) > cooldown_frames
            ):
                bx1, by1, bx2, by2 = last_gripped_bar_xyxy
                hx, hy = last_head["x"], last_head["y"]

                currently_above = head_above_bar(hy, by1, by2, h)

                if not prev_head_above and currently_above:
                    pullups += 1
                    last_score_frame = frame_idx
                    scored_this_frame = True

                prev_head_above = currently_above
            else:
                prev_head_above = False

        boxed = frame.copy()
        for p in last_preds:
            x, y = p["x"], p["y"]
            bw, bh = p["width"], p["height"]
            cls = p.get("class", "obj")
            conf = p.get("confidence", None)

            x1, y1 = int(x - bw / 2), int(y - bh / 2)
            x2, y2 = int(x + bw / 2), int(y + bh / 2)
            cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{cls}" if conf is None else f"{cls} {conf:.2f}"
            cv2.putText(
                boxed,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if last
