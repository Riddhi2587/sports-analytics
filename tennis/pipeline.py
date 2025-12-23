import json
import cv2
import numpy as np
from collections import defaultdict, deque
from inference_sdk import InferenceHTTPClient

# ---- Roboflow client ----
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="atL3rkIRqnsfe1BXXGHf"
)

# ---- Config ----
video_path = "tennis-cropped.mp4"
output_path = "tennis_tracked_output.mp4"

workspace_name = "sportsapp"
workflow_id = "find-players-and-balls"

process_fps = 12
POS_HISTORY_LEN = 30

# IMPORTANT: class names from your JSON
PLAYER_CLASS_NAMES = {"players"}   # <-- from debug_frame_0.json
BALL_CLASS_NAMES = {"tennis-ball"}

MIN_PLAYER_CONF = 0.6
MIN_BALL_CONF = 0.01

# ---------- helpers ----------

def parse_workflow_preds(result):
    
    if not isinstance(result, list) or not result:
        return []

    root = result[0]
    pred_root = root.get("predictions", {})
    preds = pred_root.get("predictions", [])  # <--- inner list

    out = []
    for p in preds:
        if not isinstance(p, dict):
            continue
        out.append({
            "x": float(p["x"]),
            "y": float(p["y"]),
            "width": float(p["width"]),
            "height": float(p["height"]),
            "confidence": float(p.get("confidence", 0)),
            "class": str(p.get("class", "")),
            # no tracker_id yet; this is SAM output, not ByteTrack
            "tracker_id": p.get("tracker_id", None),
        })
    return out

def get_candidates(preds, class_names, min_conf):
    out = []
    for p in preds:
        cls = str(p.get("class", "")).lower()
        if cls in class_names and p.get("confidence", 0.0) >= min_conf:
            out.append(p)
    return out

# ---------- video setup ----------

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))

stride = max(1, int(round(src_fps / process_fps)))
print(f"Processing {video_path} at ~{process_fps} FPS (stride={stride})")

# ---------- simple tracking state (no tracker_id) ----------

# Just store recent centers for N strongest detections per frame
player_tracks = defaultdict(lambda: deque(maxlen=POS_HISTORY_LEN))
next_track_id = 0
last_frame_centers = []  # [(x,y,track_id)]

# nearest-neighbor threshold (pixels) to link detections across frames
ASSIGN_THRESH = 80.0

frame_idx = 0
processed_idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if frame_idx % stride == 0:
        # send frame to workflow
        tmp_path = "tmp_frame.jpg"
        cv2.imwrite(tmp_path, frame)

        result = client.run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            images={"image": tmp_path},
            use_cache=True,
        )

        if processed_idx == 0:
            print("TOP-LEVEL KEYS:", result[0].keys())
            print("PREDICTIONS TYPE:", type(result[0].get("predictions")))
            print(json.dumps(result[0].get("predictions"), indent=2)[:500])


        preds = parse_workflow_preds(result)
        if processed_idx < 5:
            print("CLASSES IN FRAME:", {p["class"] for p in preds})

        players = get_candidates(preds, PLAYER_CLASS_NAMES, MIN_PLAYER_CONF)
        balls = get_candidates(preds, BALL_CLASS_NAMES, MIN_BALL_CONF)

        # sort players by confidence, keep top K if desired
        players = sorted(players, key=lambda p: -p["confidence"])

        # --------- simple data association for players (no ByteTrack) ---------
        curr_centers = []
        used_prev = set()

        for p in players:
            cx, cy = p["x"], p["y"]

            # find nearest existing track from last_frame_centers
            best_id = None
            best_d = ASSIGN_THRESH
            for (px, py, tid) in last_frame_centers:
                if tid in used_prev:
                    continue
                d = np.hypot(cx - px, cy - py)
                if d < best_d:
                    best_d = d
                    best_id = tid

            if best_id is None:
                # start new track
                tid = next_track_id
                next_track_id += 1
            else:
                tid = best_id
                used_prev.add(best_id)

            player_tracks[tid].append((cx, cy))
            curr_centers.append((cx, cy, tid))

        last_frame_centers = curr_centers
        processed_idx += 1

        print(f"Processed frame {frame_idx}: {len(players)} players, {len(balls)} balls")

        # optional: you can do similar nearest-neighbor for balls here

        # ---------- draw output ----------
    boxed = frame.copy()

    # draw player tracks (blue)
    for tid, pts in player_tracks.items():
        pts_list = list(pts)
        if len(pts_list) > 1:
            for i in range(1, len(pts_list)):
                p1 = (int(pts_list[i-1][0]), int(pts_list[i-1][1]))
                p2 = (int(pts_list[i][0]), int(pts_list[i][1]))
                cv2.line(boxed, p1, p2, (255, 0, 0), 2)
            # also draw last point with ID
            x_last, y_last = pts_list[-1]
            cv2.circle(boxed, (int(x_last), int(y_last)), 4, (255, 0, 0), -1)
            cv2.putText(
                boxed, f"P{tid}",
                (int(x_last)+5, int(y_last)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
            )

    # draw raw detections for visual sanity
    for p in preds if 'preds' in locals() else []:
        x, y = p["x"], p["y"]
        bw, bh = p["width"], p["height"]
        x1, y1 = int(x - bw/2), int(y - bh/2)
        x2, y2 = int(x + bw/2), int(y + bh/2)
        cls = p["class"]
        conf = p["confidence"]
        color = (0, 255, 0) if cls in PLAYER_CLASS_NAMES else (0, 255, 255)
        cv2.rectangle(boxed, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            boxed, f"{cls} {conf:.2f}",
            (x1, max(0, y1-6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    cv2.putText(
        boxed,
        f"tracks: {len(player_tracks)}  frame: {frame_idx}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    writer.write(boxed)
    frame_idx += 1

cap.release()
writer.release()
print(f"Done. Wrote {output_path}, tracks={len(player_tracks)}")
