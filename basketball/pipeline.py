import os
import json
from dotenv import load_dotenv

import cv2
from inference_sdk import InferenceHTTPClient

load_dotenv()

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)

# ---- Config ----
video_path = "lakers-vid-17s.mp4"
output_path = "output_vid-17s.mp4"

workspace_name = "sportsapp"
workflow_id = "find-basketballs-and-basketball-hoops-2"

process_fps = 10

save_debug_json = False
debug_every_k_processed = 30

HOOP_CLASS_NAMES = {"basketball hoop"}
BALL_CLASS_NAMES = {"basketball"}

MIN_BALL_CONF = 0.10
MIN_HOOP_CONF = 0.20

ABOVE_X_PAD_FRAC = 0.15
ABOVE_MAX_H_FRAC = 0.75
BELOW_MAX_H_FRAC = 1.00

COOLDOWN_SECONDS = 1.0
# ----------------

TEAM_LEFT = "team_left"    # you can rename later
TEAM_RIGHT = "team_right"

def xywh_to_xyxy(p):
    x, y, bw, bh = p["x"], p["y"], p["width"], p["height"]
    x1, y1 = x - bw / 2, y - bh / 2
    x2, y2 = x + bw / 2, y + bh / 2
    return x1, y1, x2, y2

def is_ball_above(ball_p, hoop_xyxy):
    bx, by = ball_p["x"], ball_p["y"]
    hx1, hy1, hx2, hy2 = hoop_xyxy
    hoop_h = hy2 - hy1
    pad = (hx2 - hx1) * ABOVE_X_PAD_FRAC

    inside_x = (hx1 - pad) <= bx <= (hx2 + pad)
    above_top = by < hy1
    dy = hy1 - by
    within = dy <= (ABOVE_MAX_H_FRAC * hoop_h)
    return inside_x and above_top and within

def is_ball_below(ball_p, hoop_xyxy):
    bx, by = ball_p["x"], ball_p["y"]
    hx1, hy1, hx2, hy2 = hoop_xyxy
    hoop_h = hy2 - hy1

    inside_x = hx1 <= bx <= hx2
    below_bottom = by > hy2
    dy = by - hy2
    within = dy <= (BELOW_MAX_H_FRAC * hoop_h)
    return inside_x and below_bottom and within

def pick_best(preds, class_names, min_conf):
    cands = [p for p in preds if p.get("class","") in class_names and p.get("confidence",0.0) >= min_conf]
    return max(cands, key=lambda p: p.get("confidence", 0.0)) if cands else None

def get_candidates(preds, class_names, min_conf):
    return [p for p in preds if p.get("class","") in class_names and p.get("confidence",0.0) >= min_conf]

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

# ---- State ----
team_scores = {TEAM_LEFT: 0, TEAM_RIGHT: 0}

prev_any_above = False
last_score_frame = -10**9
cooldown_frames = int(COOLDOWN_SECONDS * src_fps)

last_preds = []
frame_idx = 0
processed_idx = 0

last_hoop_xyxy = None
ball_side = {}       # tracker_id -> "left"/"right" relative to hoop center
scored_balls = set() # tracker_ids that have already scored

while True:
    ok, frame = cap.read()
    if not ok:
        break

    scored_this_frame = False

    if frame_idx % stride == 0:
        tmp_path = "tmp_frame.jpg"
        cv2.imwrite(tmp_path, frame)

        result = client.run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            images={"image": tmp_path},
            use_cache=False,
        )

        wf_outputs = result[0]["predictions"]
        last_preds = wf_outputs["predictions"]   # ByteTrack output

        processed_idx += 1

        if save_debug_json and (processed_idx % debug_every_k_processed == 0):
            with open(f"roboflow_response_frame_{frame_idx}.json", "w") as f:
                json.dump(result, f, indent=2)

        # --- Hoop + all ball candidates ---
        hoop = pick_best(last_preds, HOOP_CLASS_NAMES, MIN_HOOP_CONF)
        balls = get_candidates(last_preds, BALL_CLASS_NAMES, MIN_BALL_CONF)

        if hoop is not None:
            last_hoop_xyxy = xywh_to_xyxy(hoop)

        if last_hoop_xyxy is not None and balls:
            hx1, hy1, hx2, hy2 = last_hoop_xyxy
            hoop_cx = (hx1 + hx2) / 2.0

            any_above = False
            any_below = False
            scoring_ball = None

            for b in balls:
                tid = b.get("tracker_id")
                if tid is None:
                    continue

                # 1) remember which side this ball came from
                if tid not in ball_side:
                    bx = b["x"]
                    side = "left" if bx < hoop_cx else "right"
                    ball_side[tid] = side

                # 2) check above / below flags
                if is_ball_above(b, last_hoop_xyxy):
                    any_above = True
                if is_ball_below(b, last_hoop_xyxy):
                    any_below = True
                    scoring_ball = b   # last ball seen below this frame

            # Count on above -> below, with cooldown and per-ball uniqueness
            if prev_any_above and any_below and (frame_idx - last_score_frame) > cooldown_frames and scoring_ball is not None:
                tid = scoring_ball.get("tracker_id")
                if tid is not None and tid not in scored_balls:
                    side = ball_side.get(tid)
                    if side == "left":
                        team_scores[TEAM_LEFT] += 1
                    elif side == "right":
                        team_scores[TEAM_RIGHT] += 1
                    scored_balls.add(tid)

                    last_score_frame = frame_idx
                    scored_this_frame = True
                prev_any_above = False
            else:
                prev_any_above = any_above or prev_any_above
        else:
            prev_any_above = False

    # ---- Drawing ----
    boxed = frame.copy()
    for p in last_preds:
        x, y = p["x"], p["y"]
        bw, bh = p["width"], p["height"]
        cls = p.get("class", "obj")
        conf = p.get("confidence", None)
        tracker_id = p.get("tracker_id")

        x1, y1 = int(x - bw / 2), int(y - bh / 2)
        x2, y2 = int(x + bw / 2), int(y + bh / 2)

        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if tracker_id is not None:
            if conf is None:
                label = f"{cls} id={tracker_id}"
            else:
                label = f"{cls} {conf:.2f} id={tracker_id}"
        else:
            label = f"{cls}" if conf is None else f"{cls} {conf:.2f}"

        cv2.putText(
            boxed, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
        )

    if last_hoop_xyxy is not None:
        hx1, hy1, hx2, hy2 = last_hoop_xyxy
        cv2.rectangle(boxed, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (0, 255, 255), 2)

    # Team scores overlay
    cv2.putText(
        boxed,
        f"{TEAM_LEFT}: {team_scores[TEAM_LEFT]}  {TEAM_RIGHT}: {team_scores[TEAM_RIGHT]}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255) if scored_this_frame else (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    writer.write(boxed)
    frame_idx += 1

cap.release()
writer.release()

score_path = "latest_score.json"
with open(score_path, "w") as f:
    json.dump(
        {
            "video_path": video_path,
            "output_path": output_path,
            "team_scores": team_scores,
            "processed_fps": process_fps,
            "source_fps": src_fps,
            "processed_frames": processed_idx,
        },
        f,
        indent=2,
    )

print(f"Wrote {output_path}")
print(f"Team scores = {team_scores}")
print(f"Wrote score to {score_path}")
print(f"Source FPS={src_fps:.2f}, processed FPS~={process_fps}, stride={stride}, processed frames={processed_idx}")
