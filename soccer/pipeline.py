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
video_path = "goals-1.mp4"
output_path = "output-scored-goals-1.mp4"

workspace_name = "sportsapp"
workflow_id = "find-footballs-and-goals"

process_fps = 10

save_debug_json = False
debug_every_k_processed = 30

BALL_CLASS_NAMES = {"ball"}
GOAL_CLASS_NAMES = {"goal"}

MIN_BALL_CONF = 0.10
MIN_GOAL_CONF = 0.20

COOLDOWN_SECONDS = 1.0  # global cooldown between counted goals
# ----------------


def xywh_to_xyxy(p):
    x, y, bw, bh = p["x"], p["y"], p["width"], p["height"]
    x1, y1 = x - bw / 2, y - bh / 2
    x2, y2 = x + bw / 2, y + bh / 2
    return x1, y1, x2, y2


def pick_best(preds, class_names, min_conf):
    cands = [p for p in preds if p.get("class", "") in class_names and p.get("confidence", 0.0) >= min_conf]
    return max(cands, key=lambda p: p.get("confidence", 0.0)) if cands else None


def get_candidates(preds, class_names, min_conf):
    return [p for p in preds if p.get("class", "") in class_names and p.get("confidence", 0.0) >= min_conf]


def ball_in_goal(ball_p, goal_xyxy):
    # simple test: ball center inside goal bounding box
    bx, by = ball_p["x"], ball_p["y"]
    gx1, gy1, gx2, gy2 = goal_xyxy
    return (gx1 <= bx <= gx2) and (gy1 <= by <= gy2)


def run_frame(frame_bgr):
    # IMPORTANT: pass np.array directly (no bytes)
    result = client.run_workflow(
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        images={"image": frame_bgr},
        use_cache=True,
    )
    # adjust if your Output node uses a different structure
    return result[0]["predictions"]["predictions"]


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

# goal count state
score = 0
prev_any_in_goal = False
last_score_frame = -10**9
cooldown_frames = int(COOLDOWN_SECONDS * src_fps)

last_preds = []
frame_idx = 0
processed_idx = 0

last_goal_xyxy = None

while True:
    ok, frame = cap.read()
    if not ok:
        break

    scored_this_frame = False

    if frame_idx % stride == 0:
        print("Calling workflow on frame", frame_idx)
        preds = run_frame(frame)
        print("Got response on frame", frame_idx, "len:", len(preds))
        last_preds = preds
        processed_idx += 1

        if save_debug_json and (processed_idx % debug_every_k_processed == 0):
            with open(f"roboflow_response_frame_{frame_idx}.json", "w") as f:
                json.dump(preds, f, indent=2)

        # pick one goal box (best confidence)
        goal = pick_best(last_preds, GOAL_CLASS_NAMES, MIN_GOAL_CONF)
        if goal is not None:
            last_goal_xyxy = xywh_to_xyxy(goal)

        balls = get_candidates(last_preds, BALL_CLASS_NAMES, MIN_BALL_CONF)

        if last_goal_xyxy is not None and balls:
            any_in_goal = any(ball_in_goal(b, last_goal_xyxy) for b in balls)

            # count on "outside -> inside goal box" with cooldown
            if prev_any_in_goal is False and any_in_goal and (frame_idx - last_score_frame) > cooldown_frames:
                score += 1
                last_score_frame = frame_idx
                scored_this_frame = True

            # latch state until we see ball leave goal again
            prev_any_in_goal = any_in_goal
        else:
            prev_any_in_goal = False

    # draw predictions
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

    # draw goal box if available
    if last_goal_xyxy is not None:
        gx1, gy1, gx2, gy2 = last_goal_xyxy
        cv2.rectangle(boxed, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 255), 2)

    # overlay score
    cv2.putText(
        boxed,
        f"Goals: {score}",
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

score_path = "football_latest_score.json"
with open(score_path, "w") as f:
    json.dump(
        {
            "video_path": video_path,
            "output_path": output_path,
            "final_goals": score,
            "processed_fps": process_fps,
            "source_fps": src_fps,
            "processed_frames": processed_idx,
        },
        f,
        indent=2,
    )

print(f"Wrote {output_path}")
print(f"Final goals = {score}")
print(f"Source FPS={src_fps:.2f}, processed FPS~={process_fps}, stride={stride}, processed frames={processed_idx}")
