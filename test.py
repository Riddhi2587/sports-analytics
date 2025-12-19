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
video_path = "lakers-trimmed.mp4"
output_path = "output_boxed.mp4"

workspace_name = "sportsapp"
workflow_id = "find-basketballs-and-basketball-hoops"

# Runtime sampling policy:
# process_fps = 5 means "send ~5 frames/sec to Roboflow"
process_fps = 5

save_debug_json = False          # set True to dump a few frame responses
debug_every_k_processed = 30     # dump JSON every N processed frames
# ----------------

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

src_fps = cap.get(cv2.CAP_PROP_FPS)
if not src_fps or src_fps != src_fps:  # NaN guard
    src_fps = 30.0

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Keep output FPS equal to source so the saved video has normal playback speed.
# (Boxes will update only on processed frames; unprocessed frames carry last result.)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))

# Decide how many frames to skip
stride = max(1, int(round(src_fps / process_fps)))

last_preds = []
frame_idx = 0
processed_idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Only run inference every "stride" frames
    if frame_idx % stride == 0:
        # Save frame to temp file (serverless client expects a file path here)
        tmp_path = "tmp_frame.jpg"
        cv2.imwrite(tmp_path, frame)

        result = client.run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            images={"image": tmp_path},
            use_cache=False,
        )

        # Your same extraction logic
        last_preds = result[0]["predictions"]["predictions"]
        processed_idx += 1

        if save_debug_json and (processed_idx % debug_every_k_processed == 0):
            with open(f"roboflow_response_frame_{frame_idx}.json", "w") as f:
                json.dump(result, f, indent=2)

    # Draw predictions on THIS frame (even if we didn't process it, reuse last_preds)
    boxed = frame.copy()
    for p in last_preds:
        # Roboflow bbox format is typically center x/y + width/height
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

    writer.write(boxed)

    frame_idx += 1

cap.release()
writer.release()

print(f"Wrote {output_path}")
print(f"Source FPS={src_fps:.2f}, processed FPS~={process_fps}, stride={stride}, processed frames={processed_idx}")

