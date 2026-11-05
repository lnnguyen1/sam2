"""
SAM2 video test (CPU, tiny model).
Run from inside the sam2 folder:
    python test_sam2_video.py path/to/video.mp4

What it does:
  1. Pulls frames out of the video (downsampled so CPU can handle it)
  2. Shows the first frame: click once on EACH object you want tracked,
     then press Enter. (Right-click undoes the last click.)
  3. Tracks those objects through the whole video
  4. Saves colored overlay frames to  output/<video name>/overlays/
     and the raw masks to            output/<video name>/masks/
     and your clicks to              output/<video name>/prompts.json
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2_video_predictor

CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

TARGET_FPS = 5      # frames kept per second of video
MAX_WIDTH = 640     # frames are shrunk to this width (keeps CPU time sane)

if len(sys.argv) < 2:
    sys.exit("Usage: python test_sam2_video.py path/to/video.mp4")
video_path = sys.argv[1]
name = os.path.splitext(os.path.basename(video_path))[0]
out_dir = os.path.join("output", name)
frames_dir = os.path.join(out_dir, "frames")
overlay_dir = os.path.join(out_dir, "overlays")
mask_dir = os.path.join(out_dir, "masks")
for d in (frames_dir, overlay_dir, mask_dir):
    os.makedirs(d, exist_ok=True)

# ---------- 1. extract frames ----------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"Could not open {video_path}")
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
step = max(1, round(src_fps / TARGET_FPS))
i = kept = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i % step == 0:
        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)))
        cv2.imwrite(os.path.join(frames_dir, f"{kept:05d}.jpg"), frame)
        kept += 1
    i += 1
cap.release()
print(f"Extracted {kept} frames (every {step}th frame of {i})")

# ---------- 2. load model + video ----------
print("Loading model on CPU...")
predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device="cpu")
with torch.inference_mode():
    state = predictor.init_state(video_path=frames_dir)

# ---------- 3. click objects on first frame ----------
first = cv2.cvtColor(cv2.imread(os.path.join(frames_dir, "00000.jpg")), cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 7))
plt.imshow(first)
plt.title("Click once per object, then press Enter (right-click = undo)")
plt.axis("off")
clicks = plt.ginput(n=-1, timeout=0)
plt.close()
if not clicks:
    sys.exit("No clicks registered.")
print(f"{len(clicks)} objects selected")

prompts = []
with torch.inference_mode():
    for obj_id, (x, y) in enumerate(clicks, start=1):
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=obj_id,
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
        prompts.append({"obj_id": obj_id, "frame": 0, "x": float(x), "y": float(y)})
with open(os.path.join(out_dir, "prompts.json"), "w") as f:
    json.dump(prompts, f, indent=2)

# ---------- 4. propagate + save ----------
cmap = plt.get_cmap("tab20")
t0 = time.time()
with torch.inference_mode():
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        masks = (mask_logits > 0.0).cpu().numpy()[:, 0]  # (num_objects, H, W)
        np.savez_compressed(os.path.join(mask_dir, f"{frame_idx:05d}.npz"),
                            obj_ids=np.array(obj_ids), masks=masks)

        img = cv2.imread(os.path.join(frames_dir, f"{frame_idx:05d}.jpg")).astype(np.float32)
        for k, oid in enumerate(obj_ids):
            color = np.array(cmap((oid - 1) % 20)[:3])[::-1] * 255  # RGB -> BGR
            m = masks[k]
            img[m] = 0.5 * img[m] + 0.5 * color
        cv2.imwrite(os.path.join(overlay_dir, f"{frame_idx:05d}.jpg"), img.astype(np.uint8))
        print(f"  frame {frame_idx + 1}/{kept}  ({time.time() - t0:.0f} s elapsed)", end="\r")

print(f"\nDone in {time.time() - t0:.0f} s. Overlays are in {overlay_dir}")