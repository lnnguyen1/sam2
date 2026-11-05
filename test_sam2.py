"""
Quick SAM2 sanity check (CPU, tiny model).
Run from inside the cloned sam2 folder:
    python test_sam2.py                      # uses the repo's sample truck image
    python test_sam2.py path/to/photo.jpg    # or your own image

A window opens: click ONE object. SAM2 shows its 3 candidate masks
and saves the best one as mask.png.
"""
import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

image_path = sys.argv[1] if len(sys.argv) > 1 else "notebooks/images/truck.jpg"
image = np.array(Image.open(image_path).convert("RGB"))

print("Loading model on CPU...")
model = build_sam2(CONFIG, CHECKPOINT, device="cpu")
predictor = SAM2ImagePredictor(model)

t0 = time.time()
with torch.inference_mode():
    predictor.set_image(image)
print(f"Image embedded in {time.time() - t0:.1f} s")

# Click one point on the object you want segmented
plt.figure(figsize=(10, 7))
plt.imshow(image)
plt.title("Click one object (window closes after the click)")
plt.axis("off")
clicks = plt.ginput(1, timeout=0)
plt.close()
if not clicks:
    sys.exit("No click registered.")

point = np.array([clicks[0]])
label = np.array([1])  # 1 = foreground point

with torch.inference_mode():
    masks, scores, _ = predictor.predict(
        point_coords=point,
        point_labels=label,
        multimask_output=True,  # 3 candidate masks (part / object / group)
    )

order = np.argsort(scores)[::-1]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, i in zip(axes, order):
    ax.imshow(image)
    ax.imshow(masks[i], alpha=0.5, cmap="jet")
    ax.scatter(point[:, 0], point[:, 1], c="lime", marker="*", s=300, edgecolors="white")
    ax.set_title(f"score {scores[i]:.3f}")
    ax.axis("off")
plt.tight_layout()
plt.show()

best = masks[order[0]].astype(np.uint8) * 255
Image.fromarray(best).save("mask.png")
print("Saved best mask to mask.png")