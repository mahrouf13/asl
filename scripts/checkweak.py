# check_weak.py
# Run this to see exactly which images are failing for weak classes

from function import *
import cv2
import numpy as np
import os

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT  = _os.path.dirname(_HERE)          # project root (one level up)
_sys.path.insert(0, _os.path.join(ROOT, 'src'))


WEAK_CLASSES = ['M', 'T', 'S', 'A', 'N', 'I']

print("Checking weak classes...\n")

with create_landmarker() as landmarker:
    for action in WEAK_CLASSES:
        folder = os.path.join(ROOT, 'data', action)
        all_images = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        detected = 0
        failed   = []

        for img_file in all_images:
            frame = cv2.imread(os.path.join(folder, img_file))
            if frame is None:
                failed.append(img_file)
                continue
            _, results, _, is_flipped = mediapipe_detection(frame, landmarker)
            kp = extract_keypoints(results, is_flipped)
            if np.all(kp == 0):
                failed.append(img_file)
            else:
                detected += 1

        rate = detected / len(all_images) * 100
        print(f"  {action}: {detected}/{len(all_images)} detected ({rate:.0f}%)")
        if failed:
            print(f"     Failed images: {failed[:5]}{'...' if len(failed)>5 else ''}")
# ```

# ---

# **What to do next based on results:**

# **Step 1 — Replace bad images for weak classes.** For M, T, S, A, N, I — add 30–50 more varied images (different lighting, angles, distances).

# **Step 2 — The confusion pairs need attention:**
# ```
# M ↔ N  (very similar ASL handshapes — need clearer photos)
# A ↔ S  (fist vs fist with thumb — subtle difference)
# I ↔ E  (pinky extended vs fingers curled)