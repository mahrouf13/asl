# fix_final.py
from function import *
import cv2, numpy as np, os

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT  = _os.path.dirname(_HERE)          # project root (one level up)
_sys.path.insert(0, _os.path.join(ROOT, 'src'))


# These are the only remaining problem pairs
CONFUSED_PAIRS = {
    'A': ['T', 'K', 'W'],   # A confused with T most (5 times!)
    'I': ['S', 'A', 'L'],
    'H': ['I'],
    'J': ['K', 'I'],
    'E': ['O', 'V', 'S'],
}

TARGET = 200  # boost these classes higher

def augment(src):
    aug = src.copy()
    aug = np.clip(aug * np.random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
    h, w = aug.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), np.random.uniform(-15, 15), 1)
    aug = cv2.warpAffine(aug, M, (w, h))
    scale = np.random.uniform(0.85, 1.15)
    M2 = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
    aug = cv2.warpAffine(aug, M2, (w, h))
    if np.random.rand() > 0.6:
        aug = cv2.GaussianBlur(aug, (3,3), 0)
    # Random horizontal shift
    tx = np.random.randint(-20, 20)
    ty = np.random.randint(-20, 20)
    M3 = np.float32([[1,0,tx],[0,1,ty]])
    aug = cv2.warpAffine(aug, M3, (w, h))
    return aug

print("Augmenting weak classes to", TARGET, "images...\n")

with create_landmarker() as landmarker:
    for action in CONFUSED_PAIRS.keys():
        folder = os.path.join(ROOT, 'data', action)
        existing = sorted([f for f in os.listdir(folder)
                           if f.lower().endswith(('.png','.jpg','.jpeg'))])
        print(f"[{action}] {len(existing)} → {TARGET}")
        idx = len(existing)
        attempts = 0

        while len([f for f in os.listdir(folder)
                   if f.lower().endswith(('.png','.jpg','.jpeg'))]) < TARGET:
            attempts += 1
            if attempts > TARGET * 10:
                print(f"  ⚠ Could not reach {TARGET} for {action}")
                break

            src_file  = np.random.choice(existing)
            src_frame = cv2.imread(os.path.join(folder, src_file))
            if src_frame is None:
                continue

            aug = augment(src_frame)
            _, res, _, flipped = mediapipe_detection(aug, landmarker)
            kp = extract_keypoints(res, flipped)

            if not np.all(kp == 0):
                cv2.imwrite(os.path.join(folder, f'aug_{idx}.png'), aug)
                idx += 1

        final = len([f for f in os.listdir(folder)
                     if f.lower().endswith(('.png','.jpg','.jpeg'))])
        print(f"  ✅ {action}: {final} images\n")

print("Done! Run: python data.py → python trainmodel.py")