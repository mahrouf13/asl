# augment_weak.py
import cv2
import numpy as np
import os

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT  = _os.path.dirname(_HERE)          # project root (one level up)
_sys.path.insert(0, _os.path.join(ROOT, 'src'))


WEAK = ['M', 'T', 'S', 'A', 'N', 'I']
TARGET = 165  # match your strongest classes

for action in WEAK:
    folder = os.path.join(ROOT, 'data', action)
    images = sorted([f for f in os.listdir(folder)
                     if f.lower().endswith(('.png','.jpg','.jpeg'))])
    
    current = len(images)
    print(f"{action}: {current} images → augmenting to {TARGET}")
    idx = current

    while len(os.listdir(folder)) < TARGET:
        src_file = np.random.choice(images)
        src = cv2.imread(os.path.join(folder, src_file))
        if src is None:
            continue

        aug = src.copy()

        # Random brightness
        factor = np.random.uniform(0.7, 1.3)
        aug = np.clip(aug * factor, 0, 255).astype(np.uint8)

        # Random rotation ±10°
        h, w = aug.shape[:2]
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        aug = cv2.warpAffine(aug, M, (w, h))

        # Random slight zoom
        scale = np.random.uniform(0.9, 1.1)
        M2 = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
        aug = cv2.warpAffine(aug, M2, (w, h))

        # Random horizontal flip (only valid for symmetric signs)
        if action in ['M', 'N', 'S', 'T'] and np.random.rand() > 0.7:
            aug = cv2.flip(aug, 1)

        cv2.imwrite(os.path.join(folder, f'{idx}.png'), aug)
        idx += 1

    print(f"  → Done: {len(os.listdir(folder))} images")

print("\nAugmentation complete! Now re-run data.py and trainmodel.py")