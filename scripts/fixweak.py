# fix_weak.py
from function import *
import cv2
import numpy as np
import os

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT  = _os.path.dirname(_HERE)          # project root (one level up)
_sys.path.insert(0, _os.path.join(ROOT, 'src'))


WEAK_CLASSES = ['B', 'E', 'F', 'H', 'J', 'K','L','O','U','V','W','Y', 'Z']
TARGET = 165

def augment_image(src):
    """Generate a random augmentation of an image"""
    aug = src.copy()

    # Random brightness
    factor = np.random.uniform(0.75, 1.25)
    aug = np.clip(aug * factor, 0, 255).astype(np.uint8)

    # Random rotation ±12°
    h, w = aug.shape[:2]
    angle = np.random.uniform(-12, 12)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    aug = cv2.warpAffine(aug, M, (w, h))

    # Random zoom 90–110%
    scale = np.random.uniform(0.90, 1.10)
    M2 = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
    aug = cv2.warpAffine(aug, M2, (w, h))

    # Random slight blur
    if np.random.rand() > 0.7:
        ksize = np.random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

    # Random contrast
    alpha = np.random.uniform(0.85, 1.15)
    aug = np.clip(aug * alpha, 0, 255).astype(np.uint8)

    return aug

print("=" * 60)
print("  Step 1: Remove bad images (no hand detected)")
print("=" * 60)

with create_landmarker() as landmarker:
    for action in WEAK_CLASSES:
        folder = os.path.join(ROOT, 'data', action)
        all_images = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        bad_files  = []
        good_files = []

        for img_file in all_images:
            path  = os.path.join(folder, img_file)
            frame = cv2.imread(path)
            if frame is None:
                bad_files.append(img_file)
                continue
            _, results, _, is_flipped = mediapipe_detection(frame, landmarker)
            kp = extract_keypoints(results, is_flipped)
            if np.all(kp == 0):
                bad_files.append(img_file)
            else:
                good_files.append(img_file)

        # ── Delete bad images ──────────────────────────────────────
        print(f"\n[{action}]")
        print(f"  Good: {len(good_files)}  |  Bad: {len(bad_files)}")
        for fname in bad_files:
            os.remove(os.path.join(folder, fname))
            print(f"  ❌ Deleted: {fname}")

        # ── Augment to reach TARGET ────────────────────────────────
        print(f"\n  Augmenting from {len(good_files)} → {TARGET} images...")
        existing = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        idx = len(existing)

        while len(os.listdir(folder)) < TARGET:
            src_file  = np.random.choice(existing)
            src_frame = cv2.imread(os.path.join(folder, src_file))
            if src_frame is None:
                continue

            aug = augment_image(src_frame)

            # Verify augmented image still has a detectable hand
            _, aug_results, _, aug_flipped = mediapipe_detection(aug, landmarker)
            aug_kp = extract_keypoints(aug_results, aug_flipped)

            if not np.all(aug_kp == 0):
                out_path = os.path.join(folder, f'aug_{idx}.png')
                cv2.imwrite(out_path, aug)
                idx += 1

        final_count = len(os.listdir(folder))
        print(f"  ✅ {action}: now has {final_count} valid images")

print("\n" + "=" * 60)
print("  Step 2: Verify final counts")
print("=" * 60)

with create_landmarker() as landmarker:
    total_good = 0
    total_bad  = 0
    for action in WEAK_CLASSES:
        folder = os.path.join(ROOT, 'data', action)
        all_images = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        good = 0
        bad  = 0
        for img_file in all_images:
            frame = cv2.imread(os.path.join(folder, img_file))
            if frame is None:
                bad += 1
                continue
            _, results, _, is_flipped = mediapipe_detection(frame, landmarker)
            kp = extract_keypoints(results, is_flipped)
            if np.all(kp == 0):
                bad += 1
            else:
                good += 1
        total_good += good
        total_bad  += bad
        rate = good / (good + bad) * 100 if (good + bad) > 0 else 0
        flag = '✅' if rate >= 95 else '⚠️ ' if rate >= 85 else '❌'
        print(f"  {flag} {action}: {good}/{good+bad} ({rate:.1f}%)")

    print(f"\n  Overall: {total_good}/{total_good+total_bad} "
          f"({total_good/(total_good+total_bad)*100:.1f}%)")

print("\n" + "=" * 60)
print("  Done! Now run:")
print("    python data.py")
print("    python trainmodel.py")
print("=" * 60)