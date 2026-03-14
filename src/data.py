# src/data.py  -- LETTER keypoint extraction (no changes needed from your version)
# extract_keypoints() now returns 126 features (right+left hand)
# Letter model will be retrained with input_shape=(126,)
from function import *
import os
import cv2
import numpy as np

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT  = _os.path.dirname(_HERE)
_sys.path.insert(0, _os.path.join(ROOT, 'src'))

total_detected = 0
total_skipped  = 0
strategy_hits  = {name: 0 for name in STRATEGY_NAMES}

print("=" * 62)
print("  Hand Keypoint Extraction  --  LETTERS")
print("  Output: 126 features (right 63 + left 63)")
print("=" * 62)
print(f"Actions        : {list(actions)}")
print(f"Output path    : {os.path.abspath(DATA_PATH)}")
print(f"Sequence length: {sequence_length} frames per image")
print(f"Strategies     : {len(STRATEGY_NAMES)} fallbacks tried per image")
print()

with create_landmarker() as landmarker:
    for action in actions:

        folder = os.path.join(ROOT, 'data', action)
        if not os.path.exists(folder):
            print(f"[SKIP] Folder missing: {folder}")
            continue

        all_images = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if not all_images:
            print(f"[SKIP] No images in: {folder}")
            continue

        print(f"[{action}] {len(all_images)} images")
        action_detected = 0
        action_skipped  = 0

        for sequence, img_file in enumerate(all_images):

            img_path = os.path.join(folder, img_file)
            frame    = cv2.imread(img_path)

            seq_dir = os.path.join(DATA_PATH, action, str(sequence))
            os.makedirs(seq_dir, exist_ok=True)

            if frame is None:
                print(f"    [SKIP] Cannot read: {img_path}")
                # 126 zeros -- both hands absent
                keypoints = np.zeros(TWO_HAND_KP)
                action_skipped += 1
                for frame_num in range(sequence_length):
                    np.save(os.path.join(seq_dir, str(frame_num)), keypoints)
                continue

            image, results, strategy_idx, is_flipped = mediapipe_detection(
                frame, landmarker)

            # extract_keypoints returns 126 features
            keypoints  = extract_keypoints(results, is_flipped)
            hand_found = not np.all(keypoints == 0)

            if hand_found:
                action_detected += 1
                strategy_hits[STRATEGY_NAMES[strategy_idx]] += 1
            else:
                action_skipped += 1

            draw_styled_landmarks(image, results, is_flipped)

            label = "DETECTED"  if hand_found else "NO HAND"
            color = (0, 200, 0) if hand_found else (0, 0, 255)
            strat = STRATEGY_NAMES[strategy_idx] if strategy_idx >= 0 else "none"

            cv2.putText(image,
                        f"{action} | {sequence+1}/{len(all_images)} | {label} [{strat}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(image,
                        f"OK: {action_detected}  Miss: {action_skipped}",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(image,
                        img_file,
                        (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

            cv2.imshow('Keypoint Extraction', image)

            # Save same keypoint 30 times
            # Static sign = 30 identical frames = near-zero velocity
            for frame_num in range(sequence_length):
                np.save(os.path.join(seq_dir, str(frame_num)), keypoints)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuit by user.")
                cv2.destroyAllWindows()
                exit()

        rate = action_detected / len(all_images) * 100 if all_images else 0
        print(f"    OK {action_detected:3d}  MISS {action_skipped:3d}  ({rate:.0f}%)")
        total_detected += action_detected
        total_skipped  += action_skipped

cv2.destroyAllWindows()

total = total_detected + total_skipped
rate  = total_detected / total * 100 if total else 0

print()
print("=" * 62)
print("  Results")
print("=" * 62)
print(f"  Total   : {total}")
print(f"  Hit     : {total_detected}  ({rate:.1f}%)")
print(f"  Miss    : {total_skipped}  ({100-rate:.1f}%)")
print()
print("  Strategy breakdown:")
for name, count in strategy_hits.items():
    print(f"    {name:<20} {count:4d}")
print()
if rate < 70:
    print("  WARNING: Below 70% detection. Retake photos with better lighting.")
elif rate < 90:
    print("  Acceptable. Replace low-quality photos to improve further.")
else:
    print("  Good detection rate!")
print()
print("  Next step: python src/trainmodel.py")