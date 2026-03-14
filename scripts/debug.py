# debug.py
from function import *
import cv2
import numpy as np
from keras.models import model_from_json

with open('model.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('model.h5')

# ── Check what training data looks like ───────────────────────────────────────
print("=== TRAINING DATA SAMPLE ===")
for action in ['A', 'B', 'C']:
    npy = np.load(f'MP_Data/{action}/0/0.npy')
    print(f"{action}: min={npy.min():.3f} max={npy.max():.3f} zeros={np.sum(npy==0)} shape={npy.shape}")
    # Predict with training data directly
    res = model.predict(np.expand_dims(npy, axis=0), verbose=0)[0]
    print(f"   Predicted: {actions[np.argmax(res)]} ({res.max():.2%})\n")

# ── Check webcam keypoints ─────────────────────────────────────────────────────
print("\n=== WEBCAM KEYPOINTS (press Q to quit) ===")
cap = cv2.VideoCapture(0)
with create_landmarker() as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, landmarker)
        draw_styled_landmarks(image, results)

        if results.hand_landmarks:
            kp = extract_keypoints(results)
            res = model.predict(np.expand_dims(kp, axis=0), verbose=0)[0]
            top3 = np.argsort(res)[::-1][:3]

            print(f"Keypoints: min={kp.min():.3f} max={kp.max():.3f} zeros={np.sum(kp==0)}")
            print(f"Top 3: {[(actions[i], f'{res[i]:.2%}') for i in top3]}")

            cv2.putText(image, f"Pred: {actions[top3[0]]} {res[top3[0]]:.0%}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            print("No hand detected!")
            cv2.putText(image, "NO HAND", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow('Debug', image)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()