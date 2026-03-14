# scripts/collectdata.py
import os
import cv2

# =========================
# SETTINGS
# =========================
DATA_DIR = r"D:\signlanguagetranslation\data"
IMG_SIZE = 224
CLASSES = [chr(i) for i in range(65, 91)] + ["space"]

# =========================
# CREATE FOLDERS
# =========================
for cls in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

# =========================
# START CAMERA (Windows stable)
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not working!")
    exit()

print("Camera started")
print("Press A-Z to save letters")
print("Press SPACEBAR for space")
print("Press ESC to exit")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    # ROI
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    roi = frame[40:400, 0:300]
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

    cv2.imshow("Data Collection", frame)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(1) & 0xFF

    # Exit
    if key == 27:
        print("Exiting...")
        break

    # Debug print key
    if key != 255:
        print("Key pressed:", key)

    # A-Z
    if 97 <= key <= 122:  # lowercase letters
        letter = chr(key).upper()
        save_path = os.path.join(DATA_DIR, letter)

        count = len(os.listdir(save_path))
        filename = os.path.join(save_path, f"{count}.png")

        saved = cv2.imwrite(filename, roi)

        if saved:
            print(f"✅ Saved {letter} image {count}")
        else:
            print("❌ Failed to save image")

    # SPACE
    if key == 32:
        save_path = os.path.join(DATA_DIR, "space")
        count = len(os.listdir(save_path))
        filename = os.path.join(save_path, f"{count}.png")

        saved = cv2.imwrite(filename, roi)

        if saved:
            print(f"✅ Saved SPACE image {count}")
        else:
            print("❌ Failed to save image")

cap.release()
cv2.destroyAllWindows()