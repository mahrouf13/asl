# src/function.py  -- MediaPipe 0.10+ compatible
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# PATHS
# =============================================================================
DATA_PATH      = os.path.join(ROOT, 'MP_Data')
WORD_DATA_PATH = os.path.join(ROOT, 'MP_Data_Words')

# =============================================================================
# HAND CONNECTIONS  (hardcoded -- MediaPipe 0.10 removed solutions module)
# =============================================================================
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# =============================================================================
# ACTIONS
# =============================================================================
actions = np.array([
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','space'
])

LETTER_SIGNS = actions.copy()

WORD_SIGNS = np.array([
    'hello','goodbye','thankyou','sorry','please',
    'yes','no','help','stop','more','want','again',
    'me','you','mother','father','friend',
    'eat','drink','love','like','go','come',
    'good','bad',
    'what','where','who','why','how',
    'water','food','home','work','school','doctor',
    'pain',
    'happy','sad','angry','tired','hungry','hot','cold',
    'name','today','tomorrow','morning','night',
])

# =============================================================================
# FEATURE SIZES
#   Letter model : 63  (single dominant hand)
#   Word model   : 126 (right 63 + left 63)
# =============================================================================
SINGLE_HAND_KP = 63
TWO_HAND_KP    = 126

SEQ_LEN         = 30
sequence_length = SEQ_LEN

MOTION_THRESHOLD = 0.010

# =============================================================================
# PREPROCESSING STRATEGIES
# More aggressive than before -- needed for real-world photos
# =============================================================================
STRATEGY_NAMES = [
    "original",       # 0 -- as-is
    "clahe",          # 1 -- contrast equalisation
    "brighten",       # 2 -- +40 brightness
    "darken",         # 3 -- -40 brightness
    "sharpen",        # 4 -- unsharp mask
    "flip",           # 5 -- mirror
    "clahe+brighten", # 6 -- clahe then brighten
    "gamma_bright",   # 7 -- gamma < 1 (brighten shadows)
    "gamma_dark",     # 8 -- gamma > 1 (darken highlights)
    "resize_square",  # 9 -- force square crop/pad (fixes NORM_RECT warning)
]

def _to_square(frame):
    """Pad image to square with black bars -- fixes MediaPipe NORM_RECT warning."""
    h, w = frame.shape[:2]
    if h == w:
        return frame
    side = max(h, w)
    out  = np.zeros((side, side, 3), dtype=np.uint8)
    y0   = (side - h) // 2
    x0   = (side - w) // 2
    out[y0:y0+h, x0:x0+w] = frame
    return out

def preprocess_frame(frame, strategy_idx):
    is_flipped = False
    if strategy_idx == 0:
        return _to_square(frame), is_flipped
    elif strategy_idx == 1:
        sq   = _to_square(frame)
        gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq   = clahe.apply(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR), is_flipped
    elif strategy_idx == 2:
        sq = _to_square(frame)
        return np.clip(sq.astype(np.int16)+40, 0, 255).astype(np.uint8), is_flipped
    elif strategy_idx == 3:
        sq = _to_square(frame)
        return np.clip(sq.astype(np.int16)-40, 0, 255).astype(np.uint8), is_flipped
    elif strategy_idx == 4:
        sq     = _to_square(frame)
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        return cv2.filter2D(sq, -1, kernel), is_flipped
    elif strategy_idx == 5:
        is_flipped = True
        return _to_square(cv2.flip(frame, 1)), is_flipped
    elif strategy_idx == 6:
        sq    = _to_square(frame)
        gray  = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq    = clahe.apply(gray)
        bright = np.clip(
            cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR).astype(np.int16)+40,
            0, 255).astype(np.uint8)
        return bright, is_flipped
    elif strategy_idx == 7:
        sq  = _to_square(frame)
        lut = np.array([min(255, int(255*(i/255)**0.6))
                        for i in range(256)], dtype=np.uint8)
        return cv2.LUT(sq, lut), is_flipped
    elif strategy_idx == 8:
        sq  = _to_square(frame)
        lut = np.array([min(255, int(255*(i/255)**1.6))
                        for i in range(256)], dtype=np.uint8)
        return cv2.LUT(sq, lut), is_flipped
    elif strategy_idx == 9:
        # square + strong clahe + brighten
        sq    = _to_square(frame)
        gray  = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        eq    = clahe.apply(gray)
        out   = np.clip(
            cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR).astype(np.int16)+60,
            0, 255).astype(np.uint8)
        return out, is_flipped
    return _to_square(frame), is_flipped

# =============================================================================
# LANDMARKER -- SINGLE HAND  (num_hands=1)
# Used by: data.py, trainmodel.py, predict.py letter branch
# Lower confidence thresholds improve detection on static photos
# =============================================================================
def create_landmarker():
    task_path    = os.path.join(ROOT, 'models', 'hand_landmarker.task')
    base_options = BaseOptions(model_asset_path=task_path)
    options      = HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.25,   # lowered from 0.5
        min_hand_presence_confidence=0.25,    # lowered from 0.5
        min_tracking_confidence=0.25,         # lowered from 0.5
        running_mode=vision.RunningMode.IMAGE
    )
    return HandLandmarker.create_from_options(options)

# =============================================================================
# LANDMARKER -- TWO HANDS  (num_hands=2)
# Used by: collect_word_data.py, predict.py word branch
# =============================================================================
def create_landmarker_two_hands():
    task_path    = os.path.join(ROOT, 'models', 'hand_landmarker.task')
    base_options = BaseOptions(model_asset_path=task_path)
    options      = HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.25,
        running_mode=vision.RunningMode.IMAGE
    )
    return HandLandmarker.create_from_options(options)

# =============================================================================
# DETECTION  -- tries all strategies until hand found
# =============================================================================
def mediapipe_detection(frame, landmarker):
    best_result   = None
    best_strategy = 0
    is_flipped    = False

    for i in range(len(STRATEGY_NAMES)):
        processed, flipped = preprocess_frame(frame, i)
        rgb    = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)
        if result.hand_landmarks:
            best_result   = result
            best_strategy = i
            is_flipped    = flipped
            break

    if best_result is None:
        best_result = type('R',(object,),{'hand_landmarks':[],'handedness':[]})()

    return frame.copy(), best_result, best_strategy, is_flipped

# =============================================================================
# DRAW LANDMARKS
# =============================================================================
def draw_styled_landmarks(image, results, is_flipped=False):
    if not results.hand_landmarks:
        return
    h, w   = image.shape[:2]
    colors = [(0,255,160),(0,165,255)]

    for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
        color  = colors[hand_idx % 2]
        points = []
        for lm in hand_landmarks:
            x = w - int(lm.x * w) if is_flipped else int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
        for start, end in HAND_CONNECTIONS:
            cv2.line(image, points[start], points[end], (60,60,60), 1)
        for pt in points:
            cv2.circle(image, pt, 4, color, -1)

# =============================================================================
# NORMALIZE
# =============================================================================
def normalize_keypoints(keypoints):
    kp    = keypoints.reshape(21, 3).copy()
    wrist = kp[0].copy()
    kp    = kp - wrist
    scale = np.linalg.norm(kp[9])
    if scale > 0:
        kp = kp / scale
    return kp.flatten()

# =============================================================================
# EXTRACT KEYPOINTS -- 63 features (single dominant hand)
# Used by: data.py, trainmodel.py, predict.py letter branch
# =============================================================================
def extract_keypoints(results, is_flipped=False):
    if not results.hand_landmarks:
        return np.zeros(SINGLE_HAND_KP)
    hand = results.hand_landmarks[0]
    kp   = np.array([[lm.x, lm.y, lm.z] for lm in hand]).flatten()
    if is_flipped:
        kp_r      = kp.reshape(21, 3).copy()
        kp_r[:,0] = 1.0 - kp_r[:,0]
        kp        = kp_r.flatten()
    return normalize_keypoints(kp)

# =============================================================================
# EXTRACT KEYPOINTS -- 126 features (both hands)
# Used by: collect_word_data.py, trainmodel_words.py, predict.py word branch
# =============================================================================
def extract_keypoints_two_hands(results, is_flipped=False):
    right_kp = np.zeros(SINGLE_HAND_KP)
    left_kp  = np.zeros(SINGLE_HAND_KP)

    if not results.hand_landmarks:
        return np.concatenate([right_kp, left_kp])

    handedness_list = getattr(results, 'handedness', [])

    for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
        kp = np.array([[lm.x, lm.y, lm.z]
                        for lm in hand_landmarks]).flatten()
        if is_flipped:
            kp_r      = kp.reshape(21, 3).copy()
            kp_r[:,0] = 1.0 - kp_r[:,0]
            kp        = kp_r.flatten()
        kp_norm = normalize_keypoints(kp)
        label   = 'Right'
        if handedness_list and hand_idx < len(handedness_list):
            cats = handedness_list[hand_idx]
            if cats:
                label = cats[0].category_name
        if label == 'Right':
            right_kp = kp_norm
        else:
            left_kp  = kp_norm

    return np.concatenate([right_kp, left_kp])

# =============================================================================
# MOTION + UTILITIES
# =============================================================================
def compute_motion_score(frame_buffer):
    if len(frame_buffer) < 2:
        return 0.0
    frames = np.array(list(frame_buffer))
    deltas = np.diff(frames, axis=0)
    return float(np.mean(np.abs(deltas)))

def text_to_tokens(text):
    tokens = []
    for ch in text.upper():
        if ch.isalpha():
            tokens.append(ch)
        elif ch == ' ' and tokens and tokens[-1] != '_SPACE_':
            tokens.append('_SPACE_')
    if tokens and tokens[-1] == '_SPACE_':
        tokens.pop()
    return tokens

def resample_sequence(frames_list, target_len=30):
    n = len(frames_list)
    if n == 0:
        return np.zeros((target_len, TWO_HAND_KP))
    arr = np.array(frames_list)
    if n == target_len:
        return arr
    idx = np.linspace(0, n-1, target_len).astype(int)
    return arr[idx]