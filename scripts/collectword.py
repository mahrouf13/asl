# scripts/collect_word_data.py
# Browser opens reference video for each word -- no ASL knowledge needed
# RUN: python scripts/collect_word_data.py

import os, sys, cv2, numpy as np, webbrowser, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from function import (create_landmarker, mediapipe_detection,
                      draw_styled_landmarks, extract_keypoints,
                      TWO_HAND_KP)

WORD_DATA_PATH = os.path.join(ROOT, 'MP_Data_Words')
SEQ_LEN    = 30
SEQUENCES  = 40

WORD_LIST = [
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
]

ASL_DESC = {
    'hello'    :'Open hand at forehead, wave outward (salute-wave)',
    'goodbye'  :'Open hand, wave fingers folding down repeatedly',
    'thankyou' :'Flat hand touches chin then moves forward and down',
    'sorry'    :'A-hand fist makes circles on chest',
    'please'   :'Flat palm on chest, rub in circles',
    'yes'      :'S-hand fist nods forward like nodding yes',
    'no'       :'Index + middle snap closed to thumb twice',
    'help'     :'Thumb-up fist on flat palm -- BOTH hands lift upward [2 HANDS]',
    'stop'     :'Dominant flat hand chops down onto non-dominant palm [2 HANDS]',
    'more'     :'Both flat-O hands tap fingertips together twice [2 HANDS]',
    'want'     :'Both curved hands pull toward body, fingers bent [2 HANDS]',
    'again'    :'Bent hand arcs and taps into flat non-dominant palm [2 HANDS]',
    'me'       :'Index finger points at own chest',
    'you'      :'Index finger points forward at other person',
    'mother'   :'5-hand spread fingers, dominant thumb taps chin',
    'father'   :'5-hand spread fingers, dominant thumb taps forehead',
    'friend'   :'Hook index fingers, flip which is on top, hook again [2 HANDS]',
    'eat'      :'Flat-O hand fingers to thumb, tap mouth 2-3 times',
    'drink'    :'C-hand like holding cup, tilt toward mouth',
    'love'     :'Cross BOTH arms over chest hug yourself [2 HANDS]',
    'like'     :'Middle finger + thumb pinch chest, pull outward',
    'go'       :'Both index fingers point, arc forward away from body [2 HANDS]',
    'come'     :'Both index fingers point out, curl toward yourself [2 HANDS]',
    'good'     :'Flat hand at chin moves down onto other flat palm [2 HANDS]',
    'bad'      :'Flat hand at mouth flips down and away',
    'what'     :'Open hands out, index brushes down across fingers [2 HANDS]',
    'where'    :'Index finger wags side to side in the air',
    'who'      :'Index finger makes small circle at chin or lips',
    'why'      :'Fingers touch forehead, pull away bending into Y shape',
    'how'      :'Both bent-hands knuckles touching, roll forward [2 HANDS]',
    'water'    :'W-hand 3 fingers up, tap chin twice',
    'food'     :'Flat-O hand, tap mouth once',
    'home'     :'Flat-O at mouth, then open hand moves to cheek',
    'work'     :'Both S-fists, dominant wrist taps other wrist [2 HANDS]',
    'school'   :'Non-dominant palm up, dominant claps onto it twice [2 HANDS]',
    'doctor'   :'Fingertips tap wrist pulse point twice',
    'pain'     :'Both index fingers point at each other, jab in and out [2 HANDS]',
    'happy'    :'Flat hand on chest, brush upward twice',
    'sad'      :'Both open hands in front of face, slide slowly down [2 HANDS]',
    'angry'    :'Curved hand in front of face, pull fingers inward tensely',
    'tired'    :'Both bent-hands on chest, drop rotate forward slump [2 HANDS]',
    'hungry'   :'C-hand at upper chest, slide down toward stomach',
    'hot'      :'Curved hand at mouth, twist and pull away to side',
    'cold'     :'Both S-fists near shoulders, shake like shivering [2 HANDS]',
    'name'     :'Both H-hands index+middle together, dominant taps other [2 HANDS]',
    'today'    :'Both Y-hands thumb+pinky out, lower simultaneously [2 HANDS]',
    'tomorrow' :'Thumb of A-hand on cheek, arc forward',
    'morning'  :'Non-dominant arm flat, dominant flat hand rises up arm [2 HANDS]',
    'night'    :'Non-dominant arm flat, dominant bent hand arcs over it [2 HANDS]',
}

TWO_HAND_WORDS = {
    'help','stop','more','want','again','friend','love','go','come',
    'good','what','how','work','school','pain','sad','tired',
    'cold','name','today','morning','night',
}

def get_url(word):
    return f"https://www.signasl.org/sign/{word.replace(' ','-').lower()}"

def count_done(word):
    d = os.path.join(WORD_DATA_PATH, word)
    if not os.path.exists(d):
        return 0
    return sum(1 for s in os.listdir(d)
               if os.path.isdir(os.path.join(d, s)) and
               os.path.exists(os.path.join(d, s, '29.npy')))

def draw_wrapped(img, text, x, y, col, scale=0.44, maxc=55):
    words, line, dy = text.split(), '', 0
    for w in words:
        t = (line + ' ' + w).strip()
        if len(t) > maxc:
            cv2.putText(img, line, (x, y+dy), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, col, 1, cv2.LINE_AA)
            dy += int(scale * 30)
            line = w
        else:
            line = t
    if line:
        cv2.putText(img, line, (x, y+dy), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, col, 1, cv2.LINE_AA)

for word in WORD_LIST:
    os.makedirs(os.path.join(WORD_DATA_PATH, word), exist_ok=True)

todo = [w for w in WORD_LIST if count_done(w) < SEQUENCES]
done = [w for w in WORD_LIST if count_done(w) >= SEQUENCES]

print("=" * 62)
print("  ASL WORD COLLECTION  --  44 Words  (126 keypoints)")
print("=" * 62)
print(f"  Features per frame : {TWO_HAND_KP} (right 63 + left 63)")
print(f"  Done: {len(done)}   Todo: {len(todo)}")
print()
print("  ENTER=Record  L=Reopen video  S=Skip  Q=Quit")
print()

if not todo:
    print("  All words collected!")
    print("  Run: python src/trainmodel_words.py")
    sys.exit(0)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    sys.exit(1)

with create_landmarker() as landmarker:

    for wi, word in enumerate(todo):
        existing  = count_done(word)
        needed    = SEQUENCES - existing
        desc      = ASL_DESC.get(word, 'Check signasl.org for this sign')
        url       = get_url(word)
        two_hands = word in TWO_HAND_WORDS

        print(f"\n  [{wi+1}/{len(todo)}] {word.upper()}"
              f"  {'[2 HANDS]' if two_hands else '[1 HAND]'}")
        print(f"  Opening: {url}")
        webbrowser.open(url)
        time.sleep(1)

        # ----------------------------------------------------------------
        # LEARN SCREEN
        # ----------------------------------------------------------------
        skip = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame  = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            img    = cv2.addWeighted(frame, 0.50,
                                     np.zeros_like(frame), 0.50, 0)

            _, results, _, is_flipped = mediapipe_detection(frame, landmarker)
            draw_styled_landmarks(img, results, is_flipped)

            # header
            hdr_col = (30, 210, 255) if two_hands else (0, 200, 255)
            cv2.rectangle(img, (0, 0), (fw, 68), (12, 12, 18), -1)
            cv2.line(img, (0, 68), (fw, 68), hdr_col, 1)
            cv2.putText(img, f"LEARN:  {word.upper()}",
                        (10, 26), cv2.FONT_HERSHEY_DUPLEX,
                        0.85, hdr_col, 1, cv2.LINE_AA)

            if two_hands:
                cv2.rectangle(img, (fw-125, 6), (fw-4, 42), (20, 50, 0), -1)
                cv2.rectangle(img, (fw-125, 6), (fw-4, 42), (30, 210, 255), 1)
                cv2.putText(img, "BOTH HANDS",
                            (fw-118, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.40, (30, 210, 255), 1)

            cv2.putText(img,
                        f"Word {wi+1}/{len(todo)}  --  need {needed} recordings",
                        (10, 54), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (100, 100, 100), 1)

            # description panel
            cv2.rectangle(img, (0, 70), (fw, 158), (18, 18, 28), -1)
            cv2.putText(img, "HOW TO SIGN:",
                        (10, 88), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (0, 255, 160), 1)
            draw_wrapped(img, desc, 10, 110, (210, 210, 210))

            # hand count indicator
            n_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
            req     = 2 if two_hands else 1
            hc      = (0, 220, 0) if n_hands >= req else (0, 0, 200)
            cv2.circle(img, (fw-22, 84), 9, hc, -1)
            cv2.putText(img,
                        f"{n_hands} hand{'s' if n_hands != 1 else ''}",
                        (fw-75, 104), cv2.FONT_HERSHEY_SIMPLEX,
                        0.33, hc, 1)
            if two_hands and n_hands < 2:
                cv2.putText(img, "Need 2 hands!",
                            (fw-100, 122), cv2.FONT_HERSHEY_SIMPLEX,
                            0.33, (0, 0, 200), 1)

            # url bar
            cv2.rectangle(img, (0, fh-96), (fw, fh-62), (18, 18, 28), -1)
            cv2.putText(img, "Reference (check your browser):",
                        (10, fh-80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (70, 70, 70), 1)
            cv2.putText(img, url,
                        (10, fh-62), cv2.FONT_HERSHEY_SIMPLEX,
                        0.37, (0, 165, 255), 1)

            # controls bar
            cv2.rectangle(img, (0, fh-58), (fw, fh), (12, 12, 18), -1)
            cv2.line(img, (0, fh-58), (fw, fh-58), (0, 255, 160), 1)
            cv2.putText(img,
                        "ENTER=Start Recording   L=Reopen Video   S=Skip   Q=Quit",
                        (8, fh-36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, (0, 255, 160), 1)
            cv2.putText(img,
                        "Practice in the mirror until confident, then press ENTER",
                        (8, fh-12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (80, 80, 80), 1)

            cv2.imshow('ASL Word Collection', img)
            k = cv2.waitKey(30) & 0xFF

            if k == 13:
                break
            elif k == ord('l'):
                webbrowser.open(url)
            elif k == ord('s'):
                skip = True
                break
            elif k == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("\n  Stopped. Progress saved automatically.")
                print("  Run again to continue from where you left off.")
                sys.exit(0)

        if skip:
            print(f"  [SKIP] {word}")
            continue

        # ----------------------------------------------------------------
        # RECORD SEQUENCES
        # ----------------------------------------------------------------
        seq_idx    = existing
        bad_streak = 0
        min_hands  = 2 if two_hands else 1

        print(f"  Recording {word.upper()} "
              f"({'2-hand' if two_hands else '1-hand'}, "
              f"{needed} sequences needed) ...")

        while seq_idx < SEQUENCES:

            # countdown 3-2-1
            for cd in range(3, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    break
                frame  = cv2.flip(frame, 1)
                fh, fw = frame.shape[:2]
                img    = cv2.addWeighted(frame, 0.38,
                                         np.zeros_like(frame), 0.62, 0)

                cv2.putText(img, word.upper(),
                            (20, 90), cv2.FONT_HERSHEY_DUPLEX,
                            2.0, (0, 255, 160), 3, cv2.LINE_AA)
                cv2.putText(img, f"Sequence {seq_idx+1} / {SEQUENCES}",
                            (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (180, 180, 180), 1)
                cv2.putText(img, str(cd),
                            (fw//2-22, fh//2+30),
                            cv2.FONT_HERSHEY_DUPLEX,
                            3.5, (0, 200, 255), 5, cv2.LINE_AA)
                if two_hands:
                    cv2.putText(img, "USE BOTH HANDS",
                                (fw//2-110, fh//2+80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (30, 210, 255), 2)
                cv2.putText(img, desc[:60],
                            (10, fh-20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (90, 90, 90), 1)

                cv2.imshow('ASL Word Collection', img)
                cv2.waitKey(1000)

            # record 30 frames
            frames_kp   = []
            hand_frames = 0
            max_hands   = 0

            for fn in range(SEQ_LEN):
                ret, frame = cap.read()
                if not ret:
                    break
                frame  = cv2.flip(frame, 1)
                fh, fw = frame.shape[:2]

                image, results, _, is_flipped = mediapipe_detection(
                    frame, landmarker)
                draw_styled_landmarks(image, results, is_flipped)

                kp    = extract_keypoints(results, is_flipped)
                frames_kp.append(kp)

                n_det = len(results.hand_landmarks) \
                        if results.hand_landmarks else 0
                max_hands = max(max_hands, n_det)
                if n_det >= min_hands:
                    hand_frames += 1

                # right / left hand dot indicators
                right_on = False
                left_on  = False
                if results.hand_landmarks:
                    handedness = getattr(results, 'handedness', [])
                    for hi in range(len(results.hand_landmarks)):
                        if handedness and hi < len(handedness):
                            lbl = handedness[hi][0].category_name
                            if lbl == 'Right':
                                right_on = True
                            else:
                                left_on  = True
                        else:
                            right_on = True

                rc = (0, 220, 0) if right_on else (50, 50, 50)
                lc = (0, 220, 0) if left_on  else (50, 50, 50)
                cv2.circle(image, (fw-22, 28), 8, rc, -1)
                cv2.putText(image, "R", (fw-29, 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, rc, 1)
                cv2.circle(image, (fw-22, 60), 8, lc, -1)
                cv2.putText(image, "L", (fw-29, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, lc, 1)

                if two_hands and n_det < 2:
                    cv2.putText(image, "Need 2 hands!",
                                (fw//2-95, fh//2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 0, 255), 2)

                # progress bar
                prog = int((fn+1) / SEQ_LEN * (fw-10))
                cv2.rectangle(image, (0, fh-20), (fw, fh), (20, 20, 28), -1)
                if prog > 0:
                    cv2.rectangle(image, (0, fh-20), (prog, fh),
                                  (0, 255, 160), -1)

                cv2.putText(image,
                            f"REC  {word.upper()}  "
                            f"Seq {seq_idx+1}/{SEQUENCES}",
                            (10, 38), cv2.FONT_HERSHEY_SIMPLEX,
                            0.80, (0, 0, 200), 2)
                cv2.putText(image,
                            f"Frame {fn+1} / {SEQ_LEN}",
                            (10, 68), cv2.FONT_HERSHEY_SIMPLEX,
                            0.50, (160, 160, 160), 1)

                if (fn+1 - hand_frames) > 12:
                    cv2.putText(image,
                                "Keep hand in frame!",
                                (fw//2-120, fh//2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 255), 2)

                cv2.imshow('ASL Word Collection', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\n  Stopped. Progress saved.")
                    sys.exit(0)

            # quality check
            if hand_frames / SEQ_LEN < 0.5:
                bad_streak += 1
                msg = (f"both hands only {hand_frames}/{SEQ_LEN} frames "
                       f"(max seen: {max_hands})"
                       if two_hands else
                       f"hand only {hand_frames}/{SEQ_LEN} frames")
                print(f"\r  [REDO] {msg} (redo #{bad_streak})")

                ret, frame = cap.read()
                if ret:
                    frame  = cv2.flip(frame, 1)
                    fh, fw = frame.shape[:2]
                    warn   = cv2.addWeighted(frame, 0.25,
                                             np.zeros_like(frame), 0.75, 0)
                    line1 = ("Both hands not visible -- redo"
                             if two_hands else
                             "Hand not visible enough -- redo")
                    line2 = ("Bring both hands fully into frame"
                             if two_hands else
                             "Keep hand fully in frame")
                    cv2.putText(warn, line1,
                                (fw//2-210, fh//2-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.85, (0, 0, 255), 2)
                    cv2.putText(warn, line2,
                                (fw//2-195, fh//2+35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.60, (0, 165, 255), 1)
                    cv2.putText(warn, "Good lighting helps a lot",
                                (fw//2-160, fh//2+70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (100, 100, 100), 1)
                    cv2.imshow('ASL Word Collection', warn)
                    cv2.waitKey(1800)
                continue

            # save
            bad_streak = 0
            seq_dir = os.path.join(WORD_DATA_PATH, word, str(seq_idx))
            os.makedirs(seq_dir, exist_ok=True)
            for fn2, kp in enumerate(frames_kp):
                np.save(os.path.join(seq_dir, str(fn2)), kp)

            seq_idx += 1
            filled   = int(30 * seq_idx / SEQUENCES)
            bar_s    = '#' * filled + '-' * (30 - filled)
            print(f"\r  [{bar_s}] {seq_idx}/{SEQUENCES} "
                  f"({int(seq_idx/SEQUENCES*100)}%)  "
                  f"hand_frames={hand_frames}/{SEQ_LEN}     ",
                  end='', flush=True)

        print()
        print(f"  [DONE] {word.upper()}  --  {seq_idx} sequences saved")

# ----------------------------------------------------------------
# FINAL SUMMARY
# ----------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()

print()
print("=" * 62)
print("  COLLECTION SUMMARY")
print("=" * 62)
print()
print(f"  {'Word':<20} {'Done':>6}  {'Type':<10}  Status")
print(f"  {'-'*52}")

ready = 0
for word in WORD_LIST:
    cnt  = count_done(word)
    th   = "[2H]" if word in TWO_HAND_WORDS else "[1H]"
    if cnt >= SEQUENCES:
        status = "[READY]"
        ready += 1
    elif cnt >= 20:
        status = f"[{cnt}/{SEQUENCES} -- run again]"
    elif cnt > 0:
        status = f"[{cnt}/{SEQUENCES} -- need more]"
    else:
        status = "[NOT STARTED]"
    print(f"  {word:<20} {cnt:>6}  {th:<10}  {status}")

print()
print(f"  Ready : {ready} / {len(WORD_LIST)} words")

if ready >= 5:
    print()
    print("  Next step:")
    print("  python src/trainmodel_words.py")
    print()
    print("  NOTE: trainmodel_words.py LSTM input_shape must be (30, 126)")
else:
    print()
    print("  Run this script again to collect more words.")
    print("  Already-done words are skipped automatically.")