# src/predict.py  --  Sign Language Translator
#
# FOUR MODES, ONE SHARED SENTENCE
# =================================
# MODE 1 - LETTER SIGNING  (green)
#   Hold a letter still -> confirmed -> appended to sentence.
#   Open hand (space class) held still -> space inserted.
#   S key = instant space shortcut.
#
# MODE 2 - WORD SIGNING  (cyan)
#   Sign a word with motion. Buffer is FROZEN at peak motion moment.
#   Word model fires on the frozen buffer -> clean window = correct result.
#   Open hand held still -> space inserted.
#
# MODE 3 - VIDEO REFERENCE  (gold)
#   Browse all signs with videos. Click or arrow to select.
#   Video plays showing how to perform the sign.
#   Read-only -- nothing added to sentence.
#
# MODE 4 - SPEECH TO SIGN  (purple)
#   Speak a sentence -> tokenised -> sign images + videos shown as collage.
#   Words with videos auto-play inline. Letters show sign images.
#
# M = cycle 1->2->3->4->1.  Sentence persists across all switches.

import os, sys, cv2, numpy as np, subprocess, threading
from collections import deque

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT  = _os.path.dirname(_HERE)
_sys.path.insert(0, _os.path.join(ROOT, 'src'))

from function import *

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

from keras.models import load_model

# ===========================================================================
# MODELS
# ===========================================================================
letter_path   = os.path.join(ROOT, 'models', 'model.h5')
word_path     = os.path.join(ROOT, 'models', 'word_model.h5')
word_act_path = os.path.join(ROOT, 'models', 'word_actions.npy')

if not os.path.exists(letter_path):
    print(f"ERROR: Letter model not found: {letter_path}"); sys.exit(1)

letter_model = load_model(letter_path)
WORD_MODEL   = None
WORD_LABELS  = None
WORD_ACTIVE  = False

if os.path.exists(word_path) and os.path.exists(word_act_path):
    WORD_MODEL  = load_model(word_path)
    WORD_LABELS = np.array([str(w) for w in np.load(word_act_path, allow_pickle=True)])
    WORD_ACTIVE = True
    print(f"Letter model : {len(LETTER_SIGNS)} classes")
    print(f"Word model   : {len(WORD_LABELS)} classes")
else:
    print(f"Letter model : {len(LETTER_SIGNS)} classes")
    print("[WARN] Word model not found -- mode 2 unavailable")

# ===========================================================================
# PARAMETERS
# ===========================================================================

# -- Mode 1: Letter engine --
EMA_ALPHA    = 0.65
CONSEC_NEED  = 12
THR_LETTER   = 0.72
COOLDOWN_L   = 45

SPACE_IDX    = len(LETTER_SIGNS) - 1
THR_SPACE    = 0.82
CONSEC_SPACE = 14

# -- Mode 2: Word engine --
# KEY FIX: buffer is FROZEN during peak motion and evaluated when hand stops.
# Old bug: buffer was evaluated AFTER settling, so it contained mostly still
# frames rather than the actual sign. Now frozen=snapshot of buffer at peak.
MOTION_DELTA  = 0.045   # frame delta above this = actively signing
STILL_DELTA   = 0.030   # frame delta below this = hand stopped
CONSEC_MOT    = 3       # frames above MOTION_DELTA to arm
CONSEC_STILL  = 3       # frames below STILL_DELTA to fire
THR_WORD      = 0.78  # raised: low conf = wrong gesture, reject it
COOLDOWN_W    = 50

CONSEC_SPACE_W = 20

MAX_SENT = 150

# ===========================================================================
# ASSETS
# ===========================================================================
VIDEO_DIR  = os.path.join(ROOT, 'videos')
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

def find_video(word):
    if not os.path.isdir(VIDEO_DIR): return None
    for ext in VIDEO_EXTS:
        p = os.path.join(VIDEO_DIR, word.lower() + ext)
        if os.path.exists(p): return p
    return None

word_videos    = {}
all_sign_words = set(str(w) for w in WORD_SIGNS)
if WORD_ACTIVE: all_sign_words |= set(str(w) for w in WORD_LABELS)
for w in all_sign_words:
    v = find_video(w)
    if v: word_videos[w] = v
print(f"Sign videos  : {len(word_videos)} found")

sign_images = {}
for letter in LETTER_SIGNS:
    if letter == 'space': continue
    folder = os.path.join(ROOT, 'data', letter)
    if not os.path.exists(folder): continue
    imgs = sorted([f for f in os.listdir(folder)
                   if f.lower().endswith(('.png','.jpg','.jpeg'))
                   and not f.startswith('aug_')])
    if imgs:
        img = cv2.imread(os.path.join(folder, imgs[0]))
        if img is not None:
            sign_images[letter] = cv2.resize(img, (100, 100))
print(f"Sign images  : {len(sign_images)}/26")

# ===========================================================================
# TTS
# ===========================================================================
def speak_windows(text):
    # Strip chars that break PowerShell strings
    safe = text.replace('"','').replace("'",'').replace('`','').replace('\n',' ').strip()
    if not safe: return
    script = (
        'Add-Type -AssemblyName System.Speech; '
        '$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
        '$s.Rate = 0; $s.Volume = 100; '
        f'$s.Speak("{safe}")'
    )
    def _run():
        try:
            subprocess.run(['powershell', '-Command', script],
                           shell=False, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[TTS] error: {e}")
    threading.Thread(target=_run, daemon=True).start()
    print(f"[TTS] {text}")

# ===========================================================================
# MICROPHONE  (mode 4)
# ===========================================================================
recognizer     = None
mic_obj        = None
listening_flag = threading.Event()
heard_text     = {"value": ""}
mic_status     = {"state": "unavailable"}   # "listening" | "idle" | "unavailable"

if SR_AVAILABLE:
    try:
        recognizer = sr.Recognizer()
        recognizer.energy_threshold    = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold     = 0.6    # shorter pause = faster response
        mic_obj = sr.Microphone()
        print("Calibrating mic...")
        with mic_obj as src:
            recognizer.adjust_for_ambient_noise(src, duration=1)
        mic_status["state"] = "idle"
        print("Mic ready!")

        def _listen_bg():
            while True:
                listening_flag.wait()   # blocks until mode 4 is active
                mic_status["state"] = "listening"
                try:
                    with mic_obj as src:
                        audio = recognizer.listen(src, timeout=4, phrase_time_limit=8)
                    mic_status["state"] = "processing"
                    text = recognizer.recognize_google(audio)
                    print(f"[MIC] heard: {text!r}")
                    heard_text["value"] = text
                    mic_status["state"] = "listening"
                except sr.WaitTimeoutError:
                    mic_status["state"] = "listening"   # silence, keep looping
                except sr.UnknownValueError:
                    mic_status["state"] = "listening"   # couldn't understand, keep looping
                except Exception as e:
                    print(f"[MIC] {e}")
                    mic_status["state"] = "listening"

        threading.Thread(target=_listen_bg, daemon=True).start()
    except Exception as e:
        print(f"[WARN] Mic: {e}")
        SR_AVAILABLE = False

# ===========================================================================
# VIDEO PLAYER
# ===========================================================================
class VideoPlayer:
    def __init__(self):
        self.frames=[]; self.idx=0; self.tick=0
        self.frame_dur=1; self.done=False; self.word=None

    def load(self, path, word, w, h):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): self.frames=[]; return False
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        self.frame_dur = max(1, round(30.0/fps))
        self.frames = []
        while True:
            ret, f = cap.read()
            if not ret: break
            self.frames.append(cv2.resize(f,(w,h)))
        cap.release()
        self.word=word; self.idx=0; self.tick=0
        self.done=(len(self.frames)==0)
        return len(self.frames)>0

    def current_frame(self):
        if not self.frames: return None
        return self.frames[min(self.idx, len(self.frames)-1)]

    def advance(self):
        if not self.frames or self.done: return True
        self.tick+=1
        if self.tick>=self.frame_dur:
            self.tick=0; self.idx+=1
            if self.idx>=len(self.frames): self.done=True; return True
        return False

    def reset(self): self.idx=0; self.tick=0; self.done=False

# ===========================================================================
# SHARED STATE
# ===========================================================================
sentence    = []
speak_flash = 0
ui_mode     = 1

# Mode 1
l_ema=None; l_consec=0; l_consec_cls=-1; l_consec_conf=0.0
l_cooldown=0; l_pred_label=''; l_pred_conf=0.0; l_top5_probs=None

# Mode 2
w_buf        = deque(maxlen=SEQ_LEN)  # rolling buffer always accumulating
w_sign_frames= []     # grows during active signing, evaluated on stop
w_prev_kp    = None
w_delta      = 0.0
w_mot_cnt    = 0
w_mot_seen   = False   # motion confirmed (arm built up)
w_still_cnt  = 0
w_cooldown   = 0
w_peak_delta = 0.0
w_frozen     = None    # kept for compat with reset fn
w_pred_label=''; w_pred_conf=0.0; w_top5_probs=None
ws_ema=None; ws_consec=0; ws_cls=-1; ws_cooldown=0
# Space gesture detection in mode 2 (letter_lm, single hand)
w_spc_ema=None; w_spc_consec=0; w_spc_cooldown=0
# Independent space EMA for mode 1 (tracks space prob directly, not via argmax)
l_spc_ema=None; l_spc_consec=0

# Mode 3
m3_all    = sorted([w for w in all_sign_words if find_video(w) is not None])
m3_sel=0; m3_scroll=0; m3_vp=VideoPlayer(); m3_loaded=False

# Mode 4
m4_tokens=[]; m4_idx=0; m4_timer=0; m4_text=""
m4_vp=VideoPlayer(); m4_vp_word=""
LETTER_DUR=30

# ===========================================================================
# HELPERS
# ===========================================================================
def append_to_sentence(token, source):
    global sentence
    if token == 'space':
        if sentence and sentence[-1] != ' ':
            sentence.append(' ')
            print(f"+ space [{source}]  -> {''.join(sentence)!r}")
    else:
        sentence.append(token)
        print(f"+ {token:<16} [{source}]  -> {''.join(sentence)!r}")
    if len(sentence)>MAX_SENT: del sentence[:-MAX_SENT]

def reset_letter_engine():
    global l_ema,l_consec,l_consec_cls,l_consec_conf,l_pred_label,l_pred_conf,l_top5_probs,l_spc_ema,l_spc_consec
    l_ema=None; l_consec=0; l_consec_cls=-1; l_consec_conf=0.0
    l_pred_label=''; l_pred_conf=0.0; l_top5_probs=None
    l_spc_ema=None; l_spc_consec=0

def reset_word_engine():
    global w_buf,w_sign_frames,w_frozen,w_peak_delta,w_prev_kp,w_delta
    global w_mot_cnt,w_mot_seen,w_still_cnt
    global w_pred_label,w_pred_conf,w_top5_probs,ws_ema,ws_consec,ws_cls
    global w_spc_ema,w_spc_consec,w_spc_cooldown
    w_buf.clear(); w_sign_frames=[]; w_frozen=None; w_peak_delta=0.0
    w_prev_kp=None; w_delta=0.0
    w_mot_cnt=0; w_mot_seen=False; w_still_cnt=0
    w_pred_label=''; w_pred_conf=0.0; w_top5_probs=None
    ws_ema=None; ws_consec=0; ws_cls=-1
    w_spc_ema=None; w_spc_consec=0; w_spc_cooldown=0

def tokenise_speech(text):
    known = set(str(w).lower() for w in WORD_LABELS) if WORD_ACTIVE else set()
    known |= {k.lower() for k in word_videos}
    tokens = []
    for raw in text.strip().split():
        w = raw.lower().strip(".,!?;:'\"")
        if w in known:
            tokens.append({'type':'word','val':w})
        else:
            for ch in w.upper():
                if ch.isalpha():
                    tokens.append({'type':'letter','val':ch})
        tokens.append({'type':'space'})
    if tokens and tokens[-1]['type']=='space':
        tokens.pop()
    return tokens

# ===========================================================================
# MODE 3 DRAW
# ===========================================================================
_DW3=210; _IH3=30; _VI3=16

def draw_m3(img, fw, fh):
    GOLD=(30,200,220); DIM=(60,60,60); SEL=(40,35,0); ALT=(22,22,28); NV=(30,30,45)
    cv2.rectangle(img,(0,0),(fw,54),(12,12,18),-1)
    cv2.line(img,(0,54),(fw,54),GOLD,1)
    cv2.putText(img,"MODE 3  VIDEO REFERENCE",
                (8,22),cv2.FONT_HERSHEY_SIMPLEX,0.60,GOLD,1,cv2.LINE_AA)
    cv2.putText(img,"UP/DOWN or click  SPACE:replay  M:next mode  Q:quit",
                (8,44),cv2.FONT_HERSHEY_SIMPLEX,0.31,(90,90,90),1,cv2.LINE_AA)
    cv2.rectangle(img,(0,54),(_DW3,fh),(18,18,24),-1)
    cv2.line(img,(_DW3,54),(_DW3,fh),(40,40,55),1)
    cv2.putText(img,f"{len(m3_all)} videos",(6,76),
                cv2.FONT_HERSHEY_SIMPLEX,0.34,(0,200,255),1)
    ly0=84
    for li in range(_VI3):
        wi=m3_scroll+li
        if wi>=len(m3_all): break
        item=m3_all[wi]
        iy0=ly0+li*_IH3; iy1=iy0+_IH3-1
        bg=SEL if wi==m3_sel else (ALT if wi%2==0 else (15,15,20))
        cv2.rectangle(img,(1,iy0),(_DW3-1,iy1),bg,-1)
        tc=GOLD if wi==m3_sel else (0,200,255)
        cv2.putText(img,item,(8,iy0+20),cv2.FONT_HERSHEY_SIMPLEX,0.44,tc,1)
        cv2.circle(img,(_DW3-10,iy0+14),4,(0,200,100),-1)
    if len(m3_all)>_VI3:
        sbh=fh-ly0; sbx=_DW3-7
        th=max(16,int(sbh*_VI3/len(m3_all)))
        tys=ly0+int((sbh-th)*m3_scroll/max(1,len(m3_all)-_VI3))
        cv2.rectangle(img,(sbx,ly0),(sbx+5,ly0+sbh),(30,30,40),-1)
        cv2.rectangle(img,(sbx,tys),(sbx+5,tys+th),(0,180,160),-1)
    rx=_DW3+8; vw=fw-rx-8; vy=58; vh=fh-58-8
    cw=m3_all[m3_sel] if 0<=m3_sel<len(m3_all) else ''
    if m3_vp.frames:
        vf=m3_vp.current_frame()
        if vf is not None: img[vy:vy+vh,rx:rx+vw]=cv2.resize(vf,(vw,vh))
        cv2.rectangle(img,(rx,vy),(rx+vw,vy+40),(0,0,0),-1)
        cv2.putText(img,cw.upper(),(rx+10,vy+30),
                    cv2.FONT_HERSHEY_DUPLEX,1.0,GOLD,2,cv2.LINE_AA)
        prog=m3_vp.idx/max(1,len(m3_vp.frames)-1)
        pw=int(vw*prog)
        cv2.rectangle(img,(rx,vy+vh),(rx+vw,vy+vh+5),(35,35,50),-1)
        if pw>0: cv2.rectangle(img,(rx,vy+vh),(rx+pw,vy+vh+5),GOLD,-1)
        if m3_vp.done:
            mid=vy+vh//2
            cv2.rectangle(img,(rx,mid-20),(rx+vw,mid+24),(0,0,0),-1)
            cv2.putText(img,"SPACE to replay",(rx+vw//2-95,mid+10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,GOLD,2,cv2.LINE_AA)
    else:
        cv2.rectangle(img,(rx,vy),(rx+vw,vy+vh),NV,-1)
        cv2.putText(img,"No video",(rx+vw//2-55,vy+vh//2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.75,DIM,1)

# ===========================================================================
# MODE 4 DRAW
# ===========================================================================
_TILE=110; _TCOL=6; _TPAD=6

def draw_m4(img, tokens, idx, vp, fw, fh, tick):
    PURPLE=(180,80,255); DIM=(60,60,60); NEON=(0,255,160)
    SEL=(40,0,60); PND=(20,20,35); DNE=(22,22,28)

    cv2.rectangle(img,(0,0),(fw,54),(12,12,18),-1)
    cv2.line(img,(0,54),(fw,54),PURPLE,1)
    cv2.putText(img,"MODE 4  SPEECH TO SIGN",
                (8,22),cv2.FONT_HERSHEY_SIMPLEX,0.60,PURPLE,1,cv2.LINE_AA)
    ms=mic_status.get("state","unavailable")
    mic_col=(0,255,160) if ms=="listening" else (0,200,255) if ms=="processing" else (80,80,80)
    mic_lbl=f"MIC: {ms.upper()}"
    cv2.putText(img,f"{mic_lbl}  |  T:type text  C:clear  M:next mode  Q:quit",
                (8,44),cv2.FONT_HERSHEY_SIMPLEX,0.31,mic_col,1,cv2.LINE_AA)

    if not tokens:
        cy2=fh//2-30
        cv2.putText(img,"Press T to type a sentence",(fw//2-165,cy2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.70,PURPLE,2,cv2.LINE_AA)
        cv2.putText(img,"or speak into the microphone",(fw//2-165,cy2+36),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,DIM,1,cv2.LINE_AA)
        return

    cy=58
    cv2.rectangle(img,(0,cy),(fw,cy+26),(18,18,28),-1)
    cv2.putText(img,f"Heard: {m4_text[:60]}",(8,cy+18),
                cv2.FONT_HERSHEY_SIMPLEX,0.44,(200,200,200),1,cv2.LINE_AA)
    cy+=30
    pct=idx/max(1,len(tokens)); pw=int(fw*pct)
    cv2.rectangle(img,(0,cy),(fw,cy+4),(30,30,40),-1)
    if pw>0: cv2.rectangle(img,(0,cy),(pw,cy+4),PURPLE,-1)
    cy+=8

    cur_tok=tokens[idx] if idx<len(tokens) else None

    # Full-width video player for word tokens with video
    if cur_tok and cur_tok['type']=='word' and cur_tok['val'] in word_videos:
        vw2=fw-16; vh2=fh-cy-68; rx=8
        if vp.frames:
            vf=vp.current_frame()
            if vf is not None:
                img[cy:cy+vh2,rx:rx+vw2]=cv2.resize(vf,(vw2,vh2))
            cv2.rectangle(img,(rx,cy),(rx+vw2,cy+44),(0,0,0),-1)
            cv2.putText(img,cur_tok['val'].upper(),(rx+12,cy+34),
                        cv2.FONT_HERSHEY_DUPLEX,1.1,PURPLE,2,cv2.LINE_AA)
            vpr=vp.idx/max(1,len(vp.frames)-1); ppw=int(vw2*vpr)
            cv2.rectangle(img,(rx,cy+vh2),(rx+vw2,cy+vh2+5),(35,35,50),-1)
            if ppw>0: cv2.rectangle(img,(rx,cy+vh2),(rx+ppw,cy+vh2+5),PURPLE,-1)
        else:
            cv2.rectangle(img,(rx,cy),(rx+vw2,cy+vh2),(20,20,30),-1)
            cv2.putText(img,cur_tok['val'].upper(),(rx+12,cy+vh2//2),
                        cv2.FONT_HERSHEY_DUPLEX,1.5,PURPLE,2,cv2.LINE_AA)
        # Mini collage row below showing upcoming tokens
        mini_y=cy+vh2+10
        for ti in range(idx+1, min(idx+7, len(tokens))):
            tok2=tokens[ti]; col2=ti-idx-1
            mx=8+col2*(_TILE//2+4); my=mini_y
            if mx+_TILE//2>fw: break
            cv2.rectangle(img,(mx,my),(mx+_TILE//2,my+_TILE//2),(22,22,30),-1)
            cv2.rectangle(img,(mx,my),(mx+_TILE//2,my+_TILE//2),(50,50,70),1)
            if tok2['type']=='letter':
                cv2.putText(img,tok2['val'],(mx+12,my+36),
                            cv2.FONT_HERSHEY_DUPLEX,0.75,(0,200,255),2)
            elif tok2['type']=='word':
                cv2.putText(img,tok2['val'][:6],(mx+2,my+28),
                            cv2.FONT_HERSHEY_SIMPLEX,0.32,(0,200,255),1)
    else:
        # Collage grid -- sliding window so active token is always visible
        ts=_TILE+_TPAD
        ry=cy+4
        avail_h=fh-ry-65
        rows_vis=max(1,avail_h//ts)
        active_row=idx//_TCOL
        start_row=max(0,active_row-rows_vis//2)
        start_ti=start_row*_TCOL
        for ti,tok in enumerate(tokens):
            if ti<start_ti: continue
            disp_row=(ti-start_ti)//_TCOL
            disp_col=ti%_TCOL
            if disp_row>=rows_vis: break
            x1=disp_col*ts+4; y1=ry+disp_row*ts; x2=x1+_TILE; y2=y1+_TILE
            is_active=(ti==idx); is_done=(ti<idx)
            bg=SEL if is_active else DNE if is_done else PND
            bc=PURPLE if is_active else (0,80,80) if is_done else (40,40,55)
            cv2.rectangle(img,(x1,y1),(x2,y2),bg,-1)
            cv2.rectangle(img,(x1,y1),(x2,y2),bc,2 if is_active else 1)
            if tok['type']=='space':
                mid=(x1+x2)//2; midy=(y1+y2)//2
                tc=PURPLE if is_active else DIM
                cv2.arrowedLine(img,(x1+12,midy),(x2-12,midy),tc,2,tipLength=0.4)
                cv2.putText(img,"SPC",(mid-14,y2-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.32,tc,1)
            elif tok['type']=='letter':
                ch=tok['val']
                tc=PURPLE if is_active else (0,160,120) if is_done else (0,200,255)
                if ch in sign_images and not is_done:
                    isz=_TILE-22; th=cv2.resize(sign_images[ch],(isz,isz))
                    px=x1+(_TILE-isz)//2; py=y1+2
                    px2=min(px+isz,img.shape[1]); py2=min(py+isz,img.shape[0])
                    tw=px2-px; th2=py2-py
                    if tw>0 and th2>0: img[py:py2,px:px2]=th[:th2,:tw]
                sz=cv2.getTextSize(ch,cv2.FONT_HERSHEY_DUPLEX,0.65,2)[0]
                cv2.putText(img,ch,((x1+x2)//2-sz[0]//2,y2-5),
                            cv2.FONT_HERSHEY_DUPLEX,0.65,tc,2)
            elif tok['type']=='word':
                wv=tok['val']
                tc=PURPLE if is_active else (0,160,120) if is_done else (0,200,255)
                if wv in word_videos:
                    cv2.putText(img,"[VIDEO]",(x1+4,y1+16),
                                cv2.FONT_HERSHEY_SIMPLEX,0.28,(0,180,100),1)
                lines=[wv[i:i+8] for i in range(0,len(wv),8)]
                for li2,ln in enumerate(lines[:2]):
                    cv2.putText(img,ln,(x1+4,y1+34+li2*18),
                                cv2.FONT_HERSHEY_SIMPLEX,0.38,tc,1)
            if is_active:
                cv2.line(img,(x1+4,y1+3),(x2-4,y1+3),PURPLE,2)
    done_t=sum(1 for t in tokens[:idx] if t['type']!='space')
    tot_t=sum(1 for t in tokens if t['type']!='space')
    cv2.putText(img,f"{done_t}/{tot_t} signs",
                (8,fh-58),cv2.FONT_HERSHEY_SIMPLEX,0.38,(120,120,120),1)

# ===========================================================================
# CAMERA
# ===========================================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam"); sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize camera buffer lag

print()
print("="*60)
print("  SIGN LANGUAGE TRANSLATOR  -- 4 MODES")
print(f"  Letters:{len(LETTER_SIGNS)}  Words:{len(WORD_LABELS) if WORD_ACTIVE else 0}  Videos:{len(word_videos)}")
print("="*60)
print("  M      : Cycle  1(Letter)->2(Word)->3(Ref)->4(Speech->Sign)")
print("  SPACE  : Speak sentence  (modes 1,2)")
print("  S      : Insert space  (mode 1)")
print("  B      : Backspace")
print("  C      : Clear")
print("  Q      : Quit")
print("="*60)

def _mouse_cb(event,x,y,flags,param):
    global m3_sel,m3_scroll,m3_loaded
    if ui_mode!=3: return
    ly0=84
    if event==cv2.EVENT_LBUTTONDOWN and x<_DW3 and y>ly0:
        li=(y-ly0)//_IH3; wi=m3_scroll+li
        if 0<=wi<len(m3_all): m3_sel=wi; m3_loaded=False
    elif event==cv2.EVENT_MOUSEWHEEL:
        d=-1 if flags>0 else 1
        m3_scroll=max(0,min(m3_scroll+d,max(0,len(m3_all)-_VI3)))

with create_landmarker() as letter_lm, \
     create_landmarker_two_hands() as word_lm:

    cv2.namedWindow('Sign Language Translator')
    cv2.setMouseCallback('Sign Language Translator',_mouse_cb)

    while cap.isOpened():
        ret,frame=cap.read()
        if not ret: break

        frame=cv2.flip(frame,1)
        fh,fw=frame.shape[:2]
        image=cv2.addWeighted(frame,0.55,np.full_like(frame,(15,15,20)),0.45,0)
        tick=cv2.getTickCount()/cv2.getTickFrequency()

        if speak_flash>0: speak_flash-=1

        # ===================================================================
        # MODE 1 -- LETTER SIGNING
        # ===================================================================
        if ui_mode==1:
            listening_flag.clear()
            if l_cooldown>0: l_cooldown-=1
            try:
                _,res_l,_,flip_l=mediapipe_detection(frame,letter_lm)
            except:
                res_l=type('R',(object,),{'hand_landmarks':[],'handedness':[]})(); flip_l=False
            draw_styled_landmarks(image,res_l,flip_l)
            hand_present=bool(res_l.hand_landmarks)
            if hand_present:
                kp63=extract_keypoints(res_l,flip_l)
                l_probs=letter_model.predict(np.expand_dims(kp63,0),verbose=0)[0]
                if l_ema is None: l_ema=l_probs.copy()
                else: l_ema=EMA_ALPHA*l_ema+(1-EMA_ALPHA)*l_probs
                l_cls=int(np.argmax(l_ema)); l_conf=float(l_ema[l_cls])
                if l_cls==l_consec_cls: l_consec+=1
                else: l_consec_cls=l_cls; l_consec=1
                l_consec_conf=l_conf
                l_pred_label=str(LETTER_SIGNS[l_cls]); l_pred_conf=l_conf; l_top5_probs=l_ema
                # ── Independent space tracking (does NOT reset when argmax flips)
                spc_raw=float(l_probs[SPACE_IDX])
                l_spc_ema=(EMA_ALPHA*spc_raw+(1-EMA_ALPHA)*l_spc_ema
                           if l_spc_ema is not None else spc_raw)
                if l_spc_ema>=THR_SPACE: l_spc_consec+=1
                else: l_spc_consec=0
                if l_cooldown==0 and l_spc_consec>=CONSEC_SPACE:
                    append_to_sentence('space',"SPACE-M1")
                    l_ema=None; l_consec=0; l_consec_cls=-1; l_cooldown=COOLDOWN_L
                    l_spc_ema=None; l_spc_consec=0
                elif l_cooldown==0 and l_consec>=CONSEC_NEED and l_conf>=THR_LETTER and l_cls!=SPACE_IDX:
                    append_to_sentence(str(LETTER_SIGNS[l_cls]),"LETTER")
                    l_ema=None; l_consec=0; l_consec_cls=-1; l_cooldown=COOLDOWN_L
                    l_spc_ema=None; l_spc_consec=0
            else:
                l_ema=None; l_consec=0; l_consec_cls=-1
                l_spc_ema=None; l_spc_consec=0
            if l_top5_probs is not None and hand_present:
                for rank,i in enumerate(np.argsort(l_top5_probs)[::-1][:5]):
                    i=int(i); y1b=60+rank*24; bc=(0,255,160) if rank==0 else (60,60,180)
                    bw=int(l_top5_probs[i]*160)
                    cv2.rectangle(image,(0,y1b),(160,y1b+16),(25,25,30),-1)
                    if bw>0: cv2.rectangle(image,(0,y1b),(bw,y1b+16),bc,-1)
                    cv2.putText(image,f"{LETTER_SIGNS[i]}:{l_top5_probs[i]:.0%}",
                                (164,y1b+12),cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1,cv2.LINE_AA)
            if l_pred_label and hand_present:
                col=(0,255,160) if l_pred_conf>=THR_LETTER else (0,165,255)
                cv2.putText(image,l_pred_label,(12,298),cv2.FONT_HERSHEY_DUPLEX,
                            2.4,tuple(c//6 for c in col),6)
                cv2.putText(image,l_pred_label,(10,296),cv2.FONT_HERSHEY_DUPLEX,
                            2.4,col,3,cv2.LINE_AA)
                cv2.putText(image,f"{l_pred_conf:.0%}",(90,296),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,col,1)
            if hand_present and l_consec>0:
                need_now=CONSEC_SPACE if l_consec_cls==SPACE_IDX else CONSEC_NEED
                pct=min(l_consec/need_now,1.0); pw=int(fw*pct)
                cv2.rectangle(image,(0,302),(fw,308),(25,25,30),-1)
                cb=(0,200,100) if pct>=1.0 else (0,140,255)
                if pw>0: cv2.rectangle(image,(0,302),(pw,308),cb,-1)
                cv2.putText(image,f"hold {l_consec}/{need_now}",
                            (4,322),cv2.FONT_HERSHEY_SIMPLEX,0.38,(160,160,160),1)
            if l_cooldown>0:
                cv2.rectangle(image,(0,325),(int(fw*l_cooldown/COOLDOWN_L),329),(0,180,255),-1)
            cv2.rectangle(image,(0,0),(fw,52),(12,12,18),-1)
            cv2.line(image,(0,52),(fw,52),(0,255,160),1)
            cv2.putText(image,"MODE 1  LETTER SIGNING",
                        (8,22),cv2.FONT_HERSHEY_SIMPLEX,0.60,(0,255,160),1,cv2.LINE_AA)
            cv2.putText(image,"M:WordMode  S:Space  T:SpeechSign  B:Back  C:Clear  SPACE:Speak  Q:Quit",
                        (8,44),cv2.FONT_HERSHEY_SIMPLEX,0.31,(90,90,90),1,cv2.LINE_AA)

        # ===================================================================
        # MODE 2 -- WORD SIGNING
        # ONE mediapipe call only (word_lm).  Space via S key.
        # Buffer frozen ONCE at first frame of peak -- never overwritten.
        # Sign fires exactly once per motion event.
        # ===================================================================
        elif ui_mode==2:
            listening_flag.clear()
            if w_cooldown>0: w_cooldown-=1
            if not WORD_ACTIVE:
                cv2.rectangle(image,(0,0),(fw,52),(12,12,18),-1)
                cv2.putText(image,"Word model not found -- press M for letter mode",
                            (8,30),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,100,255),1)
            else:
                try:
                    _,res_w,_,flip_w=mediapipe_detection(frame,word_lm)
                except:
                    res_w=type('R',(object,),{'hand_landmarks':[],'handedness':[]})(); flip_w=False
                draw_styled_landmarks(image,res_w,flip_w)
                hand_w=bool(res_w.hand_landmarks)

                if hand_w:
                    kp126=extract_keypoints_two_hands(res_w,flip_w)
                    w_buf.append(kp126)
                    if w_prev_kp is not None:
                        w_delta=float(np.mean(np.abs(kp126-w_prev_kp)))
                    w_prev_kp=kp126.copy()

                    # ── Space gesture (reuse res_w hand - no second mediapipe call) ──
                    if not w_mot_seen and w_delta<STILL_DELTA and w_spc_cooldown==0:
                        try:
                            # Reuse already-detected hand from res_w - NO extra mediapipe call
                            kp63=extract_keypoints(res_w,flip_w)
                            spc_probs=letter_model.predict(kp63.reshape(1,-1),verbose=0)[0]
                            spc_raw=float(spc_probs[SPACE_IDX])
                            w_spc_ema=(EMA_ALPHA*spc_raw+(1-EMA_ALPHA)*w_spc_ema
                                       if w_spc_ema is not None else spc_raw)
                            if w_spc_ema>=THR_SPACE: w_spc_consec+=1
                            else: w_spc_consec=0
                            if w_spc_consec>=CONSEC_SPACE:
                                append_to_sentence('space','SPACE-W')
                                w_spc_consec=0; w_spc_ema=None
                                w_spc_cooldown=COOLDOWN_L
                        except: pass
                    elif w_mot_seen:
                        w_spc_consec=0; w_spc_ema=None
                    if w_spc_cooldown>0: w_spc_cooldown-=1

                    if w_delta>=MOTION_DELTA:
                        w_mot_cnt+=1; w_still_cnt=0
                        if w_mot_cnt>=CONSEC_MOT:
                            w_mot_seen=True
                        # Collect ALL frames once signing is armed
                        if w_mot_seen:
                            w_sign_frames.append(kp126.copy())
                    else:
                        w_mot_cnt=max(w_mot_cnt-1,0)
                        # Keep collecting during brief pauses inside sign
                        if w_mot_seen:
                            w_sign_frames.append(kp126.copy())
                        if w_delta<STILL_DELTA: w_still_cnt+=1
                        else: w_still_cnt=0

                    # Fire once: motion confirmed + hand clearly stopped + have frames
                    if (w_mot_seen and w_still_cnt>=CONSEC_STILL
                            and len(w_sign_frames)>=CONSEC_MOT and w_cooldown==0):
                        # Resample the ENTIRE sign to exactly SEQ_LEN frames.
                        # This matches training: collect_words records 30 frames of
                        # the whole sign from start to end at natural speed.
                        # Resampling handles both fast signs (<30 frames) and
                        # slow signs (>30 frames) correctly.
                        sf = w_sign_frames
                        n  = len(sf)
                        if n >= SEQ_LEN:
                            idx = np.linspace(0, n-1, SEQ_LEN).astype(int)
                            frames_to_eval = [sf[i] for i in idx]
                        else:
                            frames_to_eval = sf[:]
                            while len(frames_to_eval) < SEQ_LEN:
                                frames_to_eval = [frames_to_eval[0]] + frames_to_eval
                        seq=np.array(frames_to_eval, dtype=np.float32)
                        w_probs=WORD_MODEL.predict(np.expand_dims(seq,0),verbose=0)[0]
                        w_best=int(np.argmax(w_probs)); w_conf=float(w_probs[w_best])
                        w_label=str(WORD_LABELS[w_best])
                        print(f"[WORD] {w_label}  conf={w_conf:.0%}  sign_frames={len(w_sign_frames)}")
                        # Full reset -- dead until next sign
                        w_mot_seen=False; w_mot_cnt=0; w_still_cnt=0
                        w_sign_frames=[]; w_prev_kp=None
                        w_pred_label=w_label; w_pred_conf=w_conf; w_top5_probs=w_probs
                        if w_conf>=THR_WORD:
                            append_to_sentence(w_label,"WORD"); w_cooldown=COOLDOWN_W
                        else:
                            print(f"  rejected {w_conf:.0%} < {THR_WORD:.0%}")
                else:
                    reset_word_engine()

                if w_top5_probs is not None and hand_w:
                    for rank,i in enumerate(np.argsort(w_top5_probs)[::-1][:5]):
                        i=int(i); y1b=60+rank*24
                        bc=(0,200,255) if rank==0 else (60,60,180); bw=int(w_top5_probs[i]*160)
                        cv2.rectangle(image,(0,y1b),(160,y1b+16),(25,25,30),-1)
                        if bw>0: cv2.rectangle(image,(0,y1b),(bw,y1b+16),bc,-1)
                        cv2.putText(image,f"{WORD_LABELS[i]}:{w_top5_probs[i]:.0%}",
                                    (164,y1b+12),cv2.FONT_HERSHEY_SIMPLEX,
                                    0.38,(200,200,200),1,cv2.LINE_AA)
                if w_pred_label:
                    col=(0,200,255) if w_pred_conf>=THR_WORD else (0,165,255)
                    cv2.putText(image,w_pred_label,(10,296),cv2.FONT_HERSHEY_DUPLEX,
                                1.6,tuple(c//6 for c in col),5)
                    cv2.putText(image,w_pred_label,(8,294),cv2.FONT_HERSHEY_DUPLEX,
                                1.6,col,2,cv2.LINE_AA)
                    cv2.putText(image,f"{w_pred_conf:.0%}",
                                (10,318),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,1)
                if hand_w:
                    dv=int(min(w_delta*1200,fw)); colm=(0,200,255) if w_mot_seen else (0,255,160)
                    cv2.rectangle(image,(0,302),(fw,308),(25,25,30),-1)
                    if dv>0: cv2.rectangle(image,(0,302),(dv,308),colm,-1)
                    tdv=int(MOTION_DELTA*1200)
                    if tdv<fw: cv2.line(image,(tdv,302),(tdv,308),(100,80,255),1)
                    mpw=int(fw*min(w_mot_cnt/CONSEC_MOT,1.0))
                    cv2.rectangle(image,(0,309),(fw,313),(18,18,25),-1)
                    if mpw>0: cv2.rectangle(image,(0,309),(mpw,313),(0,160,200),-1)
                    st="SIGNING-LOCKED" if w_mot_seen else f"build {w_mot_cnt}/{CONSEC_MOT}"
                    cv2.putText(image,
                                f"d={w_delta:.3f}  [{st}]  still={w_still_cnt}/{CONSEC_STILL}"
                                f"  frozen={'YES' if w_frozen is not None else 'no'}",
                                (4,326),cv2.FONT_HERSHEY_SIMPLEX,0.33,(160,160,160),1)
                if w_cooldown>0:
                    cv2.rectangle(image,(0,328),
                                  (int(fw*w_cooldown/COOLDOWN_W),332),(0,180,255),-1)
                cv2.rectangle(image,(0,0),(fw,52),(12,12,18),-1)
                cv2.line(image,(0,52),(fw,52),(0,200,255),1)
                cv2.putText(image,"MODE 2  WORD SIGNING",
                            (8,22),cv2.FONT_HERSHEY_SIMPLEX,0.60,(0,200,255),1,cv2.LINE_AA)
                cv2.putText(image,"M:Ref  S:Space  B:Back  C:Clear  SPACE:Speak  Q:Quit",
                            (8,44),cv2.FONT_HERSHEY_SIMPLEX,0.31,(90,90,90),1,cv2.LINE_AA)

        # ===================================================================
        # MODE 3 -- VIDEO REFERENCE
        # ===================================================================
        elif ui_mode==3:
            listening_flag.clear()
            if not m3_loaded and 0<=m3_sel<len(m3_all):
                item=m3_all[m3_sel]; m3_vp=VideoPlayer()
                vk=item.lower() if item.lower() in word_videos else item.upper()
                if vk in word_videos:
                    m3_vp.load(word_videos[vk],item,fw-_DW3-20,fh-58-8)
                m3_loaded=True
            if m3_vp.frames and not m3_vp.done: m3_vp.advance()
            draw_m3(image,fw,fh)

        # ===================================================================
        # MODE 4 -- SPEECH TO SIGN
        # ===================================================================
        elif ui_mode==4:
            listening_flag.set()
            if heard_text["value"]:
                raw=heard_text["value"]; heard_text["value"]=""
                m4_text=raw; m4_tokens=tokenise_speech(raw)
                m4_idx=0; m4_timer=0; m4_vp=VideoPlayer(); m4_vp_word=""
                print(f"[M4] '{raw}' -> {len(m4_tokens)} tokens")
            if m4_tokens:
                if m4_idx>=len(m4_tokens):
                    # Loop: restart from beginning
                    m4_idx=0; m4_timer=0; m4_vp=VideoPlayer(); m4_vp_word=""
                tok=m4_tokens[m4_idx]
                if tok['type']=='space':
                    m4_timer+=1
                    if m4_timer>=14: m4_timer=0; m4_idx+=1
                elif tok['type']=='letter':
                    m4_timer+=1
                    if m4_timer>=LETTER_DUR: m4_timer=0; m4_idx+=1
                elif tok['type']=='word':
                    wv=tok['val']
                    if wv in word_videos:
                        if m4_vp_word!=wv:
                            m4_vp=VideoPlayer()
                            m4_vp.load(word_videos[wv],wv,fw-16,fh-150)
                            m4_vp_word=wv
                        if m4_vp.advance():
                            m4_idx+=1; m4_vp_word=""
                    else:
                        m4_timer+=1
                        if m4_timer>=LETTER_DUR: m4_timer=0; m4_idx+=1
            draw_m4(image,m4_tokens,m4_idx,m4_vp,fw,fh,tick)
            pulse=int(abs(np.sin(tick*3))*7)+5
            cv2.circle(image,(fw-28,fh-28),pulse+8,(25,25,30),-1)
            cv2.circle(image,(fw-28,fh-28),pulse+8,(180,80,255),1)
            cv2.circle(image,(fw-28,fh-28),5,(180,80,255),-1)

        # ===================================================================
        # SHARED SENTENCE BAR  (modes 1 & 2)
        # ===================================================================
        if ui_mode in (1,2):
            sent=''.join(sentence)
            disp=sent[-50:] if len(sent)>50 else sent
            blink="_" if int(tick*2)%2==0 else " "
            mc=(0,255,160) if ui_mode==1 else (0,200,255)
            badge="L" if ui_mode==1 else "W"
            cv2.rectangle(image,(0,fh-56),(fw,fh),(12,12,18),-1)
            cv2.line(image,(0,fh-56),(fw,fh-56),mc,1)
            cv2.rectangle(image,(4,fh-50),(22,fh-32),mc,-1)
            cv2.putText(image,badge,(7,fh-35),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,0,0),1,cv2.LINE_AA)
            if disp:
                cv2.putText(image,disp+blink,(28,fh-20),cv2.FONT_HERSHEY_DUPLEX,0.72,(0,50,25),2)
                cv2.putText(image,disp+blink,(27,fh-21),cv2.FONT_HERSHEY_DUPLEX,0.72,mc,1,cv2.LINE_AA)
            else:
                hint="hold a letter..." if ui_mode==1 else "sign a word with motion..."
                cv2.putText(image,hint+blink,(28,fh-21),cv2.FONT_HERSHEY_DUPLEX,0.50,(45,45,45),1,cv2.LINE_AA)
            if speak_flash>0:
                cv2.rectangle(image,(0,fh-80),(fw,fh-58),(0,60,0),-1)
                cv2.putText(image,f"SPEAKING: {sent.strip()[:52]}",
                            (6,fh-63),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1)
            wc=len(sent.strip().split()) if sent.strip() else 0
            cv2.putText(image,f"w:{wc}",(fw-50,fh-8),cv2.FONT_HERSHEY_SIMPLEX,0.35,(70,70,70),1)

        # ===================================================================
        # INPUT
        # ===================================================================
        cv2.imshow('Sign Language Translator',image)
        key=cv2.waitKey(10)&0xFF

        if key==ord('q'): break
        elif key==ord('t'):
            # Type text for speech-to-sign -- works in any mode, switches to mode 4
            cv2.destroyAllWindows()
            print("\n" + "="*50)
            print("  TYPE SENTENCE FOR SPEECH-TO-SIGN")
            print("  Press Enter to confirm, blank to cancel")
            print("="*50)
            raw = input("  > ").strip()
            if raw:
                heard_text["value"] = raw
                ui_mode = 4
                listening_flag.set()
                print(f"[TYPE] {raw!r}")
            cv2.namedWindow('Sign Language Translator')
            cv2.setMouseCallback('Sign Language Translator',_mouse_cb)
        elif key==ord('m'):
            ui_mode=(ui_mode%4)+1
            reset_letter_engine(); l_cooldown=0
            reset_word_engine();   w_cooldown=0
            m3_loaded=False
            if ui_mode!=4: listening_flag.clear()
            print(f"\n-- MODE {ui_mode} --  sentence: {''.join(sentence)!r}")
        elif key==ord('c'):
            sentence.clear()
            reset_letter_engine(); l_cooldown=0
            reset_word_engine();   w_cooldown=0
            m4_tokens=[]; m4_idx=0; m4_timer=0; m4_text=""
            m4_vp=VideoPlayer(); m4_vp_word=""; heard_text["value"]=""
            print("Cleared.")
        elif key==ord('b'):
            if sentence:
                removed=sentence.pop()
                print(f"Backspace: {removed!r}")
        elif key==ord('s') and ui_mode in (1,2):
            if sentence and sentence[-1]!=' ':
                sentence.append(' ')
                print(f"+ space [S]")
        elif key==32:
            if ui_mode==3: m3_vp.reset()
            elif ui_mode in (1,2):
                text=''.join(sentence).strip()
                if text: speak_windows(text); speak_flash=90
                else: print("Nothing to speak.")
        elif key in (82,119) and ui_mode==3:
            if m3_sel>0: m3_sel-=1; m3_loaded=False
            if m3_sel<m3_scroll: m3_scroll=m3_sel
        elif key in (84,115) and ui_mode==3:
            if m3_sel<len(m3_all)-1: m3_sel+=1; m3_loaded=False
            if m3_sel>=m3_scroll+_VI3: m3_scroll=m3_sel-_VI3+1

listening_flag.clear()
cap.release()
cv2.destroyAllWindows()
print("Closed.")