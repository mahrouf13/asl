import subprocess
import sys

packages = [
    "numpy",
    "opencv-python-headless",
    "mediapipe",
    "tensorflow-cpu",
    "gTTS",
    "streamlit",
    "streamlit-webrtc",
    "av",
    "aiortc"
]

for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        
import os, sys, threading, time, queue, datetime, logging, io, base64
import cv2, numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import urllib.request

logging.getLogger("tornado.websocket").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

st.set_page_config(page_title="ASL Translator", page_icon="🤟",
                   layout="wide", initial_sidebar_state="collapsed")

@st.cache_resource
def _get_queue():
    return queue.SimpleQueue()

@st.cache_resource(show_spinner="⏳ Preparing hand detection model…")
def _ensure_hand_landmarker():
    task_path = os.path.join(ROOT, 'models', 'hand_landmarker.task')
    if not os.path.exists(task_path):
        os.makedirs(os.path.dirname(task_path), exist_ok=True)
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        try:
            urllib.request.urlretrieve(url, task_path)
        except Exception as e:
            st.error(f"Failed to download hand landmarker model: {e}")
            st.stop()
    return task_path

_ensure_hand_landmarker()

from function import (
    LETTER_SIGNS        as _LS,
    HAND_CONNECTIONS    as _HC,
    create_landmarker   as _mk_lm1,
    mediapipe_detection as _mp_detect,
    extract_keypoints   as _ekp1,
)

FPS_CAP       = 12
PROC_W,PROC_H = 320, 240
SKIP_N        = 2
EMA_ALPHA     = 0.65
CONSEC_NEED   = 12;  THR_LETTER = 0.72
FLASH_FRAMES  = 8
SPACE_IDX     = len(_LS) - 1
THR_SPACE     = 0.82; CONSEC_SPACE = 14

RTC = RTCConfiguration({"iceServers":[
    {"urls":["stun:stun.l.google.com:19302"]},
    {"urls":["stun:stun1.l.google.com:19302"]},
    {"urls":["stun:stun2.l.google.com:19302"]},
]})

def asset(n):
    for d in ['assets','scripts','.']:
        p = os.path.join(ROOT,d,n)
        if os.path.exists(p): return p
    return None

@st.cache_data
def get_letter_img(letter):
    folder = os.path.join(ROOT,'data',letter.upper())
    if not os.path.isdir(folder): return None
    files = sorted([f for f in os.listdir(folder)
                    if not f.startswith('aug_') and f.lower().endswith(('.png','.jpg'))])
    return os.path.join(folder,files[0]) if files else None

@st.cache_resource(show_spinner="⏳ Loading model…")
def load_models():
    from keras.models import load_model
    model_path = os.path.join(ROOT,'models','model.h5')
    if not os.path.exists(model_path):
        st.error(f"Letter model not found at: {model_path}")
        st.stop()
    return dict(lm=load_model(model_path))

_M = load_models()

def _res_to_pts(res,pw,ph,sx,sy,flip):
    out=[]
    if not res.hand_landmarks: return out
    for hand in res.hand_landmarks:
        pts=[]
        for lm in hand:
            x=pw-lm.x*pw if flip else lm.x*pw
            pts.append((int(x*sx),int(lm.y*ph*sy)))
        out.append(pts)
    return out

_OVL_LETTER_SCALE  = 0.90
_OVL_CONF_SCALE    = 0.38
_OVL_BAR_SCALE     = 0.34
_OVL_LM_RADIUS     = 3
_OVL_BAR_W         = 130
_OVL_BAR_H         = 13
_OVL_BAR_GAP       = 18

class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.lm=_M['lm']; self.last_pred=""; self.last_conf=0.0
        self._in_q=queue.Queue(maxsize=1); self._running=True
        threading.Thread(target=self._worker,daemon=True).start()
        self._lock=threading.Lock()
        self._pts=[]; self._ovl=[]; self._bar_pct=0.0; self._bar_col=(0,100,180)
        self._fc=0

    def __del__(self): self._running=False

    def _worker(self):
        lm1=_mk_lm1(); Q=_get_queue()
        l_ema=None; l_c=0; l_cls=-1
        l_se=None;  l_sc=0
        l_flash=0; l_flash_lbl=""
        last_pts=[]; no_hand_frames=0; NO_HAND_CLEAR=8
        _dt=1.0/FPS_CAP; _t=0.0

        while self._running:
            now=time.time(); gap=_dt-(now-_t)
            if gap>0: time.sleep(gap)
            try: full,cw,ch=self._in_q.get(timeout=0.05)
            except queue.Empty: continue
            _t=time.time()

            small=cv2.resize(full,(PROC_W,PROC_H))
            sx=cw/PROC_W; sy=ch/PROC_H; h=ch

            try:
                _,res,_,flip=_mp_detect(small,lm1)
                hand=bool(res.hand_landmarks)

                if hand:
                    last_pts=_res_to_pts(res,PROC_W,PROC_H,sx,sy,flip)
                    no_hand_frames=0
                else:
                    no_hand_frames+=1
                    if no_hand_frames>=NO_HAND_CLEAR:
                        last_pts=[]

                ovl=[]; pct=0.0; bc=(0,100,180)

                if l_flash>0:
                    l_flash-=1
                    pct=1.0; bc=(0,220,100)
                    ovl=[(f"✓ {l_flash_lbl}",10,h-14,
                          _OVL_LETTER_SCALE,(0,255,160),2,True)]
                    if l_flash==0:
                        l_ema=None;l_c=0;l_cls=-1;l_se=None;l_sc=0

                elif hand:
                    kp=_ekp1(res,flip)
                    probs=self.lm(kp.reshape(1,-1),training=False).numpy()[0]

                    if l_ema is None: l_ema=probs.copy()
                    else: l_ema=EMA_ALPHA*l_ema+(1-EMA_ALPHA)*probs

                    cls=int(np.argmax(l_ema)); conf=float(l_ema[cls])
                    if cls==l_cls: l_c+=1
                    else: l_cls=cls; l_c=1

                    self.last_pred=str(_LS[cls]); self.last_conf=conf

                    spc=float(probs[SPACE_IDX])
                    if l_se is None: l_se=spc
                    else: l_se=EMA_ALPHA*spc+(1-EMA_ALPHA)*l_se
                    if l_se>=THR_SPACE: l_sc+=1
                    else: l_sc=0

                    fired=False
                    if l_sc>=CONSEC_SPACE:
                        Q.put('space')
                        l_flash=FLASH_FRAMES; l_flash_lbl="SPC"; fired=True
                    elif l_c>=CONSEC_NEED and conf>=THR_LETTER and cls!=SPACE_IDX:
                        lbl=str(_LS[cls]); Q.put(lbl)
                        l_flash=FLASH_FRAMES; l_flash_lbl=lbl; fired=True

                    if not fired:
                        is_spc=l_se is not None and l_se>=THR_SPACE
                        hold=l_sc if is_spc else l_c
                        need=CONSEC_SPACE if is_spc else CONSEC_NEED
                        pct=min(hold/max(need,1),1.0)
                        bc=(0,220,100) if pct>=1.0 else (0,160,255)
                        tc=(0,255,160) if conf>=THR_LETTER else (0,165,255)
                        ovl=[
                            (str(_LS[cls]),10,h-14,_OVL_LETTER_SCALE,tc,2,True),
                            (f"{conf:.0%}  {hold}/{need}",10,h-34,_OVL_CONF_SCALE,tc,1,False),
                        ]
                        top5=l_ema.copy()
                        for rank,i in enumerate(np.argsort(top5)[::-1][:5]):
                            i=int(i)
                            bc2=(0,255,160) if rank==0 else (60,60,160)
                            bw=int(top5[i]*_OVL_BAR_W)
                            y0=44+rank*_OVL_BAR_GAP
                            ovl.append(('__bar__',0,y0,bw,_OVL_BAR_H,(20,20,28),bc2))
                            ovl.append((f"{_LS[i]}:{top5[i]:.0%}",
                                        _OVL_BAR_W+4,y0+_OVL_BAR_H-1,
                                        _OVL_BAR_SCALE,(190,190,190),1,False))
                    else:
                        pct=1.0; bc=(0,220,100)
                        ovl=[(f"✓ {l_flash_lbl}",10,h-14,
                              _OVL_LETTER_SCALE,(0,255,160),2,True)]

                else:
                    l_ema=None;l_c=0;l_cls=-1;l_se=None;l_sc=0

                with self._lock:
                    self._pts=last_pts
                    self._bar_pct=pct; self._bar_col=bc; self._ovl=ovl

            except Exception as e:
                with self._lock:
                    self._ovl=[('ERR:'+str(e)[:55],8,30,0.36,(0,0,200),1,False)]

    def recv(self,frame:av.VideoFrame)->av.VideoFrame:
        img=frame.to_ndarray(format="bgr24")
        img=cv2.flip(img,1)
        h,w=img.shape[:2]
        self._fc+=1

        if self._fc%SKIP_N==0:
            try: self._in_q.get_nowait()
            except: pass
            try: self._in_q.put_nowait((img.copy(),w,h))
            except: pass

        with self._lock:
            pts=list(self._pts)
            bar_pct=self._bar_pct
            bar_col=self._bar_col
            ovl=list(self._ovl)

        COLS=[(0,255,160),(0,165,255)]
        for hi,hp in enumerate(pts):
            c=COLS[hi%2]
            for s,e in _HC:
                if s<len(hp) and e<len(hp):
                    cv2.line(img,hp[s],hp[e],(70,70,70),1)
            for px,py in hp:
                cv2.circle(img,(px,py),_OVL_LM_RADIUS,c,-1)

        bh=8
        cv2.rectangle(img,(0,h-bh),(w,h),(12,12,18),-1)
        if bar_pct>0:
            cv2.rectangle(img,(0,h-bh),(int(w*bar_pct),h),bar_col,-1)

        for item in ovl:
            if item[0]=='__bar__':
                _,x0,y0,bw,bh2,bg,bc2=item
                cv2.rectangle(img,(x0,y0),(x0+_OVL_BAR_W,y0+bh2),bg,-1)
                if bw>0: cv2.rectangle(img,(x0,y0),(x0+bw,y0+bh2),bc2,-1)
                continue
            txt,x,y,sc,col,th,shadow=item
            font=cv2.FONT_HERSHEY_DUPLEX if shadow else cv2.FONT_HERSHEY_SIMPLEX
            if shadow:
                cv2.putText(img,txt,(x,y),font,sc,(0,30,10),th*2+1)
            cv2.putText(img,txt,(x,y),font,sc,col,th,cv2.LINE_AA)

        cv2.rectangle(img,(0,0),(130,24),(8,8,14),-1)
        cv2.putText(img,"SIGN→TEXT",(6,17),
                    cv2.FONT_HERSHEY_SIMPLEX,0.40,(0,255,160),1,cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img,format="bgr24")

# ── helpers ───────────────────────────────────────────────────────────────────
def speak(text):
    s=text.strip().replace('"','').replace("'",'').replace('`','').replace('\n',' ')
    if not s: return
    try:
        from gtts import gTTS
        tts=gTTS(text=s,lang='en',slow=False)
        fp=io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64=base64.b64encode(fp.read()).decode()
        st.markdown(
            f'<audio autoplay style="display:none">'
            f'<source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">'
            f'</audio>',unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"🔊 TTS error: {e}")

def push_history(text,source):
    if not text.strip(): return
    ts=datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.history.insert(0,{"ts":ts,"src":source,"text":text.strip()})
    if len(st.session_state.history)>50:
        st.session_state.history=st.session_state.history[:50]

def drain():
    q=_get_queue()
    sent=st.session_state.sentence
    while not q.empty():
        try: tok=q.get_nowait()
        except: break
        if tok=='space':
            if sent and sent[-1]!=' ': sent.append(' ')
        else:
            sent.append(tok)
        if len(sent)>300: del sent[:-300]

# ── session state ─────────────────────────────────────────────────────────────
for k,v in dict(top_mode="sign",sentence=[],history=[],_sp_last="",dark_mode=True).items():
    if k not in st.session_state: st.session_state[k]=v
if not isinstance(st.session_state.sentence,list): st.session_state.sentence=[]
if not isinstance(st.session_state.history,list): st.session_state.history=[]

if st.session_state.top_mode=="sign": drain()

# ── theme ─────────────────────────────────────────────────────────────────────
DK=st.session_state.dark_mode
if DK:
    BG="#06060e";BG2="#0a0a16";BG3="#0e0e1c";BORDER="#1e1e32";BORDER2="#14142a"
    TXT="#c8d8e8";TXT2="#8090a8";TXT3="#506070";ACCENT="#00ff9d";ACCENT2="#3d7fff"
    SENT_BG="#020d05";SENT_TXT="#00f090";SENT_PH="#1a3020"
    HIST_BG="#080814";HIST_TXT="#00cc80";HIST_TS="#4a6070"
    HIST_SIGN="#00aa60";HIST_SPK="#4488cc";HIST_EMPTY="#384858"
    LBL_COL="#6a8898";TIP_BG="#080812"
    NAV_BG="linear-gradient(90deg,#040c06,#040608)"
    STAT_BG="#090918";STAT_TXT="#4a6070";STAT_VAL="#2299cc"
    SPC_BG="#070718";SPC_BORDER="#18184a";SPC_TXT="#6070a8"
    SPACE_DIV_BG="#0e0e20";SPACE_DIV_BD="#202040";SPACE_DIV_TXT="#7080b8"
    WORD_BG="#07070f";WORD_BD="#12121e";WORD_HDR="#6080a0"
    HEARD_BG="#05101a";HEARD_BD="#3d7fff30";HEARD_TXT="#5090ff"
    REF_BG="#050510";REF_BD="#101020"
    BTN_SEC_BG="#08081a";BTN_SEC_BD="#181832";BTN_SEC_TXT="#607090"
    BTN_PRI_BG="#021008";BTN_PRI_BD="#00ff9daa";BTN_PRI_TXT="#00ff9d"
    INPUT_BG="#05051a";INPUT_BD="#13132a";INPUT_TXT="#90a8c0"
    HERO_BG="linear-gradient(135deg,#06101c,#060610)";HERO_BD="#3d7fff1a"
    HERO_TXT="#3d7fff";HERO_SUB="#3a4858"
else:
    BG="#f4f6f9";BG2="#eaecf2";BG3="#dde0ea";BORDER="#c8ccd8";BORDER2="#b8bcc8"
    TXT="#1a2030";TXT2="#40505e";TXT3="#607080";ACCENT="#008855";ACCENT2="#1a5fbf"
    SENT_BG="#edfff5";SENT_TXT="#007740";SENT_PH="#90b8a0"
    HIST_BG="#ffffff";HIST_TXT="#006640";HIST_TS="#708090"
    HIST_SIGN="#008844";HIST_SPK="#1a5fbf";HIST_EMPTY="#809090"
    LBL_COL="#506878";TIP_BG="#f0f2f8"
    NAV_BG="linear-gradient(90deg,#edf5f0,#edf0f8)"
    STAT_BG="#e8eaf2";STAT_TXT="#506070";STAT_VAL="#1a6aaa"
    SPC_BG="#eef0fa";SPC_BORDER="#b8bce8";SPC_TXT="#506090"
    SPACE_DIV_BG="#e8eaf8";SPACE_DIV_BD="#b0b4d8";SPACE_DIV_TXT="#4060a0"
    WORD_BG="#f8f9fc";WORD_BD="#d0d4e0";WORD_HDR="#506080"
    HEARD_BG="#eef4ff";HEARD_BD="#3d7fff50";HEARD_TXT="#1a50cc"
    REF_BG="#ffffff";REF_BD="#d0d4e0"
    BTN_SEC_BG="#eaecf4";BTN_SEC_BD="#b8bcd0";BTN_SEC_TXT="#405060"
    BTN_PRI_BG="#e8fff4";BTN_PRI_BD="#008855aa";BTN_PRI_TXT="#007744"
    INPUT_BG="#ffffff";INPUT_BD="#c0c4d0";INPUT_TXT="#303848"
    HERO_BG="linear-gradient(135deg,#eef4ff,#f0eeff)";HERO_BD="#3d7fff44"
    HERO_TXT="#1a50cc";HERO_SUB="#4a5868"

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='font-size:0.85rem;font-weight:600;color:{TXT2};margin-bottom:10px;'>🎨 Theme</div>",unsafe_allow_html=True)
    if st.button("☀️ Light mode" if DK else "🌙 Dark mode",key="theme_toggle",use_container_width=True):
        st.session_state.dark_mode=not DK; st.rerun()
    st.markdown("<hr style='border-color:"+BORDER2+";margin:14px 0;'>",unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.72rem;color:{TXT3};line-height:1.7;'>ASL Sign Language Translator<br>Letter model · sentence builder<br>Speech synthesis · history</div>",unsafe_allow_html=True)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
html,body,*{{font-family:'Inter',sans-serif;box-sizing:border-box;}}
.stApp,.main,.block-container{{background:{BG} !important;color:{TXT} !important;}}
.block-container{{padding:0 1rem 2rem !important;max-width:100% !important;}}
#MainMenu,footer,header{{visibility:hidden !important;}}
div[data-testid="stWebRtcStreamer"] button,div[data-testid="stWebRtcStreamer"] div[role="button"],
button[title="Stop"],button[title="Pause"],button[title="Settings"]{{
  display:none !important;visibility:hidden !important;opacity:0 !important;
  pointer-events:none !important;width:0 !important;height:0 !important;}}
.nav{{display:flex;align-items:center;gap:16px;padding:11px 24px;
  background:{NAV_BG};border-bottom:2px solid {BORDER};margin:0 -1rem 1.2rem -1rem;}}
.nav-brand{{font-weight:700;font-size:1.2rem;color:{ACCENT};letter-spacing:-0.5px;}}
.nav-brand span{{color:{ACCENT2};}}
.nav-sub{{font-size:0.70rem;color:{TXT3};margin-left:4px;}}
div[data-testid="stButton"]>button{{font-family:'JetBrains Mono',monospace !important;
  font-size:0.73rem !important;font-weight:600 !important;border-radius:7px !important;
  padding:8px 0 !important;width:100% !important;transition:all .15s !important;}}
div[data-testid="stButton"]>button[kind="secondary"]{{background:{BTN_SEC_BG} !important;
  border:1px solid {BTN_SEC_BD} !important;color:{BTN_SEC_TXT} !important;}}
div[data-testid="stButton"]>button[kind="secondary"]:hover{{border-color:{ACCENT}55 !important;color:{ACCENT} !important;}}
div[data-testid="stButton"]>button[kind="primary"]{{background:{BTN_PRI_BG} !important;
  border:1.5px solid {BTN_PRI_BD} !important;color:{BTN_PRI_TXT} !important;}}
.lbl{{font-family:'JetBrains Mono',monospace;font-size:0.65rem;font-weight:700;
  letter-spacing:2px;text-transform:uppercase;color:{LBL_COL};
  margin:16px 0 7px;border-left:3px solid {ACCENT};padding-left:8px;}}
.sent-box{{background:{SENT_BG};border:1.5px solid {ACCENT}44;border-radius:10px;
  padding:14px 18px;min-height:64px;font-family:'JetBrains Mono',monospace;
  font-size:1.15rem;color:{SENT_TXT};line-height:1.75;word-break:break-all;}}
.sent-ph{{color:{SENT_PH} !important;font-style:italic;font-size:0.95rem !important;}}
.stats{{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px;}}
.stat{{background:{STAT_BG};border:1px solid {BORDER2};border-radius:5px;
  font-family:'JetBrains Mono',monospace;font-size:0.63rem;padding:3px 10px;color:{STAT_TXT};}}
.stat b{{color:{STAT_VAL};}}
.spc-tip{{display:flex;align-items:center;gap:12px;background:{SPC_BG};
  border:1.5px solid {SPC_BORDER};border-radius:9px;padding:10px 15px;margin:8px 0;
  font-size:0.80rem;color:{SPC_TXT};line-height:1.55;}}
.spc-tip strong{{color:{TXT2};}}
.tips-card{{background:{TIP_BG};border:1px solid {BORDER};border-radius:10px;padding:12px 14px;margin-bottom:10px;}}
.tip-row{{display:flex;gap:10px;padding:5px 0;border-bottom:1px solid {BORDER2};font-size:0.76rem;color:{TXT2};line-height:1.5;}}
.tip-row:last-child{{border:none;padding-bottom:0;}}
.tip-ico{{font-size:0.95rem;flex-shrink:0;margin-top:1px;}}
.tip-key{{color:{ACCENT};font-weight:700;font-family:'JetBrains Mono',monospace;font-size:0.72rem;}}
.hist-item{{background:{HIST_BG};border:1px solid {BORDER};border-radius:8px;
  padding:9px 12px;margin:5px 0;display:flex;align-items:flex-start;gap:10px;}}
.hist-meta{{display:flex;flex-direction:column;align-items:center;gap:3px;min-width:52px;}}
.hist-ts{{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:{HIST_TS};white-space:nowrap;}}
.hist-sign-badge{{font-family:'JetBrains Mono',monospace;font-size:0.50rem;
  background:{HIST_SIGN}22;border:1px solid {HIST_SIGN}44;border-radius:3px;
  padding:1px 5px;color:{HIST_SIGN};text-transform:uppercase;letter-spacing:1px;}}
.hist-speech-badge{{font-family:'JetBrains Mono',monospace;font-size:0.50rem;
  background:{HIST_SPK}22;border:1px solid {HIST_SPK}44;border-radius:3px;
  padding:1px 5px;color:{HIST_SPK};text-transform:uppercase;letter-spacing:1px;}}
.hist-txt{{font-family:'JetBrains Mono',monospace;font-size:0.85rem;
  color:{HIST_TXT};line-height:1.55;word-break:break-all;font-weight:500;}}
.hist-empty{{text-align:center;color:{HIST_EMPTY};font-size:0.75rem;padding:22px 0;font-style:italic;}}
.sp-hero{{background:{HERO_BG};border:1.5px solid {HERO_BD};border-radius:11px;
  padding:18px;text-align:center;margin-bottom:14px;}}
.sp-hero h3{{font-size:1.1rem;font-weight:700;color:{HERO_TXT};margin:0 0 4px;}}
.sp-hero p{{font-size:0.78rem;color:{HERO_SUB};margin:0;}}
.ws{{background:{WORD_BG};border:1.5px solid {WORD_BD};border-radius:10px;padding:12px 16px;margin:8px 0;}}
.wh{{font-family:'JetBrains Mono',monospace;font-size:0.68rem;font-weight:700;
  letter-spacing:2px;text-transform:uppercase;color:{WORD_HDR};margin-bottom:10px;}}
.space-div{{display:inline-flex;align-items:center;gap:8px;background:{SPACE_DIV_BG};
  border:1.5px dashed {SPACE_DIV_BD};border-radius:8px;padding:5px 14px;margin:4px 0;
  font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{SPACE_DIV_TXT};font-weight:600;}}
.heard-box{{background:{HEARD_BG};border:1.5px solid {HEARD_BD};border-radius:8px;
  padding:11px 16px;margin:8px 0;font-family:'JetBrains Mono',monospace;
  font-size:0.97rem;color:{HEARD_TXT};letter-spacing:0.5px;}}
.ref-card{{background:{REF_BG};border:1px solid {REF_BD};border-radius:9px;padding:10px 12px;margin-bottom:8px;}}
.ref-card-txt{{font-size:0.71rem;color:{TXT2};line-height:1.65;}}
.ref-card-lbl{{font-family:'JetBrains Mono',monospace;font-size:0.50rem;
  letter-spacing:2px;text-transform:uppercase;color:{TXT3};margin-bottom:5px;}}
div[data-testid="stTextArea"] label,div[data-testid="stSelectbox"] label{{display:none !important;}}
.stTextArea textarea{{background:{INPUT_BG} !important;border:1px solid {INPUT_BD} !important;
  color:{INPUT_TXT} !important;font-family:'JetBrains Mono',monospace !important;
  font-size:0.83rem !important;border-radius:8px !important;}}
section[data-testid="stSidebar"]{{background:{BG2} !important;border-right:1px solid {BORDER} !important;}}
</style>""",unsafe_allow_html=True)

# ── NAV ───────────────────────────────────────────────────────────────────────
st.markdown(f"""<div class='nav'>
  <div class='nav-brand'>🤟 ASL<span>Translate</span></div>
  <div class='nav-sub'>Sign alphabet · sentence builder · speech synthesis</div>
</div>""",unsafe_allow_html=True)

_,nb1,nb2,_=st.columns([1,2,2,1],gap="small")
with nb1:
    if st.button("✋  Sign → Text",key="nav_sign",use_container_width=True,
                 type="primary" if st.session_state.top_mode=="sign" else "secondary"):
        st.session_state.top_mode="sign"; st.rerun()
with nb2:
    if st.button("🔊  Speech → Sign",key="nav_speech",use_container_width=True,
                 type="primary" if st.session_state.top_mode=="speech" else "secondary"):
        st.session_state.top_mode="speech"; st.rerun()

st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
left_col,cam_col,right_col=st.columns([1,2.4,1.15],gap="medium")

# ── LEFT ──────────────────────────────────────────────────────────────────────
with left_col:
    st.markdown("<div class='lbl'>📖 ASL Alphabet</div>",unsafe_allow_html=True)
    pa=asset('ASL_alphabet.jpg')
    if pa: st.image(pa,width="stretch")
    else:
        st.markdown(f"<div class='ref-card'><div class='ref-card-lbl'>Reference image</div>"
                    f"<div class='ref-card-txt'>Add<br><code>assets/ASL_alphabet.jpg</code></div></div>",unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>",unsafe_allow_html=True)
    st.markdown("<div class='lbl'>🖐 Space gesture</div>",unsafe_allow_html=True)
    ps=asset('space.jpg')
    if ps: st.image(ps,width="stretch")
    else:
        st.markdown(f"<div class='ref-card'><div class='ref-card-lbl'>Space gesture</div>"
                    f"<div class='ref-card-txt'>Add<br><code>assets/space.jpg</code></div></div>",unsafe_allow_html=True)
    st.markdown(f"<div class='ref-card' style='margin-top:7px;'>"
                f"<div class='ref-card-lbl'>How to sign space</div>"
                f"<div class='ref-card-txt'>Open palm · all 5 fingers spread<br>"
                f"Hold still ~1.5 sec to insert space</div></div>",unsafe_allow_html=True)

# ── RIGHT ─────────────────────────────────────────────────────────────────────
with right_col:
    st.markdown("<div class='lbl'>📋 Instructions</div>",unsafe_allow_html=True)
    st.markdown(f"""<div class='tips-card'>
      <div class='tip-row'><span class='tip-ico'>💡</span><div><span class='tip-key'>Light</span> — Bright, even, face a window</div></div>
      <div class='tip-row'><span class='tip-ico'>🎯</span><div><span class='tip-key'>Distance</span> — 30–60 cm, full hand visible</div></div>
      <div class='tip-row'><span class='tip-ico'>🖼</span><div><span class='tip-key'>Background</span> — Plain wall works best</div></div>
      <div class='tip-row'><span class='tip-ico'>✋</span><div><span class='tip-key'>Hold still</span> — Each letter ~1.5 sec</div></div>
      <div class='tip-row'><span class='tip-ico'>🖐</span><div><span class='tip-key'>Space</span> — Open palm, hold 1.5 sec</div></div>
      <div class='tip-row'><span class='tip-ico'>⚡</span><div><span class='tip-key'>Confidence</span> — ≥ 72% to record</div></div>
      <div class='tip-row'><span class='tip-ico'>🗑</span><div><span class='tip-key'>Clear</span> — Auto-saves to history below</div></div>
    </div>""",unsafe_allow_html=True)
    st.markdown("<div class='lbl'>📜 History</div>",unsafe_allow_html=True)
    hist=st.session_state.history
    if not hist:
        st.markdown(f"<div class='hist-empty'>No history yet — clear a sentence<br>to save it here.</div>",unsafe_allow_html=True)
    else:
        for entry in hist[:8]:
            src=entry.get('src','sign')
            badge_cls="hist-speech-badge" if src=="speech" else "hist-sign-badge"
            st.markdown(
                f"<div class='hist-item'><div class='hist-meta'>"
                f"<div class='hist-ts'>{entry['ts']}</div>"
                f"<div class='{badge_cls}'>{src}</div></div>"
                f"<div class='hist-txt'>{entry['text']}</div></div>",unsafe_allow_html=True)
        if len(hist)>8:
            st.markdown(f"<div style='font-size:0.63rem;color:{TXT3};text-align:center;padding:4px;'>+ {len(hist)-8} more</div>",unsafe_allow_html=True)
        st.markdown("<div style='height:5px'></div>",unsafe_allow_html=True)
        hc1,hc2=st.columns(2,gap="small")
        with hc1:
            if st.button("🗑 Clear history",key="clr_hist",use_container_width=True):
                st.session_state.history=[]; st.rerun()
        with hc2:
            if st.button("🔊 Speak latest",key="spk_hist",use_container_width=True):
                if hist: speak(hist[0]['text'])

# ── CENTRE ────────────────────────────────────────────────────────────────────
with cam_col:
    if st.session_state.top_mode=="sign":
        ctx=webrtc_streamer(
            key="asl_cam",
            video_processor_factory=SignProcessor,
            rtc_configuration=RTC,
            media_stream_constraints={
                "video":{"width":{"min":640,"ideal":1280,"max":1920},
                         "height":{"min":480,"ideal":720,"max":1080},
                         "frameRate":{"min":15,"ideal":30,"max":60},
                         "facingMode":"user"},
                "audio":False,
            },
            async_processing=True,
        )
        st.markdown(f"""<div class='spc-tip'>
          <span style='font-size:1.4rem;'>🖐</span>
          <div><strong>SPACE gesture:</strong> open palm · all 5 fingers spread · hold still ~1.5 sec</div>
        </div>""",unsafe_allow_html=True)
        sent="".join(st.session_state.sentence)
        if sent:
            st.markdown(f"<div class='sent-box'>{sent}</div>",unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='sent-box sent-ph'>Hold a letter sign in front of the camera to begin…</div>",unsafe_allow_html=True)
        c1,c2,c3,c4=st.columns(4,gap="small")
        with c1:
            if st.button("🗑 Clear",key="sign_clear",use_container_width=True):
                push_history(sent,"sign"); st.session_state.sentence.clear(); st.rerun()
        with c2:
            if st.button("⌫ Back",key="sign_back",use_container_width=True):
                if isinstance(st.session_state.sentence,list) and st.session_state.sentence:
                    del st.session_state.sentence[-1]; st.rerun()
        with c3:
            if st.button("␣ Space",key="sign_space",use_container_width=True):
                if not st.session_state.sentence or st.session_state.sentence[-1]!=' ':
                    st.session_state.sentence.append(' '); st.rerun()
        with c4:
            if st.button("🔊 Speak",key="sign_speak",use_container_width=True):
                if sent.strip(): speak(sent)
        lp=ctx.video_processor.last_pred if ctx.video_processor else ""
        lc=ctx.video_processor.last_conf if ctx.video_processor else 0.0
        st.markdown(
            f"<div class='stats'>"
            f"<div class='stat'>Seeing: <b>{lp or '—'}</b></div>"
            f"<div class='stat'>Conf: <b>{lc:.0%}</b></div>"
            f"<div class='stat'>Letters: <b>{len([c for c in sent if c.strip()])}</b></div>"
            f"<div class='stat'>Saved: <b>{len(st.session_state.history)}</b></div>"
            f"</div>",unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
        if st.button("🔄 Refresh sentence",key="manual_refresh",use_container_width=True):
            drain(); st.rerun()
    else:
        st.markdown(f"""<div class='sp-hero'>
          <h3>🎙 Speech → Sign Letters</h3>
          <p>Type a sentence — each letter shown as its ASL hand sign</p>
        </div>""",unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.75rem;color:{TXT3};margin-bottom:4px;'>"
                    f"ℹ️ <em>Type your sentence below to see each letter as an ASL sign image.</em></div>",
                    unsafe_allow_html=True)
        heard=""
        typed=st.text_area("Type a sentence:",key="sp_typed",
                           placeholder="e.g.  hello how are you",height=56)
        if typed.strip(): heard=typed.strip(); st.session_state._sp_last=heard
        if not heard: heard=st.session_state.get('_sp_last','')
        if heard:
            st.markdown(f"<div class='heard-box'>{heard}</div>",unsafe_allow_html=True)
            words=heard.strip().split()
            for wi,word in enumerate(words):
                clean=word.lower().strip(".,!?;:'\"")
                letters=[c.upper() for c in clean if c.isalpha()]
                st.markdown(f"<div class='ws'><div class='wh'>{clean}</div>",unsafe_allow_html=True)
                if letters:
                    n_cols=min(len(letters),8)
                    cols=st.columns(n_cols)
                    for li,ch in enumerate(letters):
                        ip=get_letter_img(ch)
                        with cols[li%n_cols]:
                            if ip: st.image(ip,caption=ch,width=90)
                            else:
                                st.markdown(
                                    f"<div style='width:90px;height:90px;"
                                    f"background:{BG3};border:1.5px solid {BORDER};"
                                    f"border-radius:8px;display:flex;align-items:center;"
                                    f"justify-content:center;font-family:JetBrains Mono,"
                                    f"monospace;font-size:1.6rem;color:{TXT2};'>{ch}</div>",
                                    unsafe_allow_html=True)
                st.markdown("</div>",unsafe_allow_html=True)
                if wi<len(words)-1:
                    st.markdown(f"<div class='space-div'>🖐 &nbsp; SPACE</div>",unsafe_allow_html=True)
            sa1,sa2=st.columns(2,gap="small")
            with sa1:
                if st.button("🔊 Speak aloud",key="sp_speak",use_container_width=True):
                    push_history(heard,"speech"); speak(heard)
            with sa2:
                if st.button("🔄 Clear input",key="sp_clear",use_container_width=True):
                    st.session_state._sp_last=""; st.rerun()