"""
neospasm_core.py — logika inti & tema bersama untuk semua halaman.
Model: fusi 3-arah RGB R(2+1)D (crop bayi) + PoseTCN + VideoMAE.
"""
import os
import numpy as np
import streamlit as st

# ---------------- KONFIGURASI ----------------
MODELS_DIR = "models"
RGB_PT  = os.path.join(MODELS_DIR, "model_exp_dense.pt")
POSE_PT = os.path.join(MODELS_DIR, "model_pose_dense.pt")
VMAE_PT = os.path.join(MODELS_DIR, "model_videomae_exp1.pt")
VMAE_CKPT = "MCG-NJU/videomae-base-finetuned-kinetics"

# Model RGB besar (119MB) diunduh otomatis dari Drive saat app start (tak perlu di GitHub).
RGB_GDRIVE_ID = "1ARIr0cJi__l3N5DB3PVB15PYG1Xkprn-"

W_RGB, W_POSE, W_VMAE = 0.1, 0.3, 0.6          # bobot fusi terbaik (grid 7_fusion3)
RGB_FRAMES, RGB_SIZE = 16, 112
POSE_FRAMES = 32
VMAE_FRAMES = 16
MEAN = np.array([0.43216, 0.394666, 0.37645], np.float32)
STD  = np.array([0.22803, 0.22145, 0.216989], np.float32)
WINDOW_SEC, STRIDE_SEC = 5.0, 2.5
LR_PAIRS = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]

PRIMARY = "#0ea5e9"
TEAL    = "#14b8a6"

# ---------------- TEMA / CSS ----------------
def inject_css():
    st.markdown("""
    <style>
      .stApp { background: linear-gradient(180deg,#f0f9ff 0%,#f8fafc 40%); }
      #MainMenu, footer {visibility: hidden;}
      .hero {
        background: linear-gradient(135deg,#0ea5e9 0%,#14b8a6 100%);
        border-radius: 22px; padding: 38px 34px; color: white;
        box-shadow: 0 12px 30px rgba(14,165,233,.25); margin-bottom: 8px;
      }
      .hero h1 { color:#fff; font-size: 2.5rem; margin:0 0 6px 0; font-weight: 800; letter-spacing:-1px;}
      .hero p  { color:#e0f2fe; font-size: 1.08rem; margin:0; }
      .badge { display:inline-block; background:rgba(255,255,255,.2); padding:4px 12px;
               border-radius:999px; font-size:.78rem; margin-bottom:14px; letter-spacing:.5px;}
      .card {
        background:#fff; border-radius:18px; padding:24px 26px; margin:14px 0;
        box-shadow: 0 4px 18px rgba(15,23,42,.06); border:1px solid #e2e8f0;
      }
      .card h3 { margin-top:0; color:#0f172a; font-weight:700; }
      .step { background:#fff; border-radius:16px; padding:20px; height:100%;
              border:1px solid #e2e8f0; box-shadow:0 2px 10px rgba(15,23,42,.04); }
      .step .ico { font-size:1.8rem; } .step b { color:#0f172a; }
      .step p { color:#64748b; font-size:.9rem; margin:.4rem 0 0 0; }
      .disclaimer {
        background:#fef2f2; border-left:5px solid #ef4444; border-radius:12px;
        padding:16px 20px; color:#991b1b; font-size:.92rem;
      }
      .meter { text-align:center; padding:10px 0; }
      .meter .num { font-size:4rem; font-weight:800; line-height:1; }
      .meter .lab { color:#64748b; font-size:.95rem; margin-top:4px; }
      .bar { height:14px; background:#e2e8f0; border-radius:999px; overflow:hidden; margin:18px 0 6px; }
      .fill { height:100%; border-radius:999px; transition:width .6s; }
      .verdict { border-radius:16px; padding:22px 24px; margin:8px 0; font-size:1.02rem; }
      .v-alert { background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; }
      .v-ok    { background:#f0fdf4; border:1px solid #bbf7d0; color:#166534; }
      .pill { display:inline-block; padding:3px 12px; border-radius:999px; font-size:.82rem;
              font-weight:600; margin-right:6px; }
      div.stButton>button { background:linear-gradient(135deg,#0ea5e9,#14b8a6); color:#fff;
        border:none; border-radius:12px; padding:.6rem 1.4rem; font-weight:700; }
      div.stButton>button:hover { opacity:.92; }
    </style>""", unsafe_allow_html=True)


def _ensure_rgb_model():
    """Unduh model RGB besar dari Google Drive bila belum ada (untuk Streamlit Cloud)."""
    if os.path.exists(RGB_PT):
        return
    os.makedirs(MODELS_DIR, exist_ok=True)
    url = f"https://drive.google.com/uc?id={RGB_GDRIVE_ID}"
    with st.spinner("Mengunduh model RGB (119MB) sekali saja..."):
        try:
            import gdown
            gdown.download(url, RGB_PT, quiet=False)
        except Exception as e:
            raise RuntimeError(
                f"Gagal mengunduh model RGB dari Drive: {e}. "
                "Pastikan file di-share 'Anyone with the link'.")


# ---------------- MODEL ----------------
@st.cache_resource(show_spinner="Memuat model (sekali saja)...")
def load_all():
    import torch, torch.nn as nn
    from torchvision.models.video import r2plus1d_18
    from transformers import VideoMAEModel, VideoMAEImageProcessor
    from ultralytics import YOLO
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    _ensure_rgb_model()
    rgb = r2plus1d_18(weights=None); rgb.fc = nn.Linear(rgb.fc.in_features, 1)
    ck = torch.load(RGB_PT, map_location="cpu"); rgb.load_state_dict(ck["state_dict"]); rgb.to(dev).eval()

    class PoseTCN(nn.Module):
        def __init__(self, c, h, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(c, h, 3, padding=1), nn.BatchNorm1d(h), nn.ReLU(),
                nn.Conv1d(h, h, 3, padding=1), nn.BatchNorm1d(h), nn.ReLU(),
                nn.Conv1d(h, h, 3, padding=1), nn.BatchNorm1d(h), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
            self.drop = nn.Dropout(d); self.fc = nn.Linear(h, 1)
        def forward(self, x):
            return self.fc(self.drop(self.net(x.transpose(1,2)).squeeze(-1))).squeeze(1)
    ckp = torch.load(POSE_PT, map_location="cpu"); c = ckp["config"]
    pose = PoseTCN(c.get("feat_dim",85), c.get("hidden",64), 0.5)
    pose.load_state_dict(ckp["state_dict"]); pose.to(dev).eval()

    class Head(nn.Module):
        def __init__(self, d, h, dr):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d,h), nn.BatchNorm1d(h), nn.ReLU(),
                                     nn.Dropout(dr), nn.Linear(h,1))
        def forward(self, x): return self.net(x).squeeze(1)
    ckv = torch.load(VMAE_PT, map_location="cpu"); cv = ckv["config"]
    enc = VideoMAEModel.from_pretrained(VMAE_CKPT).to(dev).eval()
    proc = VideoMAEImageProcessor.from_pretrained(VMAE_CKPT)
    head = Head(cv.get("feat_dim",768), cv.get("hidden",128), 0.5)
    head.load_state_dict(ckv["state_dict"]); head.to(dev).eval()

    return dict(dev=dev, rgb=rgb, pose=pose, enc=enc, proc=proc, head=head,
                det=YOLO("yolov8n.pt"), yp=YOLO("yolov8n-pose.pt"), torch=torch)


# ---------------- PREPROCESS ----------------
def _read_bgr(path, start, end, n):
    import cv2
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    f0 = int(start*fps); f1 = min(int(end*fps), max(total-1,0))
    idxs = [f0]*n if f1 <= f0 else list(np.linspace(f0,f1,n).astype(int))
    out, last = [], None
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
        fr = fr if ok and fr is not None else (last if last is not None else np.zeros((64,64,3),np.uint8))
        last = fr; out.append(fr)
    cap.release(); return out

def _bbox(frames, det):
    boxes = []
    for fr in frames[::max(1,len(frames)//4)]:
        try:
            res = det.predict(fr, verbose=False, classes=[0], imgsz=320)
            for b in res[0].boxes.xyxy.cpu().numpy(): boxes.append(b)
        except Exception: return None
    if not boxes: return None
    b = np.array(boxes)
    return float(b[:,0].min()), float(b[:,1].min()), float(b[:,2].max()), float(b[:,3].max())

def _crop(frames, bbox, size, m=0.15):
    import cv2
    H, W = frames[0].shape[:2]
    if bbox is None:
        s = min(H,W); box = ((W-s)//2,(H-s)//2,(W-s)//2+s,(H-s)//2+s)
    else:
        x1,y1,x2,y2 = bbox; bw,bh = x2-x1,y2-y1
        x1-=bw*m; x2+=bw*m; y1-=bh*m; y2+=bh*m
        cx,cy=(x1+x2)/2,(y1+y2)/2; side=max(x2-x1,y2-y1)
        box=(cx-side/2,cy-side/2,cx+side/2,cy+side/2)
    x1,y1,x2,y2=[int(round(v)) for v in box]
    x1=max(0,x1);y1=max(0,y1);x2=min(W,x2);y2=min(H,y2)
    if x2-x1<8 or y2-y1<8: x1,y1,x2,y2=0,0,W,H
    return np.stack([cv2.resize(f[y1:y2,x1:x2],(size,size)) for f in frames]).astype(np.uint8)

def _pose_norm(seq):
    seq=seq.astype(np.float32); xy=seq[:,:,:2]; conf=seq[:,:,2:]
    valid=conf[:,:,0]>0.1; out=np.zeros_like(xy)
    for t in range(seq.shape[0]):
        if valid[t].sum()<2: continue
        sh,hp=xy[t][[5,6]],xy[t][[11,12]]
        if valid[t][[5,6]].all() and valid[t][[11,12]].all():
            c=(sh.mean(0)+hp.mean(0))/2; sc=np.linalg.norm(sh.mean(0)-hp.mean(0))
        else:
            p=xy[t][valid[t]]; c=p.mean(0); sc=p.std(0).mean()*2+1e-3
        out[t]=(xy[t]-c)/max(sc,1e-3)
    return np.concatenate([out*valid[:,:,None],conf],axis=2)

def _pose_feats(n):
    xy=n[:,:,:2]; conf=n[:,:,2:]; vel=np.zeros_like(xy); vel[1:]=xy[1:]-xy[:-1]
    f=np.concatenate([xy,vel,conf],axis=2); return f.reshape(f.shape[0],-1)


def predict_window(M, path, s, e):
    import cv2
    torch = M["torch"]; dev = M["dev"]
    bgr32 = _read_bgr(path, s, e, POSE_FRAMES)
    bgr16 = bgr32[::2][:RGB_FRAMES]
    if len(bgr16) < RGB_FRAMES: bgr16 = (bgr16 + [bgr16[-1]]*RGB_FRAMES)[:RGB_FRAMES]
    rgb16 = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr16]

    arr = _crop(rgb16, _bbox(bgr16, M["det"]), RGB_SIZE)
    x = (arr.astype(np.float32)/255.0 - MEAN)/STD
    xt = torch.from_numpy(x).permute(3,0,1,2).unsqueeze(0).float().to(dev)
    with torch.no_grad(): p_rgb = torch.sigmoid(M["rgb"](xt).squeeze(1)).item()

    seq = []
    for fr in bgr32:
        try:
            res = M["yp"].predict(fr, verbose=False, imgsz=384); kp = res[0].keypoints
            d = kp.data.cpu().numpy() if (kp is not None and kp.data is not None) else np.zeros((0,17,3))
        except Exception: d = np.zeros((0,17,3),np.float32)
        seq.append(np.zeros((17,3),np.float32) if d.shape[0]==0 else d[np.argmax(d[:,:,2].sum(1))].astype(np.float32))
    pt = torch.from_numpy(_pose_feats(_pose_norm(np.stack(seq)))).unsqueeze(0).float().to(dev)
    with torch.no_grad(): p_pose = torch.sigmoid(M["pose"](pt)).item()

    px = M["proc"](list(rgb16), return_tensors="pt")["pixel_values"].to(dev)
    with torch.no_grad():
        h = M["enc"](pixel_values=px).last_hidden_state.mean(1)
        p_vmae = torch.sigmoid(M["head"](h)).item()

    fused = W_RGB*p_rgb + W_POSE*p_pose + W_VMAE*p_vmae
    return dict(rgb=p_rgb, pose=p_pose, vmae=p_vmae, fused=fused, t0=s, t1=e)


def analyze_video(M, path, progress=None):
    import cv2
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0; cap.release()
    dur = total/fps if fps else 0
    if dur <= WINDOW_SEC:
        wins = [(0.0, max(dur,0.1))]
    else:
        starts = list(np.arange(0, dur-WINDOW_SEC+1e-6, STRIDE_SEC))
        if not starts or starts[-1] < dur-WINDOW_SEC: starts.append(dur-WINDOW_SEC)
        wins = [(t, t+WINDOW_SEC) for t in starts]
    res = []
    for i, (s, e) in enumerate(wins):
        res.append(predict_window(M, path, s, e))
        if progress: progress.progress((i+1)/len(wins), text=f"Menganalisis jendela {i+1}/{len(wins)}")
    return res, dur

def models_missing():
    # RGB diunduh otomatis dari Drive -> cukup cek 2 file kecil yang ada di repo
    return [p for p in (POSE_PT, VMAE_PT) if not os.path.exists(p)]
