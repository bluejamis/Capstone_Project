# this is yoloold.py (with StrongSORT-style tracker + ReID track_id)
import sys, os, glob, csv, cv2, re, math, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
np.int, np.float = int, float  # legacy compat

# ================= 0) 사용자 설정 =================
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # <-- thisthis/ (scripts의 상위 폴더)
sys.path.insert(0, project_root)

IMAGE_DIR   = os.path.join(project_root, "abcnew")
RECURSIVE   = False
META_USE    = "first"      # 'first' or 'mean'
HEAD_SMOOTH = 5
EXCLUDE_CLASSES = {"roadsign", "roadsign1", "roadsign2"}  # 제외 클래스

# 카메라/지오 파라미터
FX_MM = FY_MM = 3.07
SENSOR_W_MM, SENSOR_H_MM = 6.40, 4.80
CX_MM = CY_MM = 3.04
PITCH_DEG = -3.0
CAM_HEIGHT_M = 1.50
DEFAULT_SIGN_H = 0.75
R_EARTH = 6_378_137.0  # meters

CLASS_SIGN_HEIGHT = {}
YOLO_IMGSZ = 1408
YOLO_CONF  = 0.7
YOLO_IOU   = 0.7
SHOW_DEBUG_TEXT = True

# ===== EfficientNet 분류기 =====
CLS_WEIGHTS    = os.path.join(project_root, "model", "thismodel.pth")  # ckpt에 idx2name 저장된 버전 권장
CLS_INPUT_SIZE = 260
CLS_LABELS     = None   # ckpt에 없을 때만 외부 레이블 파일(.json/.txt) 사용

# ===== ReID (ResNet50) 체크포인트 =====
REID_CKPT = os.path.join(project_root, "reid_runs", "reid_resnet50_best.pth")  # 네가 학습한 pth

# ================ 1) 메타 CSV 로드 ================
meta_candidates = ["hudaters_sorted.csv"]
csv_meta = next((os.path.join(project_root, nm) for nm in meta_candidates if os.path.exists(os.path.join(project_root, nm))), None)
if not csv_meta:
    raise FileNotFoundError("메타 CSV를 찾을 수 없습니다.")
df_meta = pd.read_csv(csv_meta)
if "filename" not in df_meta.columns:
    raise RuntimeError("메타 CSV에 'filename' 컬럼이 없습니다.")
df_meta["filename"] = (
    df_meta["filename"].astype(str)
    .apply(lambda p: os.path.basename(p))
    .str.lower()
    .apply(lambda n: n if os.path.splitext(n)[1] else n + ".jpg")
)
for col in ["lat", "lon"]:
    if col not in df_meta.columns:
        raise RuntimeError(f"메타 CSV에 '{col}' 컬럼이 없습니다.")
if "tmutc" not in df_meta.columns:
    df_meta["tmutc"] = None
df_meta.set_index("filename", inplace=True)

# ================= 2) 출력 경로(롤링) =================
def _roll_dir(base):
    if not os.path.exists(base):
        os.makedirs(base); return base
    ids = [int(m.group(1)) for d in glob.glob(f"{base}_*")
           if (m:=re.search(r"_(\d+)$", d))]
    new = f"{base}_{max(ids)+1}" if ids else f"{base}_1"
    os.makedirs(new, exist_ok=True); return new

def _roll_file(path):
    stem,ext=os.path.splitext(path)
    if not os.path.exists(path): return path
    ids=[int(m.group(1)) for f in glob.glob(f"{stem}_*{ext}")
         if (m:=re.search(r"_(\d+)"+re.escape(ext)+"$", f))]
    return f"{stem}_{max(ids)+1}{ext}" if ids else f"{stem}_1{ext}"

out_root     = os.path.join(project_root, "after")
os.makedirs(out_root, exist_ok=True)
close_dir    = _roll_dir(os.path.join(out_root, "closeup"))
annot_dir    = _roll_dir(os.path.join(out_root, "annot"))
yolo_csv_out = _roll_file(os.path.join(out_root, "yoloold.csv"))

# --- CSV (track_id 컬럼 추가) ---
yolo_w_csv = open(yolo_csv_out, "w", newline="", encoding="utf-8-sig")
yolo_csv_w = csv.writer(yolo_w_csv)
yolo_csv_w.writerow([
    "index","track_id","image_path_rel",
    "yolo_class","cls_name",
    "cam_lon","cam_lat","sign_lon","sign_lat",
    "center_x_px","center_y_px","width_px","height_px","area_px2"
])

# ================= 3) YOLO 모델 =================
from ultralytics import YOLO
WEIGHTS_YOLO = os.path.join(project_root, "model", "you8m", "weights", "best.pt")
if not os.path.exists(WEIGHTS_YOLO):
    raise FileNotFoundError(f"YOLO 가중치를 찾을 수 없습니다: {WEIGHTS_YOLO}")

yolo_model = YOLO(WEIGHTS_YOLO)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model.to(device)
half_flag = (device == 'cuda')
print(f"[INFO] device={device}, half_flag={half_flag}, cuda={torch.cuda.is_available()}")
print(f"[INFO] YOLO weights: {WEIGHTS_YOLO}")

# ================= 4) 유틸 =================
def fpx(f_mm, sensor_w_mm, img_w_px): return f_mm / sensor_w_mm * img_w_px
def wrap_angle(rad): return (rad + math.pi) % (2*math.pi) - math.pi
def heading(lat1, lon1, lat2, lon2):
    lat1,lon1,lat2,lon2 = map(math.radians,(float(lat1),float(lon1),float(lat2),float(lon2)))
    d = lon2 - lon1
    x = math.sin(d) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d)
    return math.atan2(x, y)
def offset(lat, lon, dist_m, bearing_rad):
    lat = math.radians(float(lat)); lon = math.radians(float(lon))
    a = dist_m / R_EARTH
    lat2 = math.asin(math.sin(lat)*math.cos(a) + math.cos(lat)*math.sin(a)*math.cos(bearing_rad))
    lon2 = lon + math.atan2(math.sin(bearing_rad)*math.sin(a)*math.cos(lat),
                            math.cos(a) - math.sin(lat)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def safe_latlon_tmutc(df_idx, filename):
    try:
        rows = df_idx.loc[filename, ["lat","lon","tmutc"]]
    except KeyError:
        return None
    if isinstance(rows, pd.Series):
        return float(rows["lat"]), float(rows["lon"]), rows.get("tmutc", None)
    if META_USE == "mean":
        lat = float(rows["lat"].astype(float).mean())
        lon = float(rows["lon"].astype(float).mean())
        tmutc = rows["tmutc"].iloc[0] if "tmutc" in rows.columns else None
        return lat, lon, tmutc
    first = rows.iloc[0]
    return float(first["lat"]), float(first["lon"]), first.get("tmutc", None)

def wrap_angle_series(arr, window=HEAD_SMOOTH):
    arr_unwrap = np.unwrap(np.where(np.isnan(arr), 0.0, arr))
    yaw_sm = pd.Series(arr_unwrap).replace(0.0, np.nan).ffill().bfill().rolling(
        window=window, center=True, min_periods=1).mean().to_numpy()
    return np.vectorize(wrap_angle)(yaw_sm)

# ---------- PIL 텍스트(한글, 굵게, 색상) ----------
from PIL import Image, ImageDraw, ImageFont
def _pick_font(size=24):
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size=size)
            except: pass
    return ImageFont.load_default()

def draw_label_combo(ann_bgr, x, y, yolo_name, eff_name, track_id=None):
    """
    ann_bgr: OpenCV BGR 이미지
    (x,y): 텍스트 좌상단
    yolo_name(초록) | eff_name(빨강) | #id
    """
    img = Image.fromarray(cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _pick_font(size=24)

    text_yolo = str(yolo_name)
    text_sep  = " | "
    text_eff  = str(eff_name) if eff_name is not None else "?"
    text_id   = f" | id={track_id}" if track_id is not None else ""

    text_all = text_yolo + text_sep + text_eff + text_id

    tw = int(draw.textlength(text_all, font=font))
    th = int(font.size * 1.6)

    bg = Image.new("RGBA", (tw + 12, th + 8), (0, 0, 0, 140))
    img.paste(bg, (x, max(0, y - th - 6)), bg)

    tx = x + 6
    ty_top = y - th
    draw.text((tx, ty_top), text_all, font=font,
              fill=(255,255,255), stroke_width=2, stroke_fill=(0,0,0))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ================= 5) 프레임 목록 =================
p = Path(IMAGE_DIR)
if not p.exists():
    raise RuntimeError(f"IMAGE_DIR가 존재하지 않습니다: {IMAGE_DIR}")

img_exts_lower = {".jpg",".jpeg",".png",".bmp"}
paths = ([str(q) for q in p.rglob("*") if q.is_file() and q.suffix.lower() in img_exts_lower]
         if RECURSIVE else
         [str(q) for q in p.iterdir() if q.is_file() and q.suffix.lower() in img_exts_lower])
paths.sort()
if not paths:
    raise RuntimeError(f"이미지 파일이 없습니다: {IMAGE_DIR}")

id2path = {i: fp for i, fp in enumerate(paths)}
ids = sorted(id2path.keys())
min_id, max_id = ids[0], ids[-1]
print(f"[INFO] 이미지 {len(ids)}장 감지, ID 범위: {min_id}~{max_id}")

# ================= 6) yaw 계산 =================
raw_yaw = {}
prev_fid = None
for fid in range(min_id, max_id+1):
    fp = id2path.get(fid)
    if not fp: continue
    fn = os.path.basename(fp).lower()
    cur = safe_latlon_tmutc(df_meta, fn)
    if cur is None: continue
    cur_lat, cur_lon, _ = cur
    nxt = next((j for j in range(fid+1, max_id+1) if id2path.get(j)), None)
    if prev_fid is not None:
        prev_fn = os.path.basename(id2path[prev_fid]).lower()
        prev_meta = safe_latlon_tmutc(df_meta, prev_fn)
        if prev_meta is not None:
            lat1, lon1, _ = prev_meta
            raw_yaw[fid] = heading(lat1, lon1, cur_lat, cur_lon)
    elif nxt is not None:
        nxt_fn = os.path.basename(id2path[nxt]).lower()
        nxt_meta = safe_latlon_tmutc(df_meta, nxt_fn)
        if nxt_meta is not None:
            lat2, lon2, _ = nxt_meta
            raw_yaw[fid] = heading(cur_lat, cur_lon, lat2, lon2)
    prev_fid = fid

arr = np.full(max_id+1, np.nan, dtype=float)
for k,v in raw_yaw.items(): arr[k] = v
yaw_sm = wrap_angle_series(arr)
def get_yaw(fid): return float(yaw_sm[fid]) if fid <= max_id else 0.0

# ================= EfficientNet 로더 =================
from torchvision import models, transforms
def _build_classifier(weights_path=None, input_size=CLS_INPUT_SIZE, labels_path=None):
    device_cls = "cuda" if torch.cuda.is_available() else "cpu"

    def _safe_load(path):
        try: return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError: return torch.load(path, map_location="cpu")

    def _strip_prefix(sd, prefixes=("module.","model.","net.","backbone.")):
        out = {}
        for k,v in sd.items():
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    break
            out[k] = v
        return out

    def _infer_num_classes(sd):
        for k in ["classifier.1.weight","classifier.weight","heads.classifier.weight","head.fc.weight","fc.weight"]:
            if k in sd and hasattr(sd[k], "shape"): return int(sd[k].shape[0])
        return None

    idx2name = None
    if weights_path and os.path.exists(weights_path):
        model = models.efficientnet_b2(weights=None)
        state = _safe_load(weights_path)
        if isinstance(state, dict) and "model" in state:
            sd = state["model"]; idx2name = state.get("idx2name")
        elif isinstance(state, dict) and "state_dict" in state:
            sd = state["state_dict"]
        else:
            sd = state if isinstance(state, dict) else None
        if sd is None: raise RuntimeError("잘못된 EfficientNet 체크포인트 형식입니다.")
        sd = _strip_prefix(sd)
        numc = _infer_num_classes(sd) or 1000
        if model.classifier[1].out_features != numc:
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, numc)
        model.load_state_dict(sd, strict=False)
        if not isinstance(idx2name, dict) and labels_path and os.path.exists(labels_path):
            if labels_path.lower().endswith(".json"):
                with open(labels_path,"r",encoding="utf-8") as f:
                    m=json.load(f); idx2name={int(k):str(v) for k,v in m.items()}
            else:
                names=[ln.strip() for ln in open(labels_path,"r",encoding="utf-8") if ln.strip()]
                idx2name={i:n for i,n in enumerate(names)}
        if not isinstance(idx2name, dict):
            idx2name={i:f"class_{i}" for i in range(numc)}
    else:
        w=models.EfficientNet_B2_Weights.IMAGENET1K_V1
        model=models.efficientnet_b2(weights=w)
        cats=w.meta.get("categories",[])
        idx2name={i:n for i,n in enumerate(cats)}
    model.eval().to(device_cls)

    tfm = transforms.Compose([
        transforms.Resize(input_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return model, tfm, device_cls, idx2name

def _crop(img_bgr, x1,y1,x2,y2):
    h,w=img_bgr.shape[:2]
    x1=max(0,int(x1)); y1=max(0,int(y1)); x2=min(w,int(x2)); y2=min(h,int(y2))
    if x2<=x1 or y2<=y1: return None
    roi = img_bgr[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

cls_model, cls_tfm, device_cls, idx2name = _build_classifier(CLS_WEIGHTS, CLS_INPUT_SIZE, CLS_LABELS)

# ================= ReID (ResNet50 feature extractor) =================
def build_reid_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ReID 체크포인트를 찾을 수 없습니다: {ckpt_path}")
    device_reid = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"]
    num_classes = ckpt["num_classes"]

    backbone = models.resnet50(weights=None)
    in_features = backbone.fc.in_features  # 2048
    backbone.fc = nn.Linear(in_features, num_classes)
    backbone.load_state_dict(state, strict=True)
    backbone.to(device_reid).eval()

    # fc 를 제거한 feature extractor (conv~avgpool까지)
    feat_extractor = nn.Sequential(*list(backbone.children())[:-1]).to(device_reid).eval()

    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    emb_dim = in_features
    print(f"[INFO] ReID emb_dim={emb_dim}, device={device_reid}")
    return feat_extractor, tfm, device_reid, emb_dim

reid_backbone, reid_tfm, device_reid, REID_EMB_DIM = build_reid_model(REID_CKPT)

def extract_reid_embedding(img_bgr, box_xyxy):
    """img_BGR와 [x1,y1,x2,y2]에서 ReID 임베딩 벡터(np.array)를 뽑는다."""
    x1,y1,x2,y2 = map(int, box_xyxy)
    h,w = img_bgr.shape[:2]
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h-1, y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(REID_EMB_DIM, dtype=np.float32)

    crop = img_bgr[y1:y2, x1:x2]
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        t = reid_tfm(pil).unsqueeze(0).to(device_reid)  # [1,3,H,W]
        feat = reid_backbone(t)  # [1, C, 1, 1]
        feat = feat.view(1, -1)  # [1, C]
        feat = F.normalize(feat, p=2, dim=1)  # L2 정규화
    return feat[0].cpu().numpy()

# ================= Simple StrongSORT-style Tracker =================
def iou_xyxy(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    if inter <= 0:
        return 0.0
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union

def cos_sim(a, b, eps=1e-6):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + eps))

class Track:
    def __init__(self, track_id, bbox, emb, frame_id, cls_name):
        self.id = track_id
        self.bbox = np.array(bbox, dtype=float)
        self.emb  = np.array(emb, dtype=float)
        self.last_frame = frame_id
        self.misses = 0
        self.hits   = 1

        # 클래스 + 중심좌표 저장
        self.cls_name = cls_name
        self.cx = 0.5 * (bbox[0] + bbox[2])
        self.cy = 0.5 * (bbox[1] + bbox[3])

class SimpleStrongSort:
    """
    진짜 StrongSORT는 아니고, ReID + IoU + 위치 기반의 간단한 StrongSORT 스타일 트래커.
    - appearance(코사인) + IoU + center distance + (soft) class check
    """
    def __init__(self, emb_dim, max_age=15,
                 iou_thresh=0.1, sim_thresh=0.15, alpha=0.95,
                 max_center_dist=300, sim_mismatch_min=0.7):
        self.emb_dim = emb_dim
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.sim_thresh = sim_thresh
        self.alpha = alpha  # score = alpha*sim + (1-alpha)*iou

        # 추가 파라미터
        self.max_center_dist = max_center_dist      # 너무 큰 점프면 다른 물체 취급
        self.sim_mismatch_min = sim_mismatch_min    # 클래스 다를 때 허용할 최소 sim

        self.tracks = []
        self.next_id = 0

    def update(self, det_boxes, det_embs, det_classes, frame_id):
        """
        det_boxes: (N,4) np.array, xyxy
        det_embs : (N,emb_dim) np.array
        det_classes: length N list/array (yolo_name 같은 문자열)
        return: length N list of track_ids (각 detection에 대응)
        """
        N = det_boxes.shape[0]
        track_ids = [-1] * N

        # detection 중심 좌표
        det_centers = []
        for box in det_boxes:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            det_centers.append((cx, cy))

        used_tracks = set()

        # 1) 기존 트랙과 매칭
        for i in range(N):
            box_i = det_boxes[i]
            emb_i = det_embs[i]
            cls_i = det_classes[i]
            cx_i, cy_i = det_centers[i]

            best_track = None
            best_score = -1e9

            for t in self.tracks:
                if t.id in used_tracks:
                    continue

                iou = iou_xyxy(box_i, t.bbox)
                sim = cos_sim(emb_i, t.emb)

                # 중심 점프가 너무 크고, IoU도 매우 작으면 → 다른 표지판일 가능성 높음
                dist = math.hypot(cx_i - t.cx, cy_i - t.cy)
                if dist > self.max_center_dist and iou < self.iou_thresh:
                    continue

                # 클래스 soft 체크
                same_class = (t.cls_name == cls_i)
                if not same_class:
                    # YOLO 오분류를 허용하기 위해,
                    # ReID 유사도가 충분히 높을 때만 예외적으로 매칭 허용
                    if sim < self.sim_mismatch_min:
                        continue

                # 완화된 AND 로직:
                # 둘 다 안 좋을 때만 버림 → (iou >= thresh) 또는 (sim >= thresh)이면 허용
                if (iou < self.iou_thresh) and (sim < self.sim_thresh):
                    continue

                score = self.alpha * sim + (1.0 - self.alpha) * iou
                if score > best_score:
                    best_score = score
                    best_track = t

            if best_track is not None:
                # 매칭 성공
                best_track.bbox = box_i
                best_track.emb  = emb_i
                best_track.last_frame = frame_id
                best_track.misses = 0
                best_track.hits  += 1
                best_track.cx = cx_i
                best_track.cy = cy_i

                # track의 클래스는 최신 detection 기준으로 업데이트
                best_track.cls_name = cls_i

                used_tracks.add(best_track.id)
                track_ids[i] = best_track.id

        # 2) 매칭 안 된 detection -> 새 트랙 생성
        for i in range(N):
            if track_ids[i] != -1:
                continue
            box_i = det_boxes[i]
            emb_i = det_embs[i]
            cls_i = det_classes[i]
            new_t = Track(self.next_id, box_i, emb_i, frame_id, cls_i)
            self.tracks.append(new_t)
            track_ids[i] = self.next_id
            self.next_id += 1

        # 3) 너무 오래 안 보인 트랙 제거
        alive_tracks = []
        for t in self.tracks:
            if t.last_frame != frame_id:
                t.misses += 1
            if t.misses <= self.max_age:
                alive_tracks.append(t)
        self.tracks = alive_tracks

        return track_ids

# 트래커 인스턴스 (파라미터는 나중에 튜닝 가능)
tracker = SimpleStrongSort(
    emb_dim=REID_EMB_DIM,
    max_age=30,
    iou_thresh=0.01,     # IoU는 매우 느슨하게
    sim_thresh=0.05,     # 기본 sim 기준
    alpha=0.95,          # appearance 위주
    max_center_dist=300, # 프레임 내에서 이 픽셀 이상 점프하면 다른 표지판 취급
    sim_mismatch_min=0.7 # 클래스 다를 때 이 값 이상이면 YOLO 오분류로 보고 허용
)

# ================= 7) 메인 루프 =================
t0 = time.time()
stats = {"saved_rows":0, "saved_ann":0, "ann_skipped_empty":0,
         "skip_no_detection":0, "skip_no_meta":0, "skip_imread_fail":0}
yolo_idx = 0
pr = math.radians(PITCH_DEG)

with torch.no_grad():
    torch.backends.cudnn.benchmark = True
    for fid in range(min_id, max_id+1):
        fp = id2path.get(fid)
        if not fp: continue

        img = cv2.imread(fp)
        if img is None:
            stats["skip_imread_fail"] += 1; continue

        fn = os.path.basename(fp).lower()
        meta = safe_latlon_tmutc(df_meta, fn)
        if meta is None:
            stats["skip_no_meta"] += 1; continue
        cam_lat, cam_lon, _ = meta
        yaw = get_yaw(fid)

        pred = yolo_model(img, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, half=half_flag, verbose=False)
        det = pred[0]
        ann = img.copy()
        drew_any = False

        frame_dets = []  # 이 프레임의 detection 정보를 일단 다 모았다가 -> tracking 후 CSV/그리기

        if det is not None and hasattr(det, "boxes") and len(det):
            h, w = img.shape[:2]
            fx = fpx(FX_MM, SENSOR_W_MM, w)
            fy = fpx(FX_MM, SENSOR_H_MM, h)
            cx_pix = CX_MM / SENSOR_W_MM * w
            img_rel_original = os.path.relpath(fp, start=project_root)

            boxes_xyxy = det.boxes.xyxy.detach().cpu().numpy()
            clss       = det.boxes.cls.detach().cpu().numpy().astype(int)

            for box, cid in zip(boxes_xyxy, clss):
                yolo_name = yolo_model.names[cid] if (0 <= cid < len(yolo_model.names)) else str(cid)
                if yolo_name in EXCLUDE_CLASSES:  # roadsign, roadsign1, roadsign2 제외
                    continue

                x1, y1, x2, y2 = map(int, map(round, box))
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w - 1, x2), min(h - 1, y2)
                bw, bh = x2c - x1c, y2c - y1c
                if bw <= 0 or bh <= 0:
                    continue

                cx_p = x1c + bw // 2
                cy_p = y1c + bh // 2
                area = bw * bh

                # 거리/방위 추정
                Hs = CLASS_SIGN_HEIGHT.get(yolo_name, DEFAULT_SIGN_H)
                z = Hs * fy / max(bh, 1) * math.cos(pr)
                horiz = math.atan((cx_p - cx_pix) / max(fx, 1e-6))
                bear = wrap_angle(yaw + horiz)
                flat = max(z**2 - (Hs - CAM_HEIGHT_M)**2, 0.0)
                gdist = math.sqrt(flat)
                s_lat, s_lon = offset(cam_lat, cam_lon, gdist, bear)

                # EfficientNet 분류 (표시용)
                eff_name = None
                roi_pil = _crop(img, x1c, y1c, x2c, y2c)
                if roi_pil is not None:
                    t_img = cls_tfm(roi_pil).unsqueeze(0).to(device_cls)
                    logits = cls_model(t_img)
                    k = int(torch.argmax(logits, dim=1).item())
                    eff_name = idx2name.get(k, f"class_{k}")

                # ReID 임베딩
                emb = extract_reid_embedding(img, [x1c, y1c, x2c, y2c])

                frame_dets.append({
                    "box": [x1c, y1c, x2c, y2c],
                    "cx": cx_p, "cy": cy_p,
                    "bw": bw, "bh": bh, "area": area,
                    "yolo_name": yolo_name,
                    "eff_name": eff_name,
                    "cam_lon": cam_lon, "cam_lat": cam_lat,
                    "s_lon": s_lon, "s_lat": s_lat,
                    "img_rel": img_rel_original,
                    "fn": fn,
                    "emb": emb,
                })

        # === StrongSORT-style tracking ===
        if frame_dets:
            det_boxes   = np.array([d["box"] for d in frame_dets], dtype=float)
            det_embs    = np.stack([d["emb"] for d in frame_dets], axis=0)
            det_classes = [d["yolo_name"] for d in frame_dets]

            track_ids_frame = tracker.update(det_boxes, det_embs, det_classes, frame_id=fid)

            # detection + track_id로 실제 그리기/저장/CSV 기록
            for det_info, t_id in zip(frame_dets, track_ids_frame):
                x1c, y1c, x2c, y2c = det_info["box"]
                bw, bh, area = det_info["bw"], det_info["bh"], det_info["area"]
                cx_p, cy_p = det_info["cx"], det_info["cy"]
                yolo_name = det_info["yolo_name"]
                eff_name  = det_info["eff_name"]
                cam_lon, cam_lat = det_info["cam_lon"], det_info["cam_lat"]
                s_lon, s_lat     = det_info["s_lon"], det_info["s_lat"]
                img_rel_original = det_info["img_rel"]
                fn = det_info["fn"]

                # 박스/라벨
                cv2.rectangle(ann, (x1c,y1c), (x2c,y2c), (0,255,0), 2)
                if SHOW_DEBUG_TEXT:
                    top_y = max(0, y1c - 6)
                    ann = draw_label_combo(ann, x1c, top_y, yolo_name, eff_name, track_id=t_id)

                # 크롭 저장
                crop = img[y1c:y2c, x1c:x2c]
                cfile = f"{yolo_idx:05d}_{fn}"
                cv2.imwrite(os.path.join(close_dir, cfile), crop)

                # CSV 기록 (track_id 포함)
                yolo_csv_w.writerow([
                    yolo_idx, t_id, img_rel_original,
                    yolo_name, eff_name,
                    cam_lon, cam_lat, s_lon, s_lat,
                    cx_p, cy_p, bw, bh, area
                ])
                stats["saved_rows"] += 1
                yolo_idx += 1

            drew_any = True
        else:
            stats["skip_no_detection"] += 1

        # 어노테이션 저장
        if drew_any:
            anno_path = os.path.join(annot_dir, os.path.basename(fp))
            if os.path.exists(anno_path):
                stem, ext = os.path.splitext(os.path.basename(fp))
                anno_path = os.path.join(annot_dir, f"{stem}_{yolo_idx}{ext}")
            cv2.imwrite(anno_path, ann)
            stats["saved_ann"] += 1
        else:
            stats["ann_skipped_empty"] += 1

# --- 파일 핸들러 닫기 ---
yolo_w_csv.close()

# ================= 8) 요약 =================
elapsed = time.time() - t0
print("\n" + "—"*60)
print("✅ 처리 완료 ")
print("—"*60)
print(f"✅ CSV(utf-8-sig) → {yolo_csv_out}")
print(f"✅ 크롭 이미지     → {close_dir}")
print(f"✅ Annot 이미지    → {annot_dir}")
print(f"⏱️ 처리 시간        → {elapsed:.1f}s")
print("—"*60)
