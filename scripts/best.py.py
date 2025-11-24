#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 + Ego-Compensated DeepSORT → World-coord (Full Script)
- 출력은 '예측'이 아닌 '업데이트에 사용된 YOLO 측정 박스'로 고정
- 외부 ID는 "트랙 객체" 단위로 1회 발급 (tid 재사용 문제 제거)
- 이미 사용한 외부 ID는 절대 재사용하지 않음(전역 json에 영구 저장)
- YOLO-only, YOLO+DeepSORT, 트랙 중앙값 CSV 출력
- 어노테이션/클로즈업 이미지 저장
customtrack.py
"""

import sys, os, glob, csv, cv2, re, math, time, json, atexit
import torch
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
np.int, np.float = int, float  # legacy compat

# ================= 0) 사용자 설정 =================
IMAGE_DIR   = r"C:\Users\sang\Desktop\LoFTR\abc"
RECURSIVE   = False
META_USE    = "first"      # 'first' or 'mean'
HEAD_SMOOTH = 5
EXCLUDE_CLASSES = {"roadsign", "roadsign1", "roadsign2"}

# 카메라/지오 파라미터
FX_MM = FY_MM = 3.07
SENSOR_W_MM, SENSOR_H_MM = 6.40, 4.80
CX_MM = CY_MM = 3.04
PITCH_DEG = -3.0
CAM_HEIGHT_M = 1.50
DEFAULT_SIGN_H = 0.75
R_EARTH = 6_378_137.0  # meters

CLASS_SIGN_HEIGHT = {
    # "speed_limit": 0.60,
    # "stop": 0.75,
}

WRITE_MANIFEST = True

# YOLO 추론 파라미터
YOLO_IMGSZ = 1408
YOLO_CONF  = 0.45
YOLO_IOU   = 0.45

# 매칭/시각화 옵션
IOU_MATCH_TH = 0.2
SHOW_DEBUG_TEXT = True

# ================ 1) 경로/프로젝트 루트 ================
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 메타 CSV 탐색
meta_candidates = ["hudaters_sorted.csv"]
csv_meta = None
for nm in meta_candidates:
    test = os.path.join(project_root, nm)
    if os.path.exists(test):
        csv_meta = test
        break
if not csv_meta:
    raise FileNotFoundError(
        "메타 CSV를 찾을 수 없습니다. 시도한 경로: " +
        ", ".join(os.path.join(project_root, x) for x in meta_candidates)
    )

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

out_root   = os.path.join(project_root, "before")
os.makedirs(out_root, exist_ok=True)
close_dir  = _roll_dir(os.path.join(out_root, "closeup"))
annot_dir  = _roll_dir(os.path.join(out_root, "annot"))
csv_out    = _roll_file(os.path.join(out_root, "my.csv"))
center_out = _roll_file(os.path.join(out_root, "sign_centers.csv"))
yolo_csv_out = _roll_file(os.path.join(out_root, "yoloonly.csv"))

# --- my.csv (YOLO+DeepSORT) ---
w_csv = open(csv_out, "w", newline="", encoding="utf-8")
csv_w = csv.writer(w_csv)
csv_w.writerow([
    "index","image_path_rel","class_id","class_name","track_id",
    "cam_lon","cam_lat","sign_lon","sign_lat",
    "center_x_px","center_y_px","width_px","height_px","area_px2","tmutc"
])

# --- yoloonly.csv ---
yolo_w_csv = open(yolo_csv_out, "w", newline="", encoding="utf-8")
yolo_csv_w = csv.writer(yolo_w_csv)
yolo_csv_w.writerow([
    "index","image_path_rel","class_id","class_name","track_id",
    "cam_lon","cam_lat","sign_lon","sign_lat",
    "center_x_px","center_y_px","width_px","height_px","area_px2","tmutc"
])

# ================= 3) 모델/DeepSORT =================
from ultralytics import YOLO

# --- Feature Extractor 호환 임포트 ---
try:
    from deep_sort_pytorch.deep_sort.deep.feature_extractor import FeatureExtractor as _Extractor
except Exception:
    try:
        from deep_sort_pytorch.deep_sort.deep.feature_extractor import Extractor as _Extractor
    except Exception:
        class _Extractor:
            """폴백(appearance 미사용): 0 벡터 반환"""
            def __init__(self, *args, **kwargs): pass
            def __call__(self, im_crops):
                return np.zeros((len(im_crops), 128), dtype=np.float32)

from deep_sort_pytorch.deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort_pytorch.deep_sort.sort.preprocessing import non_max_suppression
from deep_sort_pytorch.deep_sort.sort.detection import Detection
from deep_sort_pytorch.deep_sort.sort.tracker import Tracker as OriginalTracker

# --- KalmanFilterEgo 유연 임포트 ---
KalmanFilterEgo = None
for _cand in (
    "deep_sort_pytorch.deep_sort.sort.kalman_filter",
    "deep_sort_pytorch.deep_sort.kalman_filter",
    "deep_sort_pytorch.deep_sort.sort.kalman_filter",
    "kalman_filter",
):
    try:
        KalmanFilterEgo = __import__(_cand, fromlist=["KalmanFilterEgo"]).KalmanFilterEgo
        break
    except Exception:
        pass
if KalmanFilterEgo is None:
    raise ImportError("KalmanFilterEgo를 찾을 수 없습니다. kalman_filter.py가 경로상 존재하는지 확인하세요.")

weights = os.path.join(current_dir,"runs","you8m","weights","best.pt")
if not os.path.exists(weights):
    raise FileNotFoundError(f"YOLO 가중치를 찾을 수 없습니다: {weights}")
model = YOLO(weights)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
half_flag = (device == 'cuda')

# DeepSORT 파라미터
REID_CKPT          = os.path.join(current_dir, "deep_sort_pytorch", "deep_sort", "deep", "checkpoint", "ckpt.t7")
MAX_DIST           = 0.75
MIN_CONFIDENCE     = 0.30
NMS_MAX_OVERLAP    = 0.30
MAX_IOU_DISTANCE   = 0.90
MAX_AGE            = 70
N_INIT             = 2
NN_BUDGET          = 512
if not os.path.exists(REID_CKPT):
    print(f"[WARN] ReID 체크포인트가 없어 appearance 매칭이 약해질 수 있습니다: {REID_CKPT}")

# ===== 4.1) 외부 ID 발급기 (객체 기반, 영구 저장) =====
_ext_state_path = os.path.join(out_root, "ext_id_state.json")

def _load_ext_state(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            nxt = int(data.get("next_ext_id", 1))
            used = set(map(int, data.get("used_ext_ids", [])))
            return nxt if nxt >= 1 else 1, used
        except Exception:
            pass
    return 1, set()

def _save_ext_state(path, next_id, used_ids):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"next_ext_id": int(next_id),
                       "used_ext_ids": sorted(map(int, used_ids))},
                      f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] failed to persist ext-id state: {e}")

next_ext_id, used_ext_ids = _load_ext_state(_ext_state_path)

def _take_next_free_id(start_from: int) -> int:
    i = max(1, int(start_from))
    while i in used_ext_ids:
        i += 1
    used_ext_ids.add(i)
    return i

@atexit.register
def _persist_ext_ids():
    _save_ext_state(_ext_state_path, next_ext_id, used_ext_ids)

# --- Tracker 확장: ego_H 전달 ---
class EgoTracker(OriginalTracker):
    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        super().__init__(metric, max_iou_distance, max_age, n_init)
        self.kf = KalmanFilterEgo()
    def predict(self, ego_H=None):
        for track in self.tracks:
            track.predict(self.kf, ego_H=ego_H)

# --- DeepSort 래퍼 ---
class EgoDeepSort(object):
    def __init__(self, reid_ckpt, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0,
                 max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        try:
            self.extractor = _Extractor(model_path=reid_ckpt, use_cuda=use_cuda)
        except TypeError:
            try:
                self.extractor = _Extractor(reid_ckpt, use_cuda=use_cuda)
            except Exception:
                self.extractor = _Extractor()
        metric = NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
        self.tracker = EgoTracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
        # 객체기반 외부ID 매핑
        self._track2ext = {}  # id(track) -> ext_id

    def _xywh_to_tlwh(self, bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            tlwh = bbox_xywh.copy()
        else:
            tlwh = bbox_xywh.clone()
        tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return tlwh

    def _tlwh_to_xywh(self, bbox_tlwh):
        ret = np.asarray(bbox_tlwh).copy()
        ret[:2] += ret[2:] / 2.
        return ret

    def _clip_xyxy(self, x1, y1, x2, y2, w, h):
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w-1, int(x2)); y2 = min(h-1, int(y2))
        return x1, y1, x2, y2

    def _xywh_to_xyxy(self, box, w, h):
        x,y,bw,bh = box
        x1 = x - bw/2; y1 = y - bh/2
        x2 = x + bw/2; y2 = y + bh/2
        return self._clip_xyxy(x1,y1,x2,y2,w,h)

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        oh, ow = ori_img.shape[:2]
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box, ow, oh)
            if x2 > x1 and y2 > y1:
                im = ori_img[y1:y2, x1:x2]
                im_crops.append(im)
        if im_crops:
            feats = self.extractor(im_crops)
        else:
            feats = np.array([])
        return feats

    def update(self, xywhs, confidences, oids, image, ego_H=None):
        import numpy as _np

        # 1) 예측(에고 모션 반영)
        self.tracker.predict(ego_H=ego_H)

        # 2) 특징 & Detection
        features = self._get_features(xywhs, image)
        bbox_tlwh = self._xywh_to_tlwh(xywhs)

        dets = []
        confidences = _np.asarray(confidences, dtype=float).ravel()
        for i, conf in enumerate(confidences):
            if conf <= self.min_confidence:
                continue

            feat_i = _np.asarray(features[i], dtype=_np.float32) if (features is not None and len(features) > i) \
                     else _np.zeros((128,), dtype=_np.float32)

            oid_i = int(oids[i]) if oids is not None else -1

            try:
                d = Detection(bbox_tlwh[i], float(conf), feat_i, oid_i)
            except TypeError:
                try:
                    d = Detection(bbox_tlwh[i], float(conf), oid_i, feat_i)
                except TypeError:
                    d = Detection(bbox_tlwh[i], float(conf), feat_i)
                    setattr(d, "oid", oid_i)

            if not hasattr(d, "feature") or not isinstance(d.feature, _np.ndarray):
                d.feature = feat_i
            if not hasattr(d, "oid"):
                setattr(d, "oid", oid_i)

            dets.append(d)

        # 3) NMS
        if dets:
            boxes = _np.array([d.tlwh for d in dets])
            scores = _np.array([d.confidence for d in dets])
            indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
            dets = [dets[i] for i in indices]

        # 4) 업데이트
        self.tracker.update(dets)

        # 5) 출력: 이번 프레임 '실제 측정 박스'만 + 외부ID(객체 기반)
        outs = []
        global next_ext_id
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if not getattr(track, "was_updated", False):
                continue
            det = getattr(track, "last_det", None)
            if det is None:
                continue

            tkey = id(track)
            if not hasattr(track, "_ext_id"):
                eid = _take_next_free_id(next_ext_id)
                track._ext_id = eid
                self._track2ext[tkey] = eid
                next_ext_id = eid + 1
            else:
                eid = track._ext_id

            tlwh = det.tlwh if hasattr(det, "tlwh") else track.to_tlwh()
            x, y, w, h = tlwh
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            cls = getattr(track, "class_id", getattr(track, "oid", -1))
            outs.append(_np.array([x1,y1,x2,y2, eid, int(cls)], dtype=_np.float32))

        if len(outs):
            return _np.stack(outs, axis=0)
        return _np.zeros((0,6), dtype=_np.float32)

# 인스턴스 생성
tracker = EgoDeepSort(
    REID_CKPT,
    max_dist=MAX_DIST,
    min_confidence=MIN_CONFIDENCE,
    nms_max_overlap=NMS_MAX_OVERLAP,
    max_iou_distance=MAX_IOU_DISTANCE,
    max_age=MAX_AGE,
    n_init=N_INIT,
    nn_budget=NN_BUDGET,
    use_cuda=torch.cuda.is_available()
)

is_cuda = torch.cuda.is_available()
print(f"[INFO] device={device}, half_flag={half_flag}, tracker_cuda={is_cuda}")
print(f"[INFO] DeepSORT params: "
      f"N_INIT={N_INIT}, MAX_AGE={MAX_AGE}, MIN_CONF={MIN_CONFIDENCE}, "
      f"MAX_DIST={MAX_DIST}, MAX_IOU_DIST={MAX_IOU_DISTANCE}, "
      f"NMS_MAX_OVERLAP={NMS_MAX_OVERLAP}, NN_BUDGET={NN_BUDGET}")

# ================= 4) 유틸 =================
def fpx(f_mm, sensor_w_mm, img_w_px):
    return f_mm / sensor_w_mm * img_w_px

def wrap_angle(rad):
    return (rad + math.pi) % (2*math.pi) - math.pi

def heading(lat1, lon1, lat2, lon2):
    lat1 = float(lat1); lon1 = float(lon1); lat2 = float(lat2); lon2 = float(lon2)
    lat1,lon1,lat2,lon2 = map(math.radians,(lat1,lon1,lat2,lon2))
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

def xyxy_to_cxcywh(box):
    x1,y1,x2,y2 = box
    return [x1+(x2-x1)/2, y1+(y2-y1)/2, x2-x1, y2-y1]

def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0:
        return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / max(union_area, 1e-9)

def safe_latlon_tmutc(df_idx, filename):
    try:
        rows = df_idx.loc[filename, ["lat","lon","tmutc"]]
    except KeyError:
        return None
    if isinstance(rows, pd.Series):
        lat, lon, tmutc = rows["lat"], rows["lon"], rows.get("tmutc", None)
        return float(lat), float(lon), tmutc
    else:
        if META_USE == "mean":
            lat = float(rows["lat"].astype(float).mean())
            lon = float(rows["lon"].astype(float).mean())
            tmutc = rows["tmutc"].iloc[0] if "tmutc" in rows.columns else None
            return lat, lon, tmutc
        else:
            first = rows.iloc[0]
            lat, lon = float(first["lat"]), float(first["lon"])
            tmutc = first.get("tmutc", None)
            return lat, lon, tmutc

def robust_track_center(pts):
    if not pts:
        return None
    lons = np.array([p[0] for p in pts], dtype=float)
    lats = np.array([p[1] for p in pts], dtype=float)
    lon_med, lat_med = np.median(lons), np.median(lats)
    coslat = math.cos(math.radians(lat_med))
    dx = (lons - lon_med) * coslat
    dy = (lats - lat_med)
    d = np.hypot(dx, dy)
    q1, q3 = np.percentile(d, [25, 75]); iqr = q3 - q1
    thr = q3 + 1.5*iqr
    mask = d <= thr
    if not np.any(mask):
        return float(lon_med), float(lat_med), 0
    return float(np.median(lons[mask])), float(np.median(lats[mask])), int(mask.sum())

# ================= 5) 프레임 ID 매핑 =================
p = Path(IMAGE_DIR)
if not p.exists():
    raise RuntimeError(f"IMAGE_DIR가 존재하지 않습니다: {IMAGE_DIR}")

img_exts_lower = {".jpg",".jpeg",".png",".bmp"}
if RECURSIVE:
    paths = [str(q) for q in p.rglob("*") if q.is_file() and q.suffix.lower() in img_exts_lower]
else:
    paths = [str(q) for q in p.iterdir() if q.is_file() and q.suffix.lower() in img_exts_lower]
paths.sort()
if not paths:
    raise RuntimeError(f"이미지 파일이 없습니다: {IMAGE_DIR}")

id2path = {i: fp for i, fp in enumerate(paths)}
ids = sorted(id2path.keys())
min_id, max_id = ids[0], ids[-1]
print(f"[INFO] 이미지 {len(ids)}장 감지, ID 범위: {min_id}~{max_id}")

if WRITE_MANIFEST:
    manifest = os.path.join(out_root, "detected_manifest.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        for fid in ids:
            f.write(f"{fid}\t{id2path[fid]}\n")
    print(f"[INFO] 감지 목록 → {manifest}")

# ================= 6) yaw 계산 =================
def wrap_angle_series(arr, window=HEAD_SMOOTH):
    arr_unwrap = np.unwrap(np.where(np.isnan(arr), 0.0, arr))
    ser = pd.Series(arr_unwrap)
    yaw_sm = ser.replace(0.0, np.nan).ffill().bfill().rolling(
        window=window, center=True, min_periods=1
    ).mean().to_numpy()
    return np.vectorize(wrap_angle)(yaw_sm)

raw_yaw = {}
prev_fid = None
skipped_heading_meta = 0
for fid in range(min_id, max_id+1):
    fp = id2path.get(fid)
    if not fp: continue
    fn = os.path.basename(fp).lower()
    cur = safe_latlon_tmutc(df_meta, fn)
    if cur is None:
        skipped_heading_meta += 1
        continue
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

def get_yaw(fid):
    return float(yaw_sm[fid]) if fid <= max_id else 0.0

# ================= 7) 카메라 모션(호모그래피) 계산 함수 =================
def calculate_camera_motion(img1_gray, img2_gray):
    orb = cv2.ORB_create(nfeatures=1200)
    kps1, des1 = orb.detectAndCompute(img1_gray, None)
    kps2, des2 = orb.detectAndCompute(img2_gray, None)
    if des1 is None or des2 is None or len(kps1) < 20 or len(kps2) < 20:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
    except cv2.error:
        return None
    if len(matches) < 20:
        return None
    good = matches[:120]
    src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None
    inlier_ratio = float(mask.sum()) / len(mask)
    return H if inlier_ratio >= 0.4 else None

# ================= 8) 추적 & 좌표 변환 =================
track_pts = defaultdict(list)  # ext_id 기준
idx = 0
yolo_idx = 0
pr = math.radians(PITCH_DEG)

stats = Counter()
t0 = time.time()

prev_gray = None
ego_motion_H = None

with torch.no_grad():
    torch.backends.cudnn.benchmark = True
    for fid in range(min_id, max_id+1):
        fp = id2path.get(fid)
        if not fp:
            tracker.tracker.predict(ego_H=None)
            stats["skip_no_path"] += 1; continue

        img = cv2.imread(fp)
        if img is None:
            tracker.tracker.predict(ego_H=None)
            stats["skip_imread_fail"] += 1; continue

        fn = os.path.basename(fp).lower()
        meta = safe_latlon_tmutc(df_meta, fn)
        if meta is None:
            tracker.tracker.predict(ego_H=None)
            stats["skip_no_meta"] += 1; continue

        cam_lat, cam_lon, tmutc = meta[0], meta[1], meta[2]
        yaw = get_yaw(fid)

        # --- 에고 모션 계산 ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            ego_motion_H = calculate_camera_motion(prev_gray, gray)
        prev_gray = gray

        # YOLO 추론
        pred = model(img, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, half=half_flag, verbose=False)
        det = pred[0]
        ann = img.copy()
        drew_any = False

        if det is not None and hasattr(det, "boxes") and len(det):
            h, w = img.shape[:2]
            fx = fpx(FX_MM, SENSOR_W_MM, w)
            fy = fpx(FY_MM, SENSOR_H_MM, h)
            cx_pix = CX_MM / SENSOR_W_MM * w
            img_rel_original = os.path.relpath(fp, start=project_root)

            boxes_xyxy = det.boxes.xyxy.detach().cpu().numpy()
            confs       = det.boxes.conf.detach().cpu().numpy().astype(float).ravel()
            clss        = det.boxes.cls.detach().cpu().numpy().astype(int)
            boxes_cxcywh = np.array([xyxy_to_cxcywh(b) for b in boxes_xyxy], dtype=float)

            # YOLO-only CSV
            for box, cid in zip(boxes_xyxy, clss):
                cname = model.names[cid] if (0 <= cid < len(model.names)) else str(cid)
                if cname in EXCLUDE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, map(round, box))
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w - 1, x2), min(h - 1, y2)
                bw, bh = x2c - x1c, y2c - y1c
                if bw <= 0 or bh <= 0: continue
                cx_p = x1c + bw // 2
                cy_p = y1c + bh // 2
                area = bw * bh
                Hs = CLASS_SIGN_HEIGHT.get(cname, DEFAULT_SIGN_H)

                z = Hs * fy / max(bh, 1) * math.cos(pr)
                horiz = math.atan((cx_p - cx_pix) / max(fx, 1e-6))
                bear = wrap_angle(yaw + horiz)
                flat = max(z**2 - (Hs - CAM_HEIGHT_M)**2, 0.0)
                gdist = math.sqrt(flat)
                s_lat, s_lon = offset(cam_lat, cam_lon, gdist, bear)

                yolo_csv_w.writerow([
                    yolo_idx, img_rel_original, cid, cname, -1,
                    cam_lon, cam_lat, s_lon, s_lat,
                    cx_p, cy_p, bw, bh, area, tmutc
                ])
                yolo_idx += 1

            # DeepSORT 업데이트 (에고 모션 전달)
            outs = tracker.update(boxes_cxcywh, confs, clss, img, ego_H=ego_motion_H)

            # === outs: [x1,y1,x2,y2, ext_id, cls] ===
            for (x1, y1, x2, y2, ext_id, cid) in outs:
                ext_id = int(ext_id); cid = int(cid)
                cname = model.names[cid] if (0 <= cid < len(model.names)) else str(cid)
                if cname in EXCLUDE_CLASSES:
                    stats["skip_excluded"] += 1; continue

                x1c, y1c = max(0, int(round(x1))), max(0, int(round(y1)))
                x2c, y2c = min(w-1, int(round(x2))), min(h-1, int(round(y2)))
                bw, bh = x2c - x1c, y2c - y1c
                if bw <= 0 or bh <= 0:
                    continue

                # 시각화: 측정 박스 + 외부 ID
                cv2.rectangle(ann, (x1c,y1c), (x2c,y2c), (0,255,0), 2)
                if SHOW_DEBUG_TEXT:
                    cv2.putText(ann, f"id{ext_id}:{cname}", (x1c, max(0,y1c-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,255,0), 2)
                drew_any = True

                # 월드좌표 산출
                cx_p = x1c + bw // 2
                cy_p = y1c + bh // 2
                area = bw * bh
                Hs = CLASS_SIGN_HEIGHT.get(cname, DEFAULT_SIGN_H)

                z = Hs * fy / max(bh, 1) * math.cos(pr)
                horiz = math.atan((cx_p - cx_pix) / max(fx, 1e-6))
                bear = wrap_angle(yaw + horiz)
                flat = max(z**2 - (Hs - CAM_HEIGHT_M)**2, 0.0)
                gdist = math.sqrt(flat)
                s_lat, s_lon = offset(cam_lat, cam_lon, gdist, bear)

                # 크롭/CSV
                crop = img[y1c:y2c, x1c:x2c]
                cfile = f"{idx:05d}_{fn}"
                cv2.imwrite(os.path.join(close_dir, cfile), crop)

                img_rel_original = os.path.relpath(fp, start=project_root)
                csv_w.writerow([
                    idx, img_rel_original, cid, cname, ext_id,
                    cam_lon, cam_lat, s_lon, s_lat,
                    cx_p, cy_p, bw, bh, area, tmutc
                ])
                stats["saved_rows"] += 1
                track_pts[ext_id].append((s_lon, s_lat))
                idx += 1
        else:
            tracker.tracker.predict(ego_H=ego_motion_H)
            stats["skip_no_detection"] += 1

        # 어노테이션 이미지 저장
        if drew_any:
            anno_path = os.path.join(annot_dir, os.path.basename(fp))
            if os.path.exists(anno_path):
                stem, ext = os.path.splitext(os.path.basename(fp))
                anno_path = os.path.join(annot_dir, f"{stem}_{idx}{ext}")
            cv2.imwrite(anno_path, ann)
            stats["saved_ann"] += 1
        else:
            stats["ann_skipped_empty"] += 1

# --- 파일 핸들러 닫기 ---
w_csv.close()
yolo_w_csv.close()

# ================= 9) 트랙 중앙값 CSV =================
with open(center_out,"w",newline="", encoding="utf-8") as f:
    cw = csv.writer(f)
    cw.writerow(["track_id","sign_lon_med","sign_lat_med","n_frames_kept"])
    for ext_id, pts in track_pts.items():
        res = robust_track_center(pts)
        if res is None: continue
        lon_m, lat_m, n_kept = res
        cw.writerow([ext_id, lon_m, lat_m, n_kept])

# ================= 10) 요약 =================
elapsed = time.time() - t0
print("\n" + "—"*60)
print("                  🚀 처리 완료! 🚀")
print("—"*60)
print(f"✅ YOLO+SORT CSV       → {csv_out}")
print(f"✅ YOLO-Only CSV       → {yolo_csv_out}")
print(f"✅ 트랙 중앙값 CSV     → {center_out}")
print(f"✅ 크롭 이미지         → {close_dir}")
print(f"✅ Annot 이미지        → {annot_dir}")
print(f"⏱️  처리 시간           → {elapsed:.1f}s")
print("—"*60)
print("[SUMMARY] counts:", dict(stats))
print(f"[SUMMARY] yaw 준비 중 메타 누락 프레임: {skipped_heading_meta}")
