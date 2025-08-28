import os, re, json, time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from PIL import Image, ImageFile, UnidentifiedImageError
import warnings
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from pvlib.location import Location
import time

CKPT = "nvidia/segformer-b0-finetuned-ade-512-512"
INPUT_DIR = Path("/Users/aiqizhang/Desktop/2024")
OUTPUT_DIR = Path("/Users/aiqizhang/Desktop/traffic/gcc_2024")

# Toronto (approx downtown)
TORONTO_LAT = 43.6532
TORONTO_LON = -79.3832

# QC thresholds
TIME_MAX = 16
SZA_MAX_DEG = 85 
EDGE_DENSITY_MIN = 0.05

ROI_MARGINS = {"top": 0.00, "bottom": 0.00, "left": 0.00, "right": 0.00}

RESIZE_SHORT = 768      # 512 for faster, 768 for a bit more detail
MIN_REGION_AREA = 64    # remove tiny masks

# pre-processing for edge-density only
USE_CLAHE = True
USE_WIENER_DEBLUR = False   # requires scikit-image; leave False unless needed
_REF_W, _REF_H = 400, 255

# ignore all the labels at the bottom
_HIDE_RECTS = [
    (55/_REF_W,  65/_REF_H,  "left-bottom"),   # 55×65 in left-bottom corner
    (350/_REF_W, 25/_REF_H,  "right-bottom"),  # 350×25 in right-bottom corner
]

# Filename parsing
FNAME_RE = re.compile(r"^(loc\d+)\.(\d{8}_\d{6})\.jpg$", re.IGNORECASE)

def parse_filename(p: Path):
    m = FNAME_RE.match(p.name)
    if not m: return None, None
    loc = m.group(1)
    dt = datetime.strptime(m.group(2), "%Y%m%d_%H%M%S")  # treated as local Toronto time
    return loc, dt

def month_key(dt: datetime) -> str:
    return dt.strftime("%Y%m")

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def calculate_sza(dt, latitude_deg, longitude_deg, altitude_m=0):
    # Create location object
    site = Location(latitude=latitude_deg, longitude=longitude_deg, altitude=altitude_m)
    solar_position = site.get_solarposition(times=dt)
    sza_deg = solar_position['zenith'].values[0]
    return sza_deg

def load_image(path: Path, resize_short: int | None):
    """
    Robust loader:
      - try PIL normally
      - if truncated, enable tolerant decode
      - fallback to OpenCV imdecode
      - return (None, None) if unreadable
    """
    im = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = Image.open(path)
            im.load()
    except (OSError, UnidentifiedImageError):
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                im = Image.open(path)
                im.load()
        except Exception:
            im = None

    if im is None:
        data = np.fromfile(str(path), dtype=np.uint8)  # handles weird paths better than imread
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)

    if im is None:
        print(f"[WARN] Skipping unreadable image: {path}")
        return None, None

    im = im.convert("RGB")

    W, H = im.size
    return im, (W, H)


def apply_clahe_bgr(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

def wiener_deblur_gray(img_gray, psf_size=5):
    try:
        from skimage.restoration import wiener
    except Exception:
        return img_gray  # silently skip if not available
    psf = np.ones((psf_size, psf_size), np.float32) / (psf_size**2)
    img_f = img_gray.astype(np.float32) / 255.0
    deblurred = wiener(img_f, psf, balance=0.1)
    deblurred = np.clip(deblurred * 255, 0, 255).astype(np.uint8)
    return deblurred

def edge_density(img_bgr) -> float:
    proc = img_bgr.copy()
    if USE_CLAHE:
        proc = apply_clahe_bgr(proc)
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    if USE_WIENER_DEBLUR:
        gray = wiener_deblur_gray(gray, psf_size=5)
    edges = cv2.Canny(gray, 100, 200)
    return float((edges > 0).sum() / edges.size)


def roi_mask(shape_hw, margins=ROI_MARGINS):
    H, W = shape_hw
    # base ROI from margins
    t = int(H * margins.get("top", 0.0))
    b = int(H * margins.get("bottom", 0.0))
    l = int(W * margins.get("left", 0.0))
    r = int(W * margins.get("right", 0.0))
    m = np.zeros((H, W), dtype=bool)
    m[t:H - b if b > 0 else H, l:W - r if r > 0 else W] = True

    # hide fixed rectangles
    for w_frac, h_frac, align in _HIDE_RECTS:
        w_px = max(1, int(round(w_frac * W)))
        h_px = max(1, int(round(h_frac * H)))
        if align == "left-bottom":
            x0, x1 = 0, min(W, w_px)
            y1, y0 = H, max(0, H - h_px)
        elif align == "right-bottom":
            x1, x0 = W, max(0, W - w_px)
            y1, y0 = H, max(0, H - h_px)
        else:
            continue
        m[y0:y1, x0:x1] = False

    return m



# SegFormer-B0 ADE20K
class VegSegmenter:
    def __init__(self, ckpt=CKPT):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(ckpt, use_fast=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(ckpt).eval().to(self.device)
        id2label = self.model.config.id2label
        self.label2id = {v.lower(): int(k) for k, v in id2label.items()}
        self.tree_id = self.label2id.get("tree", None)
        self.grass_id = self.label2id.get("grass", None)

    def segment_grass_tree(self, image_pil: Image.Image, upsample_to=None, fast=True):
        target_hw = upsample_to if upsample_to is not None else image_pil.size[::-1]
        with torch.no_grad():
            inputs = self.processor(images=image_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if self.device == "mps":
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    logits = self.model(**inputs).logits
            else:
                logits = self.model(**inputs).logits
            if fast:
                pred_low = logits.argmax(dim=1)[0]
                if self.device == "mps": torch.mps.synchronize()
                pred = pred_low.detach().cpu().numpy().astype("uint8")
                pred = cv2.resize(pred, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
            else:
                up = torch.nn.functional.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
                pred_t = up.argmax(dim=1)[0]
                if self.device == "mps": torch.mps.synchronize()
                pred = pred_t.detach().cpu().numpy().astype("uint8")
        tree_mask = (pred == self.tree_id) if self.tree_id is not None else np.zeros_like(pred, bool)
        grass_mask = (pred == self.grass_id) if self.grass_id is not None else np.zeros_like(pred, bool)
        return tree_mask, grass_mask

def clean_mask(mask: np.ndarray, min_area=MIN_REGION_AREA, k=3) -> np.ndarray:
    m = (mask.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out > 0


def compute_gcc_stats(image_pil: Image.Image, mask: np.ndarray):
    if mask is None or mask.sum() == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "count": 0}
    rgb = np.array(image_pil).astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    denom = R + G + B
    denom[denom == 0] = np.nan
    gcc = G / denom
    vals = gcc[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "count": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "count": int(vals.size),
    }


def process_all():
    ensure_dir(OUTPUT_DIR)
    seg = VegSegmenter(CKPT)

    files = sorted([
        *INPUT_DIR.glob("**/*.[jJ][pP][gG]"),
        *INPUT_DIR.glob("**/*.[jJ][pP][eE][gG]"),
    ])
    files = [p for p in files if FNAME_RE.match(p.name)]

    last_idx = {}
    for i, p in enumerate(files):
        loc_i, dt_i = parse_filename(p)
        if loc_i is None:
            continue
        key_i = (loc_i, month_key(dt_i))
        last_idx[key_i] = i

    buckets     = defaultdict(list) 
    start_times = {}

    for i, img_path in enumerate(files):
        loc, dt = parse_filename(img_path)
        # if dt.month != 12:
        #     continue
        if loc is None:
            continue

        # optional stop-by-location (comment out if not needed)
        # loc_num = int(loc.replace("loc", ""))
        # if loc_num < 8192:
        #     continue

        hour = dt.hour
        sza = calculate_sza(dt, TORONTO_LAT, TORONTO_LON) # SZA + time check

        bgr_full = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr_full is None:
            continue
        Hf, Wf = bgr_full.shape[:2]
        roi_full = roi_mask((Hf, Wf), ROI_MARGINS)
        ys, xs = np.where(roi_full)
        if ys.size > 0:
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            bgr_roi = bgr_full[y0:y1, x0:x1].copy()
        else:
            bgr_roi = bgr_full
        ed = edge_density(bgr_roi)

        qc_pass = (sza <= SZA_MAX_DEG) and (ed >= EDGE_DENSITY_MIN) and (hour <= TIME_MAX) #edge density check

        image_pil, _ = load_image(img_path, RESIZE_SHORT)
        if image_pil is None:
            continue
        H, W = image_pil.size[1], image_pil.size[0]

        if qc_pass:
            tree_mask, grass_mask = seg.segment_grass_tree(image_pil, upsample_to=(H, W), fast=True)
            roi_small = roi_mask((H, W), ROI_MARGINS)
            tree_mask  = clean_mask(tree_mask  & roi_small)
            grass_mask = clean_mask(grass_mask & roi_small)
            stats_tree  = compute_gcc_stats(image_pil, tree_mask)
            stats_grass = compute_gcc_stats(image_pil, grass_mask)
        else:
            stats_tree  = {"mean": None, "std": None, "min": None, "max": None, "count": 0}
            stats_grass = {"mean": None, "std": None, "min": None, "max": None, "count": 0}

        rec = {
            "location": loc,
            "datetime": dt.isoformat(),
            "GCC_tree":  stats_tree,
            "GCC_grass": stats_grass,
            "qc": {
                "sza_deg": round(sza, 2),
                "edge_density": round(ed, 6),
                "passed": bool(qc_pass),
                "sza_max": SZA_MAX_DEG,
                "edge_min": EDGE_DENSITY_MIN
            }
        }

        # accumulate place month & location (rolling bucket)
        ym  = month_key(dt)
        key = (loc, ym)

        if key not in start_times:
            start_times[key] = time.perf_counter()

        buckets[key].append(rec)

        # save last 
        if i == last_idx.get(key):
            out_dir = OUTPUT_DIR / loc
            ensure_dir(out_dir)
            out_path = out_dir / f"{loc}_{ym}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(buckets[key], f, ensure_ascii=False, indent=2)
            elapsed = time.perf_counter() - start_times[key]
            print(f"Saved {len(buckets[key])} records → {out_path}  (elapsed {elapsed:.2f}s)")
            # free memory for this finished month
            del buckets[key]
            del start_times[key]



if __name__ == "__main__":
    t0 = time.perf_counter()
    process_all()
    print(f"Done in {(time.perf_counter()-t0):.2f}s")
