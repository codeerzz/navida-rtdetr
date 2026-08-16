#!/usr/bin/env python3
"""
test_buoy_folder.py  --  Klasördeki duba fotoğraflarını RT-DETR + YCrCb renk
sınıflandırıcısıyla toplu test et.

Her görüntü için:
  • RT-DETR ile nesne tespiti  (bounding box + class + confidence)
  • Her tespit edilen kutu için YCrCb renk sınıflandırması (kırmızı / yeşil / siyah)
  • --no-model: RT-DETR olmadan, YCrCb renk maskesiyle renkli bölgeleri bul ve
    her birini bounding box içine al (en büyük N bileşen, contour tabanlı)
  • Annotated görüntüyü output klasörüne kaydet
  • Terminal + test_raporu.txt + tespitler.csv (+ tespitler.json opsiyonel)

Renk tespiti iyileştirmeleri (bu script özelinde, color_classifier.py'ye dokunmadan):
  • Ön işleme zinciri: gray-world beyaz dengesi → CLAHE (sadece Y kanalı) →
    her karede otomatik uygulanır (--no-white-balance / --no-clahe ile kapatılır).
  • Glare (parlama) maskeleme: Y çok yüksekse (specular highlight) o piksel
    hiçbir rengin maskesine girmez (--no-glare-mask ile kapatılır).
  • Siyah: sadece Y-only değil, Y düşük VE Cr/Cb nötre yakın (su gölgesini eler).
  • Kırmızı: YCrCb belirsiz/kırmızıysa HSV paralel kırmızı bandı ek kanıt/kurtarma
    olarak devreye girer (var olan davranış, değişmedi).
  • Yeşil: YCrCb VE HSV ikisi de "yeşil" demedikçe kabul edilmez (AND oylama) --
    tek başına YCrCb'nin yeşile benzeyen su/yosun yanlış pozitiflerini eler.

Benchmark (etiketsiz görsellerle):
  Görselleri dosya adına renk adını (kirmizi/yesil/siyah ya da red/green/black,
  herhangi bir yerinde) geçecek şekilde yeniden adlandır, örn:
    kirmizi_duba_01.jpg, duba_yesil_14.png, siyah_03.jpeg
  Script her görüntü adından "beklenen" rengi otomatik çıkarır, en büyük kutunun
  rengiyle karşılaştırır ve confusion matrix + accuracy + hatalı dosya listesini
  benchmark_raporu.txt'ye yazar. Elle etiketleme gerekmez.

Kullanım:
  # RT-DETR modeli ile (önerilen)
  python3 scripts/test_buoy_folder.py \\
      --weights /path/to/best.pt \\
      --input duba_fotolari/

  # Model olmadan — renk maskesiyle bbox (ultralytics gerekmez)
  python3 scripts/test_buoy_folder.py \\
      --no-model \\
      --input duba_fotolari/

Bağımlılıklar:
  pip install ultralytics opencv-python pyyaml numpy
  --no-model modunda ultralytics gerekmez: pip install opencv-python pyyaml numpy
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# color_classifier'ı colcon/pip kurulumu olmadan da import edebilmek için
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PACKAGE_ROOT = _SCRIPT_DIR.parent           # rtdetr_zed_tracker/
sys.path.insert(0, str(_PACKAGE_ROOT))

from rtdetr_zed_tracker.color_classifier import (  # noqa: E402
    DEFAULT_MIN_CONFIDENCE, load_color_ranges, threshold_for,
)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
DEFAULT_RANGES = _PACKAGE_ROOT / "config" / "color_ranges.yaml"
DEFAULT_LABELS = _PACKAGE_ROOT / "config" / "class_labels.yaml"

# Bounding box kenarlık renkleri (BGR)
_BOX_BGR: dict[str | None, tuple] = {
    "red":    (30,  30, 220),
    "green":  (30, 200,  50),
    "black":  (80,  80,  80),
    None:     (0,  200, 255),   # belirsiz → sarı-turuncu
}

# Desteklenen uzantılar
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ---------------------------------------------------------------------------
# Veri sınıfları
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    class_id: int
    class_name: str
    det_conf: float
    bbox_xyxy: tuple[int, int, int, int]
    color_label: Optional[str]
    color_conf: float
    color_ratios: dict[str, float] = field(default_factory=dict)


@dataclass
class ImageResult:
    path: Path
    detections: list[Detection] = field(default_factory=list)
    error: Optional[str] = None
    inference_ms: float = 0.0
    color_ms: float = 0.0
    expected_color: Optional[str] = None    # dosya adından çıkarılan (benchmark)
    predicted_color: Optional[str] = None   # en büyük kutunun rengi (benchmark)

    @property
    def n_buoys(self) -> int:
        return len(self.detections)

    @property
    def color_summary(self) -> dict[str, int]:
        counts = {"red": 0, "green": 0, "black": 0, "uncertain": 0}
        for d in self.detections:
            key = d.color_label if d.color_label in counts else "uncertain"
            counts[key] += 1
        return counts


# ---------------------------------------------------------------------------
# class_labels.yaml yükleme
# ---------------------------------------------------------------------------

def load_labels(path: Path) -> dict[int, str]:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return {int(k): str(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Model yükleme (RT-DETR ya da YOLO -- checkpoint'ten otomatik ayırt edilir)
# ---------------------------------------------------------------------------

def _detect_model_type(weights_path: str) -> str:
    """.pt checkpoint'inin RT-DETR mi yoksa YOLO mimarisiyle mi eğitildiğini
    checkpoint'in kendi metadata'sından anlar.

    Yanlış sınıfla yüklemek (ör. düz bir YOLO .pt'yi ultralytics.RTDETR() ile
    açmak) import/yükleme aşamasında HİÇBİR hata vermez -- model "yüklenir",
    ama RT-DETR'in postprocessing'i (sabit sayıda query bekleyen decoder çıkışı)
    YOLO'nun anchor-free grid çıkışıyla (640 girdide 8400 hücre) uyuşmadığı için
    HER görüntüde 'split_with_sizes expects ... but got split_sizes=[4, 1, 1]'
    hatası fırlatır -- hata mesajı hangi mimarinin yanlış seçildiğini söylemez,
    üstüne script içindeki except bunu yakalayıp o görüntüyü sessizce atlar, o
    yüzden sonuç "hiç annotated görsel çıkmadı" gibi görünür. Bunu önlemek için
    otomatik ayırt ediyoruz.
    """
    try:
        import torch
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        model = ckpt.get("model") if isinstance(ckpt, dict) else None
        cls_name = type(model).__name__ if model is not None else ""
        return "rtdetr" if "RTDETR" in cls_name else "yolo"
    except Exception:
        return "yolo"  # tespit edilemezse en yaygın olan (YOLO) varsayılır


def load_model(weights_path: str, model_type: str = "auto"):
    kind = model_type if model_type != "auto" else _detect_model_type(weights_path)
    print(f"  Model mimarisi: {kind}"
          f"{' (checkpoint metadata’sından otomatik tespit)' if model_type == 'auto' else ' (--model-type ile belirtildi)'}")
    try:
        if kind == "rtdetr":
            from ultralytics import RTDETR
            return RTDETR(weights_path)
        from ultralytics import YOLO
        return YOLO(weights_path)
    except ImportError:
        print("HATA: ultralytics kurulu değil → pip install ultralytics", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"HATA: Model yüklenemedi ({weights_path}): {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Ön işleme: gray-world beyaz dengesi + CLAHE (sadece Y kanalı)
# ---------------------------------------------------------------------------
# Kare gelir gelmez uygulanır (RT-DETR çıkarımından ve renk sınıflandırmasından
# ÖNCE) -- amaç, aşağı akıştaki her adımın zaten düzeltilmiş bir kare görmesi.

def _gray_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    """Gray-world varsayımıyla kanal ortalamalarını eşitler.

    Deniz ortamında baskın mavi-yeşil renk cast'i kırmızıyı bastırır (kamera
    otomatik beyaz dengesi suyun rengini "nötr" sanıp kırmızıyı söndürür).
    Her kanalı, üç kanalın ortalaması ortak bir gri değere denk gelecek şekilde
    ölçekleyerek bu cast'i giderir -- basit ama düşük ışıkta/su altında sağlam.
    """
    b, g, r = cv2.split(bgr.astype(np.float32))
    b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
    gray_mean = (b_mean + g_mean + r_mean) / 3.0
    b *= gray_mean / max(b_mean, 1e-6)
    g *= gray_mean / max(g_mean, 1e-6)
    r *= gray_mean / max(r_mean, 1e-6)
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)


def _clahe_y_channel(bgr: np.ndarray, clip_limit: float = 2.0,
                     tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE'yi SADECE Y (parlaklık) kanalına uygular, Cr/Cb dokunulmadan kalır.

    Düşük ışıkta kırmızı/yeşil arasındaki fark Cr/Cb'de zaten var ama Y çok
    düşükken piksel değerleri sıkışıp ayırt edilmesi zorlaşır. Y'yi yerel
    kontrastla açmak bunu düzeltir; Cr/Cb'ye dokunmamak renk kararının bu
    ön işlemden etkilenmemesini garanti eder.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    y_eq = clahe.apply(y)
    return cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)


def preprocess_frame(bgr: np.ndarray, use_white_balance: bool = True,
                     use_clahe: bool = True, clahe_clip_limit: float = 2.0,
                     clahe_tile: int = 8) -> np.ndarray:
    """Kare gelince ilk uygulanan ön işleme zinciri: WB → CLAHE."""
    out = bgr
    if use_white_balance:
        out = _gray_world_white_balance(out)
    if use_clahe:
        out = _clahe_y_channel(out, clip_limit=clahe_clip_limit,
                               tile_grid=(clahe_tile, clahe_tile))
    return out


# ---------------------------------------------------------------------------
# Renk sınıflandırma yardımcısı
# ---------------------------------------------------------------------------
# color_classifier.classify_color'ı bu script özelinde iki şekilde sarar
# (color_classifier.py'ye dokunmadan -- kapsam bu test scriptiyle sınırlı):
#   1) Glare (parlama) maskeleme: Y çok yüksek pikseller hiçbir rengin
#      maskesine girmez VE payda (toplam piksel) hesabından da çıkarılır --
#      aksi halde parlak pikseller Cr/Cb'yi bozar ve yanlış rengi "kazanır".
#   2) Kırmızı için HSV OR (kurtarma+onay), yeşil için HSV AND (sadece onay) --
#      bkz. classify_buoy_color.

def classify_color_glare_aware(bgr_crop: np.ndarray, color_ranges: dict,
                               roi_shrink: float = 0.6,
                               min_confidence=DEFAULT_MIN_CONFIDENCE,
                               glare_y_thresh: int = 245):
    """classify_color ile aynı algoritma, tek fark: Y >= glare_y_thresh olan
    pikseller (specular highlight/parlama) hesaba hiç katılmaz."""
    if bgr_crop is None or bgr_crop.size == 0 or not color_ranges:
        return None, 0.0, {}

    h, w = bgr_crop.shape[:2]
    bw, bh = w * roi_shrink, h * roi_shrink
    x1 = int(max(0, round((w - bw) / 2)));  y1 = int(max(0, round((h - bh) / 2)))
    x2 = int(min(w, round(x1 + bw)));       y2 = int(min(h, round(y1 + bh)))
    roi = bgr_crop[y1:y2, x1:x2]
    if roi.size == 0:
        return None, 0.0, {}

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    valid = ycrcb[:, :, 0] < glare_y_thresh
    total = int(valid.sum())
    if total == 0:
        # ROI'nin tamamı parlama -- güvenilir bir okuma yok.
        return None, 0.0, {}

    ratios: dict[str, float] = {}
    for color, ranges in color_ranges.items():
        mask = np.zeros(ycrcb.shape[:2], dtype=bool)
        for y_min, y_max, cr_min, cr_max, cb_min, cb_max in ranges:
            lo = np.array([y_min, cr_min, cb_min])
            hi = np.array([y_max, cr_max, cb_max])
            mask |= cv2.inRange(ycrcb, lo, hi).astype(bool)
        mask &= valid
        ratios[color] = float(mask.sum()) / total

    best_color = max(ratios, key=ratios.get)
    best_ratio = ratios[best_color]
    label = best_color if best_ratio >= threshold_for(best_color, min_confidence) else None
    return label, best_ratio, ratios


def _hsv_green_ratio(bgr_crop: np.ndarray) -> float:
    """HSV uzayında yeşil piksel oranı. YCrCb ile AND oylamak için kullanılır --
    tek başına YCrCb'nin yeşile benzeyen su/yosun yanlış pozitiflerini eler.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return 0.0
    lo_s, lo_v = 60, 40
    mask = cv2.inRange(hsv, (35, lo_s, lo_v), (85, 255, 255))  # yeşil Hue bandı
    return float(cv2.countNonZero(mask)) / total


def classify_buoy_color(bgr_crop: np.ndarray, color_ranges: dict, roi_shrink: float,
                        min_confidence=DEFAULT_MIN_CONFIDENCE,
                        glare_y_thresh: int = 245,
                        use_hsv_red: bool = True, hsv_red_min_ratio: float = 0.10,
                        use_hsv_green: bool = True, hsv_green_min_ratio: float = 0.10):
    """classify_color_glare_aware + iki uzay (YCrCb/HSV) oylaması.

    Kırmızı: YCrCb belirsizse ya da kırmızı derse, HSV'nin paralel kırmızı
    bandı ek kanıt/kurtarma olarak devreye girer (OR benzeri) -- kırmızı buldozer
    parlamada beyaza doğru soluyunca YCrCb tek başına kaçırabiliyor.
    Yeşil: YCrCb yeşil dese BİLE HSV onaylamadıkça kabul edilmez (AND) -- yeşile
    benzeyen su/yosun YCrCb'de tek başına yanlış pozitif üretebiliyor, iki
    bağımsız uzayın ikisinin de aynı fikirde olmasını istemek bunu düşürür.
    """
    label, conf, ratios = classify_color_glare_aware(
        bgr_crop, color_ranges, roi_shrink, min_confidence, glare_y_thresh)
    base_colors = set(ratios)  # henüz *_hsv anahtarları eklenmeden önceki gerçek renk adları

    if use_hsv_red and (label == "red" or label is None):
        hsv_ratio = _hsv_red_ratio(bgr_crop)
        if hsv_ratio >= hsv_red_min_ratio:
            if label is None:
                label, conf = "red", hsv_ratio
                ratios = {**ratios, "red_hsv": round(hsv_ratio, 3)}
            else:
                conf = (conf + hsv_ratio) / 2.0
                ratios = {**ratios, "red_hsv": round(hsv_ratio, 3)}

    if use_hsv_green and label == "green":
        hsv_ratio = _hsv_green_ratio(bgr_crop)
        ratios = {**ratios, "green_hsv": round(hsv_ratio, 3)}
        if hsv_ratio < hsv_green_min_ratio:
            # AND oylaması yeşili reddetti -- ama bu "hiçbir renk yok" demek
            # DEĞİL, sadece "en yüksek YCrCb oranı yanlış renkteydi" demek.
            # Su/yosun karışımı YCrCb'de yeşili öne çıkarmış olabilir ama asıl
            # renk (ör. kırmızı) kendi eşiğini zaten geçiyor olabilir --
            # reddedilen yeşili doğrudan "belirsiz"e düşürmeden önce ikinci
            # sıradaki adayı (kendi eşiğini kendi başına geçiyorsa) dene.
            runner_up = max((c for c in base_colors if c != "green"),
                            key=lambda c: ratios[c], default=None)
            if runner_up is not None and ratios[runner_up] >= threshold_for(runner_up, min_confidence):
                label, conf = runner_up, ratios[runner_up]
            else:
                label, conf = None, 0.0
        else:
            conf = (conf + hsv_ratio) / 2.0

    return label, conf, ratios


def _classify_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                   color_ranges: dict, roi_shrink: float, **hsv_kwargs):
    """Bir bbox kırpımını renk sınıflandır. (label, confidence, ratios) döner."""
    x1c = max(0, x1);  y1c = max(0, y1)
    x2c = min(frame.shape[1], x2);  y2c = min(frame.shape[0], y2)
    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None, 0.0, {}
    return classify_buoy_color(crop, color_ranges, roi_shrink, **hsv_kwargs)


# ---------------------------------------------------------------------------
# --no-model: Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

# ── Problem 2: HSV paralel kırmızı kontrolü ────────────────────────────────

def _hsv_red_ratio(bgr_crop: np.ndarray) -> float:
    """HSV uzayında kırmızı piksel oranını döner.

    Kırmızı, HSV'de Hue=0° ve Hue=180° çevresinde iki ayrı bant oluşturur
    (renk çarkının sarmal yapısı). YCrCb'de yakalanmayan parlak/soluk kırmızı
    tonları buradan yakalanabilir.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return 0.0
    hsv   = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return 0.0
    # Düşük doygunluk/parlaklık piksellerini ele (beyaz, gri, siyah hariç)
    lo_s, lo_v = 60, 40
    mask1 = cv2.inRange(hsv, (0,   lo_s, lo_v), (10,  255, 255))  # sıcak kırmızı
    mask2 = cv2.inRange(hsv, (165, lo_s, lo_v), (180, 255, 255))  # soğuk kırmızı
    red_px = cv2.countNonZero(cv2.bitwise_or(mask1, mask2))
    return float(red_px) / total


# ── Problem 2: Adaptive confidence threshold ───────────────────────────────

def _adaptive_thresholds(frame: np.ndarray,
                         color_ranges: dict,
                         base_conf: float = 0.12,
                         background_ratio_limit: float = 0.30) -> dict[str, float]:
    """Görüntünün global renk dağılımına göre renk başına eşik hesapla.

    Mantık: bir renk görüntünün büyük kısmını kaplıyorsa (ör. su = yeşil),
    o rengin eşiğini yükselt — böylece yetersiz kanıtla label atanması önlenir.
    Oranı az olan renkler için eşiği hafifçe *düşür* (daha duyarlı ol).

    background_ratio_limit: görüntünün bu oranından fazlası bir rengin
    maskesindeyse o renk "arkaplan baskın" sayılır ve eşiği 2× artar.
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    total = frame.shape[0] * frame.shape[1]
    thresholds: dict[str, float] = {}

    for color, ranges in color_ranges.items():
        m = np.zeros(ycrcb.shape[:2], dtype=np.uint8)
        for y_min, y_max, cr_min, cr_max, cb_min, cb_max in ranges:
            lo = np.array([y_min, cr_min, cb_min], dtype=np.uint8)
            hi = np.array([y_max, cr_max, cb_max], dtype=np.uint8)
            m  = cv2.bitwise_or(m, cv2.inRange(ycrcb, lo, hi))

        global_ratio = float(cv2.countNonZero(m)) / total if total else 0.0

        if global_ratio > background_ratio_limit:
            # Bu renk görüntüye hâkim → arkaplan büyük ihtimalle bu renk → eşiği yükselt
            factor = 1.0 + (global_ratio - background_ratio_limit) * 3.0
            thresholds[color] = min(0.90, base_conf * factor)
        else:
            # Görüntüde az → duba olabilir → hafifçe düşür
            thresholds[color] = max(0.05, base_conf * 0.8)

    return thresholds


# ── Problem 3: Y-only siyah maskesi ────────────────────────────────────────

def _black_mask_y_only(ycrcb_img: np.ndarray, y_max: int = 45,
                       cr_tol: int = 15, cb_tol: int = 15) -> np.ndarray:
    """Düşük parlaklık (Y < y_max) VE nötr Cr/Cb ile siyah maskesi üret.

    Saf Y-only riskli: gölgedeki su da düşük Y'ye girer, ama genelde nötr
    DEĞİLDİR -- denizin mavi-yeşil rengi Cr/Cb'yi 128'den (nötr gri noktası)
    belirgin şekilde kaydırır. "Y düşük VE |Cr-128|,|Cb-128| küçük" şartını
    birlikte aramak su gölgesini eler, gerçek siyah dubayı (nötr renkli,
    boyanın kendisi neredeyse gri) yakalamaya devam eder.
    """
    y_ch = ycrcb_img[:, :, 0].astype(np.int16)
    cr_ch = ycrcb_img[:, :, 1].astype(np.int16)
    cb_ch = ycrcb_img[:, :, 2].astype(np.int16)
    is_dark = y_ch < y_max
    is_neutral = (np.abs(cr_ch - 128) < cr_tol) & (np.abs(cb_ch - 128) < cb_tol)
    return np.where(is_dark & is_neutral, np.uint8(255), np.uint8(0))


# ── Problem 3: Convex hull bbox ─────────────────────────────────────────────

def _hull_bbox(contours_list: list) -> tuple[int, int, int, int]:
    """Birden fazla contour'u convex hull ile birleştir → tek bbox döner.

    Parçalı maskelerde (siyah dubada gürültü küçük bölgelere böler) tüm
    parçaların dış kabuğunu tek seferde saran en küçük dik dörtgeni verir.
    Yalnız bir contour varsa da düzgün çalışır.
    """
    all_pts = np.vstack(contours_list)
    hull    = cv2.convexHull(all_pts)
    x, y, w, h = cv2.boundingRect(hull)
    return x, y, x + w, y + h


# ── Problem 1 (Yeşil): Renk Doluluk Oranı (Color Fill Ratio) ───────────────

def _color_fill_ratio(crop_bgr: np.ndarray,
                      color_ranges: dict,
                      color: str) -> float:
    """Bbox crop'u içinde `color` maskesine giren piksellerin oranını döner.

    Bir duba bbox'ını neredeyse tamamen doldurur → yüksek fill ratio.
    Su yansıması, arka plan → dağınık seyrek pikseller → düşük fill ratio.
    Şekil hakkında hiçbir varsayım yoktur; sadece renk yoğunluğu ölçülür.
    """
    if crop_bgr is None or crop_bgr.size == 0 or color not in color_ranges:
        return 0.0
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    total = ycrcb.shape[0] * ycrcb.shape[1]
    if total == 0:
        return 0.0
    m = np.zeros(ycrcb.shape[:2], dtype=np.uint8)
    for y_min, y_max, cr_min, cr_max, cb_min, cb_max in color_ranges[color]:
        lo = np.array([y_min, cr_min, cb_min], dtype=np.uint8)
        hi = np.array([y_max, cr_max, cb_max], dtype=np.uint8)
        m  = cv2.bitwise_or(m, cv2.inRange(ycrcb, lo, hi))
    return float(cv2.countNonZero(m)) / total


# ── Problem 1 (Yeşil): HSV Doygunluk Filtresi ───────────────────────────────

def _hsv_median_saturation(crop_bgr: np.ndarray) -> float:
    """Crop içindeki medyan HSV doygunluğunu (S kanalı, 0-255) döner.

    Boyalı duba yüzeyi yüksek doygunluğa sahiptir (S ≈ 120-220).
    Su yüzeyi soluk, yansımalı → düşük doygunluk (S ≈ 10-70).
    Gölgedeki duba da düşük olabilir; dolayısıyla bu filtre tek başına değil
    fill_ratio ile birlikte kullanılmalıdır.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[:, :, 1]))


# ── YCrCb renk maskeleri ────────────────────────────────────────────────────

def _color_mask_for(ycrcb_img: np.ndarray, color_ranges: dict,
                    glare_y_thresh: Optional[int] = None) -> dict[str, np.ndarray]:
    """Her renk için ikili maske (uint8, 255=içinde) döner.

    glare_y_thresh verilirse, Y >= glare_y_thresh olan pikseller (parlama)
    HİÇBİR rengin maskesine girmez -- parlak pikseller Cr/Cb'yi bozar ve
    contour/bbox bulma aşamasında yanlış renge "oy" verir.
    """
    glare = ycrcb_img[:, :, 0] >= glare_y_thresh if glare_y_thresh is not None else None
    masks = {}
    for color, ranges in color_ranges.items():
        m = np.zeros(ycrcb_img.shape[:2], dtype=np.uint8)
        for y_min, y_max, cr_min, cr_max, cb_min, cb_max in ranges:
            lo = np.array([y_min, cr_min, cb_min], dtype=np.uint8)
            hi = np.array([y_max, cr_max, cb_max], dtype=np.uint8)
            m  = cv2.bitwise_or(m, cv2.inRange(ycrcb_img, lo, hi))
        if glare is not None:
            m[glare] = 0
        masks[color] = m
    return masks


# ── Ana no-model tespit fonksiyonu ──────────────────────────────────────────

def _find_color_boxes(frame: np.ndarray,
                      color_ranges: dict,
                      min_area_px: int = 500,
                      max_boxes: int = 10,
                      morph_k: int = 7,
                      use_hsv_red: bool = True,
                      use_adaptive_thresh: bool = True,
                      use_y_only_black: bool = True,
                      hsv_red_min_ratio: float = 0.10,
                      y_only_black_max: int = 45,
                      y_only_black_cr_tol: int = 15,
                      y_only_black_cb_tol: int = 15,
                      black_close_k: int = 25,
                      use_green_fill_filter: bool = True,
                      green_fill_min: float = 0.35,
                      use_green_sat_filter: bool = True,
                      green_sat_min: float = 60.0,
                      use_glare_mask: bool = True,
                      glare_y_thresh: int = 245,
                      use_hsv_green: bool = True,
                      hsv_green_min_ratio: float = 0.10) -> list[Detection]:
    """
    Renk maskesi → contour → bbox pipeline.

    Yeni özellikler (sadece bu test scriptinde, color_classifier.py'ye dokunulmadı):

    use_hsv_red          : Kırmızı için YCrCb'ye ek HSV kontrolü (OR/kurtarma).
    use_adaptive_thresh  : Görüntü bazlı per-renk confidence eşiği.
    use_y_only_black     : Siyah için Y < eşik VE Cr/Cb nötre yakın maskesi.
    y_only_black_cr_tol/cb_tol: nötr bandın Cr/Cb toleransı (|kanal-128| < tol).
    black_close_k        : Siyah closing kernel boyutu.
    use_green_fill_filter: Yeşil bbox'ı, içindeki yeşil piksel doluluk oranına
                           göre filtrele. Düşük doluluk → su/arka plan → reddet.
    green_fill_min       : Minimum doluluk oranı eşiği (0–1).
    use_green_sat_filter : Yeşil bbox'ı, medyan HSV doygunluğuna göre filtrele.
                           Düşük doygunluk → su/soluk yüzey → reddet.
    green_sat_min        : Minimum medyan HSV S eşiği (0–255).
    use_glare_mask       : Y >= glare_y_thresh pikselleri tüm renk maskelerinden çıkar.
    use_hsv_green        : Yeşil için YCrCb + HSV AND oylaması (sadece onay, kurtarma yok).
    """
    ycrcb  = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    masks  = _color_mask_for(ycrcb, color_ranges,
                             glare_y_thresh=glare_y_thresh if use_glare_mask else None)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))

    # Problem 2 — adaptive threshold
    if use_adaptive_thresh:
        adapt_thresh = _adaptive_thresholds(frame, color_ranges)
    else:
        adapt_thresh = {c: 0.12 for c in color_ranges}

    # Problem 3 — Y-only (+ nötr Cr/Cb) siyah maskesini ana maskenin üstüne bindir
    if use_y_only_black and "black" in masks:
        y_mask = _black_mask_y_only(ycrcb, y_max=y_only_black_max,
                                    cr_tol=y_only_black_cr_tol, cb_tol=y_only_black_cb_tol)
        # Orijinal YCrCb siyah maskesiyle birleştir (OR): her iki yaklaşımın
        # bulduğu pikseller dahil edilir
        masks["black"]    = cv2.bitwise_or(masks["black"], y_mask)

    detections: list[Detection] = []

    for color, mask in masks.items():
        is_black = (color == "black")

        # Problem 3 — siyah için daha agresif closing (büyük kernel)
        if is_black and use_y_only_black:
            bk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (black_close_k, black_close_k))
            clean = cv2.morphologyEx(mask,  cv2.MORPH_OPEN,  kernel)
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, bk)
        else:
            clean = cv2.morphologyEx(mask,  cv2.MORPH_OPEN,  kernel)
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours    = sorted(contours, key=cv2.contourArea, reverse=True)

        added = 0
        i     = 0
        while i < len(contours) and added < max_boxes:
            cnt  = contours[i]
            area = cv2.contourArea(cnt)
            i   += 1

            if area < min_area_px:
                break

            # Problem 3 — convex hull bbox (tek contour için de düzgün çalışır)
            x1, y1, x2, y2 = _hull_bbox([cnt])

            crop = frame[max(0,y1):min(frame.shape[0],y2),
                         max(0,x1):min(frame.shape[1],x2)]
            if crop.size == 0:
                continue

            # YCrCb classifier (doğrulama) + glare maskeleme + kırmızı(OR)/yeşil(AND)
            # HSV oylaması -- hepsi classify_buoy_color içinde (bkz. yukarısı).
            final_label, final_conf, final_ratios = classify_buoy_color(
                crop, color_ranges, roi_shrink=0.8,
                min_confidence=adapt_thresh.get(color, 0.12),
                glare_y_thresh=glare_y_thresh if use_glare_mask else 255,
                use_hsv_red=use_hsv_red, hsv_red_min_ratio=hsv_red_min_ratio,
                use_hsv_green=use_hsv_green, hsv_green_min_ratio=hsv_green_min_ratio,
            )

            if final_label is None:
                if color == "green" and use_hsv_green:
                    # Bu kutu yeşil maskesinden geldi ama YCrCb+HSV AND oylaması
                    # yeşili onaylamadı. Diğer renkler gibi maske rengine
                    # (fallback) düşürmek tam da bu oylamanın önlemeye
                    # çalıştığı yanlış pozitif olurdu -- kutuyu tamamen atla.
                    continue
                # final_label hâlâ None ise (kırmızı/siyah) mask rengini fallback olarak kullan
                final_label = color

            # ── Problem 1 (Yeşil): Fill Ratio + Saturation filtresi ────────
            # Yeşil için iki bağımsız filtre: her ikisi de geçmek zorunda.
            # Diğer renkler (kırmızı, siyah) bu filtreden muaf.
            if final_label == "green":
                if use_green_fill_filter:
                    fill = _color_fill_ratio(crop, color_ranges, "green")
                    if fill < green_fill_min:
                        # Düşük doluluk → büyük ihtimalle su/arka plan → atla
                        final_ratios = dict(final_ratios, green_fill=round(fill, 3))
                        continue
                    final_ratios = dict(final_ratios, green_fill=round(fill, 3))

                if use_green_sat_filter:
                    sat = _hsv_median_saturation(crop)
                    if sat < green_sat_min:
                        # Düşük doygunluk → soluk/su rengine benziyor → atla
                        final_ratios = dict(final_ratios, green_sat=round(sat, 1))
                        continue
                    final_ratios = dict(final_ratios, green_sat=round(sat, 1))

            detections.append(Detection(
                class_id=-1,
                class_name="buoy",
                det_conf=min(1.0, area / (frame.shape[0] * frame.shape[1])),
                bbox_xyxy=(x1, y1, x2, y2),
                color_label=final_label,
                color_conf=final_conf,
                color_ratios=final_ratios,
            ))
            added += 1

    detections.sort(
        key=lambda d: (d.bbox_xyxy[2] - d.bbox_xyxy[0]) * (d.bbox_xyxy[3] - d.bbox_xyxy[1]),
        reverse=True,
    )
    return detections


# ---------------------------------------------------------------------------
# RT-DETR modeli ile işleme
# ---------------------------------------------------------------------------

def process_with_model(frame, model, labels, color_ranges, conf_thresh, roi_shrink,
                       **hsv_kwargs):
    t0 = time.perf_counter()
    results = model.predict(frame, conf=conf_thresh, verbose=False)
    t1 = time.perf_counter()

    detections: list[Detection] = []
    boxes = results[0].boxes if results else None

    tc0 = time.perf_counter()
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cls_id   = int(box.cls[0])
            det_conf = float(box.conf[0])
            name     = labels.get(cls_id, f"class_{cls_id}")
            color_label, color_conf, ratios = _classify_crop(
                frame, x1, y1, x2, y2, color_ranges, roi_shrink, **hsv_kwargs
            )
            detections.append(Detection(
                class_id=cls_id, class_name=name, det_conf=det_conf,
                bbox_xyxy=(x1, y1, x2, y2),
                color_label=color_label, color_conf=color_conf,
                color_ratios=ratios,
            ))
    tc1 = time.perf_counter()
    return detections, (t1 - t0) * 1000, (tc1 - tc0) * 1000


# ---------------------------------------------------------------------------
# Annotation çizimi
# ---------------------------------------------------------------------------

def annotate_frame(frame: np.ndarray, detections: list[Detection],
                   no_model: bool = False) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        box_bgr = _BOX_BGR.get(det.color_label, _BOX_BGR[None])

        # Bounding box çiz (her iki modda da)
        cv2.rectangle(out, (x1, y1), (x2, y2), box_bgr, 2)

        color_str = det.color_label if det.color_label else "belirsiz"

        if no_model:
            line1 = f"Renk: {color_str}"
            line2 = f"Conf: {det.color_conf:.2f}"
        else:
            line1 = f"RT-DETR: {det.class_name} ({det.det_conf:.2f})"
            line2 = f"Renk: {color_str} ({det.color_conf:.2f})"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        fscale     = 0.55
        fthick     = 2

        (tw1, th1), _ = cv2.getTextSize(line1, font, fscale, fthick)
        (tw2, th2), _ = cv2.getTextSize(line2, font, fscale, fthick)
        bw_  = max(tw1, tw2) + 10
        bh_  = th1 + th2 + 16
        bx1_, by1_ = x1, max(0, y1 - bh_)
        bx2_, by2_ = min(w, x1 + bw_), y1

        # Yarı saydam arka plan bandı
        overlay = out.copy()
        cv2.rectangle(overlay, (bx1_, by1_), (bx2_, by2_), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

        cv2.putText(out, line1, (x1 + 4, max(th1 + 2, y1 - th2 - 8)),
                    font, fscale, (240, 215, 50), fthick, cv2.LINE_AA)
        cv2.putText(out, line2, (x1 + 4, max(th1 + th2 + 4, y1 - 4)),
                    font, fscale, box_bgr, fthick, cv2.LINE_AA)

    # Sol alt: tespit sayısı + mod
    mode_txt = "no-model (renk maskesi)" if no_model else "RT-DETR"
    cv2.putText(out, f"{mode_txt} | Tespit: {len(detections)}", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Klasör tarama
# ---------------------------------------------------------------------------

def collect_images(folder: Path, exclude_dir: Optional[Path] = None) -> list[Path]:
    """rglob recursive olduğu için, varsayılan çıktı klasörü (<input>/test_ciktisi)
    girdi klasörünün İÇİNDE yaşıyor -- exclude_dir verilmezse ikinci bir
    çalıştırma önceki annotated_*.jpg çıktılarını da "yeni girdi" sanıp
    annotated_annotated_*.jpg üretir ve her koşuda katlanarak çoğalır.

    İki bağımsız koruma birlikte çalışır:
      1) exclude_dir: BU koşunun çıktı klasörünü tamamen tarama dışı bırakır.
      2) "annotated_" ön eki: her koşunun kendi annotate_frame çıktısı hep bu
         önekle yazılır, dolayısıyla ismi bununla başlayan HER dosyayı (farklı
         isimli eski --output klasörlerinden kalanlar dahil, hangi alt klasörde
         olursa olsun) girdi saymayı reddeder. exclude_dir tek başına yeterli
         değil -- kullanıcı her koşuda farklı bir --output adı verirse (ör.
         test_ciktisi_v2, _v3, ...) önceki klasörler hâlâ girdi klasörünün
         içinde kalır ve taranmaya devam eder.
    """
    exclude_resolved = exclude_dir.resolve() if exclude_dir is not None else None
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        and not p.name.startswith("annotated_")
        and (exclude_resolved is None or exclude_resolved not in p.resolve().parents)
    )


# ---------------------------------------------------------------------------
# Benchmark: dosya adından beklenen renk + confusion matrix
# ---------------------------------------------------------------------------
# Elle etiketleme yapmadan basit bir benchmark: görselleri dosya adına renk
# adı geçecek şekilde yeniden adlandır (Türkçe ya da İngilizce, adın herhangi
# bir yerinde -- "kirmizi_duba_01.jpg", "duba_yesil_14.png", "black_03.jpeg"
# hepsi çalışır), script geri kalanını otomatik yapar.

_COLOR_FILENAME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "red":   ("kirmizi", "kırmızı", "red"),
    "green": ("yesil", "yeşil", "green"),
    "black": ("siyah", "black"),
}


def expected_color_from_filename(path: Path) -> Optional[str]:
    """Dosya adında geçen renk anahtar kelimesinden beklenen etiketi çıkar.

    Eşleşme yoksa None döner -- o görüntü benchmark'a dahil edilmez (mevcut
    etiketsiz görsel setiyle geriye dönük uyumlu kalır, sadece yeniden
    adlandırdığın dosyalar puanlanır).
    """
    name = path.stem.lower()
    for color, keywords in _COLOR_FILENAME_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return color
    return None


def predicted_color_for_image(detections: list[Detection]) -> Optional[str]:
    """Görüntü için "birincil" renk tahmini: rengi belirlenmiş en büyük kutu.

    Bir görüntüde birden fazla tespit olabilir (gerçek dubadan büyük olan tek
    bir tanesi olması beklenir); alan bazlı seçim, küçük gürültü kutularının
    benchmark sonucunu bozmasını önler. Hiçbir kutunun rengi belirlenemediyse
    (hepsi "belirsiz") ya da hiç tespit yoksa None döner.
    """
    labeled = [d for d in detections if d.color_label]
    if not labeled:
        return None
    best = max(labeled, key=lambda d: (d.bbox_xyxy[2] - d.bbox_xyxy[0])
                                       * (d.bbox_xyxy[3] - d.bbox_xyxy[1]))
    return best.color_label


def _confusion_matrix_lines(pairs: list[tuple[str, Optional[str]]], colors: list[str],
                            extra_fp_counts: Optional[dict[str, int]] = None
                            ) -> tuple[int, int, float, list[str]]:
    """(beklenen, tahmin) çiftlerinden confusion matrix + precision/recall satırları
    üretir. Hem dosya-adı bazlı hem de kutu(IoU) bazlı benchmark raporu bunu paylaşır
    -- ikisinin de tek farkı "bir gözlem nedir" (görüntü mü, tek bir kutu mu).

    Tahmin (pred) None ise "yok" sayılır (dosya-adı benchmarkı bunu kullanır).
    match_ground_truth gibi zaten ayrık string kategoriler ("belirsiz", "kayıp")
    üreten çağıranlar için bu kategoriler AYNEN korunur -- tek bir "yok" kutusuna
    sıkıştırılmaz, çünkü "kutu bulunamadı" ile "kutu bulundu ama renk kararsız
    kaldı" birbirinden çok farklı sorunlar (biri detector, biri renk mantığı).

    extra_fp_counts: {renk: sayı} -- hiçbir GT kutusuyla eşleşmeyen ama renk
    verilmiş tespitler (match_ground_truth'un extra_fps'i). confusion matrix'in
    KENDİSİ bunları göstermez (satırları hep GT kutularıdır) ama precision
    hesabına eklenir -- yoksa "görüntüde gerçekte hiç duba olmayan bir yere
    yeşil kutu koydu" türü yanlış pozitifler precision'a hiç yansımaz."""
    extra_fp_counts = extra_fp_counts or {}
    extra_cols = sorted({("yok" if p is None else p) for _, p in pairs if p not in colors})
    predicted_cols = colors + extra_cols
    matrix: dict[str, dict[str, int]] = {e: {p: 0 for p in predicted_cols} for e in colors}

    correct = 0
    for expected, pred in pairs:
        p = pred if pred in colors else ("yok" if pred is None else pred)
        matrix[expected][p] += 1
        if p == expected:
            correct += 1

    total = len(pairs)
    accuracy = correct / total if total else 0.0

    lines = [
        "  Confusion matrix (satır=beklenen, sütun=tahmin):",
        "  " + f"{'':<10}" + "".join(f"{p:<10}" for p in predicted_cols),
    ]
    for e in colors:
        row = "".join(f"{matrix[e][p]:<10}" for p in predicted_cols)
        lines.append(f"  {e:<10}{row}")

    lines += ["", "  Renk başına precision / recall (fp_ekstra = GT'siz yerde bu renk verilen kutu):"]
    for c in colors:
        tp = matrix[c][c]
        fn = sum(matrix[c][p] for p in predicted_cols if p != c)
        fp_confused = sum(matrix[e][c] for e in colors if e != c)
        fp_extra = extra_fp_counts.get(c, 0)
        fp = fp_confused + fp_extra
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        lines.append(f"    {c:<8} precision={precision:.1%}  recall={recall:.1%}"
                     f"  (tp={tp} fp={fp} [karışan={fp_confused}+ekstra={fp_extra}] fn={fn})")

    return correct, total, accuracy, lines


def write_benchmark_report(results: list[ImageResult], output_dir: Path) -> None:
    """Dosya adından çıkarılan beklenen renkle tahmini karşılaştırıp confusion
    matrix + accuracy + per-class precision/recall + hatalı dosya listesini
    benchmark_raporu.txt'ye yazar. Beklenen rengi çıkarılamayan (adı
    yeniden adlandırılmamış) görüntüler sessizce dışarıda bırakılır.

    Bu görüntü-bazlı benchmark kaba bir ölçüm: bir görüntüdeki TÜM dubaların
    aynı renk olduğunu varsayar (dosya adı tek renk taşıyabilir). Görüntü
    başına birden fazla farklı renkli duba varsa write_bbox_benchmark_report
    (--ground-truth) çok daha doğru bir ölçüm verir."""
    labeled = [r for r in results if r.expected_color and not r.error]
    if not labeled:
        return  # hiçbir dosya adı renk içermiyor -- benchmark uygulanamaz

    colors = sorted({r.expected_color for r in labeled})
    pairs = [(r.expected_color, r.predicted_color) for r in labeled]
    correct, total, accuracy, matrix_lines = _confusion_matrix_lines(pairs, colors)

    mistakes = [r for r in labeled if r.predicted_color != r.expected_color]

    lines = [
        "=" * 72,
        "  RENK BENCHMARK RAPORU  (dosya adından çıkarılan etiketlerle)",
        "=" * 72,
        f"  Etiketli görüntü : {total} / {len(results)}",
        f"  Doğru            : {correct}",
        f"  Genel accuracy   : {accuracy:.1%}",
        "",
        *matrix_lines,
    ]

    if mistakes:
        lines += ["", f"  Hatalı sınıflandırılan {len(mistakes)} dosya:"]
        for r in mistakes:
            pred = r.predicted_color or "belirsiz/tespit yok"
            lines.append(f"    {r.path.name:<40} beklenen={r.expected_color:<8} tahmin={pred}")
    lines += ["=" * 72, ""]

    txt = "\n".join(lines) + "\n"
    p = output_dir / "benchmark_raporu.txt"
    p.write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print(f"  Benchmark → {p}")


# ---------------------------------------------------------------------------
# Kutu (bbox) bazlı benchmark: annotate_buoy_folder.py'nin ürettiği ground
# truth ile IoU eşleştirmesi
# ---------------------------------------------------------------------------
# Dosya-adı benchmarkı bir görüntüdeki TÜM dubaların aynı renk olduğunu
# varsayar; bu, annotate_buoy_folder.py ile elle çizilmiş gerçek kutularla
# TEK TEK eşleştirir -- aynı görüntüde farklı renkte birden fazla duba olsa
# bile doğru ölçer.

def load_ground_truth(path: Path) -> dict[str, list[dict]]:
    """annotate_buoy_folder.py'nin ürettiği JSON'u yükler.

    Rengi hâlâ None olan (etiketleme sırasında r/g/b'ye basılmadan bırakılmış,
    "bekleyen") kutular atılır -- yarım kalmış bir etiketleme dosyasının
    benchmark'ı sessizce bozmasını önler."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: [b for b in v if b.get("color")] for k, v in raw.items()}


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


_NO_BOX = "kayıp"       # bu GT kutusuyla yeterli IoU'da örtüşen HİÇBİR tespit yok
_UNCERTAIN = "belirsiz"  # örtüşen bir tespit var ama renk sınıflandırıcı karar veremedi
# Eslesmemis bir tespit HALA bu esigin ustunde bir GT kutusuyla ortusuyorsa
# "gercek yanlis pozitif" degil, ayni fiziksel dubanin ikinci bir sinif
# etiketiyle (NMS sinif bazinda calistigi icin) tekrar tespit edilmesi sayilir.
_DUPLICATE_OVERLAP_THRESH = 0.15


def match_ground_truth(gt_boxes: list[dict], detections: list[Detection],
                       iou_thresh: float = 0.3,
                       file_name: str = "") -> tuple[list[dict], list[dict], list[dict]]:
    """Her GT kutusunu (en yüksek IoU'lu, eşiği geçen, henüz kullanılmamış)
    bir tespitle eşleştirir. (records, extra_fps) döner.

    records: her GT kutusu için bir kayıt {file, expected, predicted, iou, ratios}.
    `predicted` üç şeyden biri olur:
      - gerçek bir renk adı  (kutu bulundu VE renk sınıflandırıcı karar verdi)
      - "belirsiz"           (kutu bulundu ama color_label None -- KLASİFİKASYON
                              sorunu, ayarlanabilir eşikler/aralıklarla düzeltilir)
      - "kayıp"               (IoU eşiğini geçen hiçbir tespit yok -- DETECTOR/
                              localization sorunu, renk mantığıyla düzeltilemez)
    Bu ayrım önemli: ikisini tek bir "yok" kutusunda toplamak "belirsizliği
    azaltmak için ne değiştirmeliyim" sorusuna yanlış cevap verdirir -- kayıp
    kutular renk eşiği ayarıyla asla düzelmez, sadece detector/RoI ayarıyla düzelir.
    `ratios` (YCrCb/HSV oranları) "belirsiz" kutuların NEDEN kararsız kaldığını
    gösterir -- bbox_benchmark_raporu.txt bunları doğrudan basar, tahmin
    yürütmek yerine gerçek sayılara bakılabilsin diye.

    extra_fps: renk atanmış, HİÇBİR GT kutusuyla (eşleşmiş olan dahil TÜM GT
    kutularıyla, sadece "used" olanlarla değil) anlamlı örtüşmesi olmayan
    tespitler -- gerçekten "burada duba yokken renk verilmiş" durumu.

    duplicates: renk atanmış, eşleşmemiş (başka bir tespit o GT kutusunu zaten
    aldı) AMA yine de bir GT kutusuyla anlamlı örtüşmesi olan tespitler. Bunlar
    yanlış pozitif DEĞİL -- modelin AYNI fiziksel dubayı birden fazla sınıfla
    (buoy/north_buoy/green_buoy/... NMS bunları birbirinden ayrı tuttuğu için)
    ikinci kez tespit etmesi. `agrees=False` olanlar asıl önemli olan kısım:
    aynı dubanın farklı çerçevelenmiş bir kopyası FARKLI bir renk okuyorsa, bu
    gerçek bir renk-kararsızlığı sinyalidir -- "yanlış pozitif" değil ama
    kutunun tam nerede kesildiğine göre rengin değişebildiğini gösterir.

    Bu üçe ayırmak önemli: hepsini tek bir "extra FP" kovasına atmak (ki ilk
    sürümü öyleydi) precision'ı YAPAY OLARAK düşürüyordu -- aynı, doğru
    renklendirilmiş bir dubanın ikinci bir sınıf etiketiyle tekrar tespit
    edilmesi gerçek bir hata değil, NMS'in sınıflar arası birleştirme
    yapmamasının sonucu.

    Açgözlü (greedy) en-iyi-IoU eşleştirmesi: Hungarian algoritması kadar
    optimal değil ama görüntü başına birkaç dubalık senaryoda fark etmez ve
    ekstra bağımlılık gerektirmez.
    """
    used: set[int] = set()
    records: list[dict] = []
    for gt in gt_boxes:
        best_iou, best_i = 0.0, -1
        for i, det in enumerate(detections):
            if i in used:
                continue
            iou = _iou(tuple(gt["bbox"]), det.bbox_xyxy)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_iou >= iou_thresh:
            used.add(best_i)
            det = detections[best_i]
            records.append({"file": file_name, "expected": gt["color"],
                            "predicted": det.color_label or _UNCERTAIN,
                            "iou": round(best_iou, 2), "ratios": det.color_ratios})
        else:
            records.append({"file": file_name, "expected": gt["color"],
                            "predicted": _NO_BOX, "iou": 0.0, "ratios": {}})

    extra_fps: list[dict] = []
    duplicates: list[dict] = []
    for i, det in enumerate(detections):
        if i in used or det.color_label is None:
            continue
        # Eslesmis olsun olmasin, TUM GT kutularina karsi en iyi ortusmeyi
        # bul -- bu tespit gercekten alakasiz mi, yoksa zaten baska bir
        # tespitin aldigi ayni dubanin ikinci bir kopyasi mi?
        best_gt_iou, best_gt = 0.0, None
        for gt in gt_boxes:
            iou = _iou(tuple(gt["bbox"]), det.bbox_xyxy)
            if iou > best_gt_iou:
                best_gt_iou, best_gt = iou, gt
        if best_gt is not None and best_gt_iou >= _DUPLICATE_OVERLAP_THRESH:
            duplicates.append({"file": file_name, "predicted": det.color_label,
                               "expected": best_gt["color"], "iou": round(best_gt_iou, 2),
                               "bbox": det.bbox_xyxy, "ratios": det.color_ratios,
                               "agrees": det.color_label == best_gt["color"]})
        else:
            extra_fps.append({"file": file_name, "predicted": det.color_label,
                              "bbox": det.bbox_xyxy, "ratios": det.color_ratios})
    return records, extra_fps, duplicates


def write_bbox_benchmark_report(records: list[dict], output_dir: Path, iou_thresh: float,
                                extra_fps: Optional[list[dict]] = None,
                                duplicates: Optional[list[dict]] = None) -> None:
    """match_ground_truth'un tüm görüntülerden biriktirdiği kayıtlardan confusion
    matrix + accuracy + precision/recall + belirsiz/yanlış/ekstra (GT'siz) kutuların
    YCrCb/HSV oranlarını bbox_benchmark_raporu.txt'ye yazar."""
    if not records:
        return  # --ground-truth verilmemiş ya da hiçbir görüntü etiketlenmemiş

    extra_fps = extra_fps or []
    duplicates = duplicates or []
    disagreeing_dups = [d for d in duplicates if not d["agrees"]]
    pairs = [(r["expected"], r["predicted"]) for r in records]
    colors = sorted({e for e, _ in pairs})
    extra_fp_counts: dict[str, int] = {}
    for fp in extra_fps:
        extra_fp_counts[fp["predicted"]] = extra_fp_counts.get(fp["predicted"], 0) + 1

    correct, total, accuracy, matrix_lines = _confusion_matrix_lines(pairs, colors, extra_fp_counts)
    missed = sum(1 for r in records if r["predicted"] == _NO_BOX)         # detector sorunu
    uncertain_recs = [r for r in records if r["predicted"] == _UNCERTAIN]  # renk sorunu
    wrong_recs = [r for r in records
                 if r["predicted"] not in (_NO_BOX, _UNCERTAIN, r["expected"])]

    def _ratio_str(ratios: dict) -> str:
        return "  ".join(f"{k}={v:.3f}" for k, v in sorted(ratios.items())
                         if isinstance(v, (int, float)))

    lines = [
        "=" * 72,
        "  KUTU BAZLI (IoU eşleştirmeli) RENK BENCHMARK RAPORU",
        "=" * 72,
        f"  IoU eşiği           : {iou_thresh:.2f}",
        f"  Etiketli kutu       : {total}",
        f"  Doğru               : {correct}",
        f"  Kayıp (kutu bulunamadı, detector sorunu) : {missed}",
        f"  Belirsiz (kutu bulundu, renk kararsız)   : {len(uncertain_recs)}",
        f"  Ekstra (GT'siz yerde renk verilmiş, gerçek YP) : {len(extra_fps)}",
        f"  Duplikat (aynı dubanın 2. sınıf/kutu tespiti)  : {len(duplicates)}"
        f"  ({len(disagreeing_dups)} tanesi FARKLI renk okudu -- bkz. altta)",
        f"  Genel accuracy      : {accuracy:.1%}",
        "",
        *matrix_lines,
    ]

    if uncertain_recs:
        lines += ["", f"  Belirsiz kalan {len(uncertain_recs)} kutu (renk sınıflandırıcı karar veremedi) --",
                 "  beklenen renk + o kutunun gerçek YCrCb/HSV oranları:"]
        for r in sorted(uncertain_recs, key=lambda r: r["expected"]):
            lines.append(f"    {r['file']:<24} beklenen={r['expected']:<7} IoU={r['iou']:.2f}  "
                         f"{_ratio_str(r['ratios'])}")

    if wrong_recs:
        lines += ["", f"  Yanlış renk verilen {len(wrong_recs)} kutu:"]
        for r in wrong_recs:
            lines.append(f"    {r['file']:<24} beklenen={r['expected']:<7} "
                         f"tahmin={r['predicted']:<7} IoU={r['iou']:.2f}  {_ratio_str(r['ratios'])}")

    if extra_fps:
        lines += ["", f"  Ekstra (GT'siz yerde renk verilen, gerçek yanlış pozitif) {len(extra_fps)} kutu --",
                 "  o görüntüde ETİKETLENMİŞ hiçbir gerçek dubayla örtüşmüyor:"]
        for fp in extra_fps:
            lines.append(f"    {fp['file']:<24} tahmin={fp['predicted']:<7} "
                         f"bbox={fp['bbox']}  {_ratio_str(fp['ratios'])}")

    if disagreeing_dups:
        lines += ["", f"  Renk KARARSIZLIĞI: aynı dubanın {len(disagreeing_dups)} duplikat kutusu FARKLI",
                 "  renk okudu (NMS aynı objeyi farklı sınıflarla 2+ kez tespit etti, bu kopyalardan",
                 "  biri asıl GT ile eşleşti/doğru okudu, öteki burada -- bbox'ın tam nerede kesildiğine",
                 "  göre rengin değiştiğini gösterir, gerçek bir yanlış pozitif değildir):"]
        for d in disagreeing_dups:
            lines.append(f"    {d['file']:<24} beklenen={d['expected']:<7} "
                         f"bu_kopya={d['predicted']:<7} IoU_vs_GT={d['iou']:.2f}  {_ratio_str(d['ratios'])}")

    lines += ["=" * 72, ""]

    txt = "\n".join(lines) + "\n"
    p = output_dir / "bbox_benchmark_raporu.txt"
    p.write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print(f"  Kutu bazlı benchmark → {p}")


# ---------------------------------------------------------------------------
# Rapor / CSV
# ---------------------------------------------------------------------------

def _sep(char="-", w=72):
    print(char * w)


def print_image_result(result: ImageResult, idx: int, total: int):
    _sep()
    status = "HATA" if result.error else f"{result.n_buoys} tespit"
    print(f"[{idx+1:>3}/{total}]  {result.path.name:<40}  →  {status}")
    if result.error:
        print(f"  ✗  {result.error}")
        return
    if not result.detections:
        print("  (Tespit yok)")
        return
    for i, det in enumerate(result.detections, 1):
        color_str  = det.color_label if det.color_label else "belirsiz"
        ratios_str = "  ".join(f"{k}={v:.2f}" for k, v in sorted(det.color_ratios.items()))
        x1, y1, x2, y2 = det.bbox_xyxy
        print(f"  [{i}] {det.class_name:<14} conf={det.det_conf:.2f}"
              f"  bbox=({x1},{y1},{x2},{y2})")
        print(f"       Renk: {color_str:<10} conf={det.color_conf:.2f}  {ratios_str}")
    print(f"  Süre: çıkarım={result.inference_ms:.0f}ms  renk={result.color_ms:.1f}ms")


def write_report(results: list[ImageResult], output_dir: Path, args):
    ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total    = len(results)
    errors   = sum(1 for r in results if r.error)
    ok       = total - errors
    n_dets   = sum(r.n_buoys for r in results)
    n_red    = sum(r.color_summary["red"]       for r in results)
    n_green  = sum(r.color_summary["green"]     for r in results)
    n_black  = sum(r.color_summary["black"]     for r in results)
    n_unc    = sum(r.color_summary["uncertain"] for r in results)

    mode = "no-model (renk maskesi)" if getattr(args, "no_model", False) else str(getattr(args, "weights", "?"))

    lines = [
        "=" * 72,
        f"  DUBA BATCH TEST RAPORU  —  {ts}",
        "=" * 72,
        f"  Girdi        : {args.input}",
        f"  Mod          : {mode}",
        f"  Renk config  : {args.ranges}",
        f"  Çıktı        : {output_dir}",
        "",
        f"  Toplam görüntü : {total}  (ok={ok}, hata={errors})",
        f"  Toplam tespit  : {n_dets}",
        "",
        "  Renk dağılımı:",
        f"    Kırmızı  : {n_red}",
        f"    Yeşil    : {n_green}",
        f"    Siyah    : {n_black}",
        f"    Belirsiz : {n_unc}",
        "",
        "-" * 72,
        f"  {'Dosya':<40} {'#':<4} {'Kırmızı':<9} {'Yeşil':<9} {'Siyah':<9} {'Belirsiz':<9} Durum",
        "-" * 72,
    ]
    for r in results:
        if r.error:
            lines.append(f"  {r.path.name:<40} {'—':<4} {'—':<9} {'—':<9} {'—':<9} {'—':<9} HATA")
        else:
            cs = r.color_summary
            lines.append(
                f"  {r.path.name:<40} {r.n_buoys:<4} "
                f"{cs['red']:<9} {cs['green']:<9} {cs['black']:<9} {cs['uncertain']:<9} OK"
            )
    lines += ["=" * 72, ""]

    txt = "\n".join(lines) + "\n"
    p = output_dir / "test_raporu.txt"
    p.write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print(f"  Rapor    → {p}")


def write_csv(results: list[ImageResult], output_dir: Path):
    p = output_dir / "tespitler.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["dosya", "tespit_no", "sinif", "det_conf",
                     "x1", "y1", "x2", "y2",
                     "renk", "renk_conf", "oran_red", "oran_green", "oran_black",
                     "beklenen", "tahmin", "dogru"])
        for r in results:
            if r.error:
                continue
            for i, det in enumerate(r.detections, 1):
                wr.writerow([
                    r.path.name, i, det.class_name, f"{det.det_conf:.4f}",
                    *det.bbox_xyxy,
                    det.color_label or "belirsiz", f"{det.color_conf:.4f}",
                    f"{det.color_ratios.get('red',   0):.4f}",
                    f"{det.color_ratios.get('green', 0):.4f}",
                    f"{det.color_ratios.get('black', 0):.4f}",
                    r.expected_color or "",
                    r.predicted_color or "",
                    "" if not r.expected_color else str(r.predicted_color == r.expected_color),
                ])
    print(f"  CSV      → {p}")


def write_json(results: list[ImageResult], output_dir: Path):
    data = [
        {
            "file": str(r.path),
            "error": r.error,
            "inference_ms": round(r.inference_ms, 1),
            "color_ms": round(r.color_ms, 2),
            "detections": [
                {
                    "class": d.class_name, "det_conf": round(d.det_conf, 4),
                    "bbox": list(d.bbox_xyxy),
                    "color": d.color_label, "color_conf": round(d.color_conf, 4),
                    "ratios": {k: round(v, 4) for k, v in d.color_ratios.items()},
                }
                for d in r.detections
            ],
        }
        for r in results
    ]
    p = output_dir / "tespitler.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  JSON     → {p}")


# ---------------------------------------------------------------------------
# Ana döngü
# ---------------------------------------------------------------------------

def run(args):
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"HATA: Klasör bulunamadı: {input_dir}", file=sys.stderr); sys.exit(1)

    output_dir = Path(args.output) if args.output else input_dir / "test_ciktisi"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_dir, exclude_dir=output_dir)
    if not images:
        print(f"HATA: '{input_dir}' içinde görüntü bulunamadı.", file=sys.stderr); sys.exit(1)

    ranges_path = Path(args.ranges)
    if not ranges_path.exists():
        print(f"HATA: color_ranges.yaml bulunamadı: {ranges_path}", file=sys.stderr); sys.exit(1)
    color_ranges = load_color_ranges(str(ranges_path))

    ground_truth = None
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        if not gt_path.exists():
            print(f"HATA: ground-truth bulunamadı: {gt_path}", file=sys.stderr); sys.exit(1)
        ground_truth = load_ground_truth(gt_path)

    model  = None
    labels = {}
    if not args.no_model:
        model  = load_model(args.weights, model_type=args.model_type)
        labels = load_labels(Path(args.labels))

    print(f"\n{'='*72}")
    print(f"  Duba Batch Test  —  {len(images)} görüntü")
    print(f"{'='*72}")
    print(f"  Girdi   : {input_dir}")
    print(f"  Çıktı   : {output_dir}")
    print(f"  Renkler : {list(color_ranges)}")
    print(f"  Mod     : {'no-model (renk maskesi + bbox)' if args.no_model else args.weights}")
    print(f"  Ön işl. : WB(gray-world) {'açık' if args.white_balance else 'kapalı'}  "
          f"CLAHE(Y) {'açık' if args.clahe else 'kapalı'}  "
          f"Glare mask {'açık' if args.glare_mask else 'kapalı'} (Y>={args.glare_y_thresh})")
    print(f"  HSV red  : {'açık' if args.hsv_red else 'kapalı'} (eşik={args.hsv_red_min:.2f})  "
          f"HSV green(AND): {'açık' if args.hsv_green else 'kapalı'} (eşik={args.hsv_green_min:.2f})")
    if args.no_model:
        print(f"  Min alan: {args.min_area}px²  |  Morph kernel: {args.morph_k}")
        print(f"  Adapt.th : {'açık' if args.adaptive_thresh else 'kapalı'}")
        print(f"  Y-only B : {'açık' if args.y_only_black else 'kapalı'}  "
              f"(Y_max={args.y_black_max}  Cr_tol={args.y_black_cr_tol}  Cb_tol={args.y_black_cb_tol})")
        print(f"  Fill flt : {'açık' if args.green_fill_filter else 'kapalı'}  "
              f"(min={args.green_fill_min:.2f})  "
              f"Sat flt: {'açık' if args.green_sat_filter else 'kapalı'}  "
              f"(min={args.green_sat_min:.0f})")
    if ground_truth is not None:
        print(f"  GT       : {args.ground_truth}  (IoU eşiği={args.gt_iou_thresh:.2f})")
    print()

    all_results: list[ImageResult] = []
    bbox_records: list[dict] = []
    bbox_extra_fps: list[dict] = []
    bbox_duplicates: list[dict] = []

    for idx, img_path in enumerate(images):
        result = ImageResult(path=img_path)
        result.expected_color = expected_color_from_filename(img_path)

        frame = cv2.imread(str(img_path))
        if frame is None:
            result.error = "Görüntü okunamadı"
            all_results.append(result)
            print_image_result(result, idx, len(images))
            continue

        # Ön işleme zinciri: gray-world WB → CLAHE(Y). Kare gelir gelmez,
        # aşağıdaki tespit/renk sınıflandırmasından ÖNCE uygulanır.
        if args.white_balance or args.clahe:
            frame = preprocess_frame(frame, use_white_balance=args.white_balance,
                                     use_clahe=args.clahe)

        try:
            if model is not None:
                dets, inf_ms, col_ms = process_with_model(
                    frame, model, labels, color_ranges, args.conf, args.roi_shrink,
                    glare_y_thresh=args.glare_y_thresh if args.glare_mask else 255,
                    use_hsv_red=args.hsv_red, hsv_red_min_ratio=args.hsv_red_min,
                    use_hsv_green=args.hsv_green, hsv_green_min_ratio=args.hsv_green_min,
                )
            else:
                tc0 = time.perf_counter()
                dets = _find_color_boxes(
                    frame, color_ranges,
                    min_area_px=args.min_area,
                    max_boxes=args.max_boxes,
                    morph_k=args.morph_k,
                    use_hsv_red=args.hsv_red,
                    use_adaptive_thresh=args.adaptive_thresh,
                    use_y_only_black=args.y_only_black,
                    hsv_red_min_ratio=args.hsv_red_min,
                    y_only_black_max=args.y_black_max,
                    y_only_black_cr_tol=args.y_black_cr_tol,
                    y_only_black_cb_tol=args.y_black_cb_tol,
                    black_close_k=args.black_close_k,
                    use_green_fill_filter=args.green_fill_filter,
                    green_fill_min=args.green_fill_min,
                    use_green_sat_filter=args.green_sat_filter,
                    green_sat_min=args.green_sat_min,
                    use_glare_mask=args.glare_mask,
                    glare_y_thresh=args.glare_y_thresh,
                    use_hsv_green=args.hsv_green,
                    hsv_green_min_ratio=args.hsv_green_min,
                )
                col_ms = (time.perf_counter() - tc0) * 1000
                inf_ms = 0.0

            result.detections   = dets
            result.inference_ms = inf_ms
            result.color_ms     = col_ms
            result.predicted_color = predicted_color_for_image(dets)

            if ground_truth is not None:
                gt_boxes = ground_truth.get(str(img_path.relative_to(input_dir)), [])
                if gt_boxes:
                    recs, extras, dups = match_ground_truth(
                        gt_boxes, dets, args.gt_iou_thresh, file_name=img_path.name)
                    bbox_records.extend(recs)
                    bbox_extra_fps.extend(extras)
                    bbox_duplicates.extend(dups)

        except Exception as e:
            result.error = str(e)
            all_results.append(result)
            print_image_result(result, idx, len(images))
            continue

        annotated = annotate_frame(frame, dets, no_model=args.no_model)
        out_path  = output_dir / f"annotated_{img_path.stem}{img_path.suffix}"
        cv2.imwrite(str(out_path), annotated)

        all_results.append(result)
        print_image_result(result, idx, len(images))

    _sep("=")
    write_report(all_results, output_dir, args)
    write_csv(all_results, output_dir)
    write_benchmark_report(all_results, output_dir)
    if ground_truth is not None:
        write_bbox_benchmark_report(bbox_records, output_dir, args.gt_iou_thresh, bbox_extra_fps, bbox_duplicates)
    if args.json:
        write_json(all_results, output_dir)

    print(f"\n  Annotated görüntüler → {output_dir}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True,
                   help="Duba fotoğraflarının bulunduğu klasör")
    p.add_argument("--weights", "-w", default=None,
                   help="Ağırlık dosyası (.pt / .onnx) -- RT-DETR ya da YOLO, ikisi de olur "
                        "(bkz. --model-type). --no-model ile kullanılmaz.")
    p.add_argument("--model-type", choices=["auto", "yolo", "rtdetr"], default="auto",
                   help="Ağırlık dosyasının mimarisi (varsayılan: auto -- checkpoint "
                        "metadata'sından otomatik tespit eder)")
    p.add_argument("--no-model", action="store_true",
                   help="Model olmadan sadece renk maskesiyle bounding box bul")
    p.add_argument("--output", "-o", default=None,
                   help="Çıktı klasörü (varsayılan: <input>/test_ciktisi/)")
    p.add_argument("--ranges", default=str(DEFAULT_RANGES),
                   help="color_ranges.yaml (varsayılan: config/color_ranges.yaml)")
    p.add_argument("--labels", default=str(DEFAULT_LABELS),
                   help="class_labels.yaml (varsayılan: config/class_labels.yaml)")
    p.add_argument("--conf", type=float, default=0.05,
                   help="Model güven eşiği (varsayılan: 0.05 -- 0.3'ten düşürüldü: "
                        "duba_fotolari + annotations.json üzerinde IoU eşleştirmeli "
                        "benchmark'ta 0.3->0.05 hiçbir renk için yanlış-pozitif eklemeden "
                        "kırmızı recall'unu %35->%60, yeşili %20->%30 çıkardı -- "
                        "bkz. bbox_benchmark_raporu.txt geçmişi)")
    p.add_argument("--roi-shrink", type=float, default=0.6,
                   help="Renk ROI merkez oranı (varsayılan: 0.6)")
    # Ön işleme: gray-world WB + CLAHE(Y) -- her karede, her modda uygulanır
    p.add_argument("--white-balance", action=argparse.BooleanOptionalAction, default=True,
                   help="Gray-world beyaz dengesi ön işlemesi (varsayılan: açık)")
    p.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=True,
                   help="CLAHE (sadece Y kanalı) ön işlemesi (varsayılan: açık)")
    # Glare (parlama) maskeleme -- her modda uygulanır
    p.add_argument("--glare-mask", action=argparse.BooleanOptionalAction, default=True,
                   help="Y >= eşik piksellerini renk maskelerinden çıkar (varsayılan: açık)")
    p.add_argument("--glare-y-thresh", type=int, default=245,
                   help="Glare (parlama) Y eşiği 0-255 (varsayılan: 245)")
    # --no-model temel parametreler
    p.add_argument("--min-area", type=int, default=800,
                   help="[no-model] Bounding box için minimum piksel alanı (varsayılan: 800)")
    p.add_argument("--max-boxes", type=int, default=8,
                   help="[no-model] Görüntü başına max bounding box (varsayılan: 8)")
    p.add_argument("--morph-k", type=int, default=9,
                   help="[no-model] Morfoloji kernel boyutu (varsayılan: 9)")
    # HSV paralel kırmızı (OR/kurtarma) -- her modda uygulanır
    p.add_argument("--hsv-red", action=argparse.BooleanOptionalAction, default=True,
                   help="HSV paralel kırmızı kontrolü (varsayılan: açık)")
    p.add_argument("--hsv-red-min", type=float, default=0.10,
                   help="HSV kırmızı kabul eşiği 0-1 (varsayılan: 0.10)")
    # YCrCb + HSV yeşil AND oylaması -- her modda uygulanır
    p.add_argument("--hsv-green", action=argparse.BooleanOptionalAction, default=True,
                   help="YCrCb+HSV yeşil AND oylaması, sadece onay (varsayılan: açık)")
    p.add_argument("--hsv-green-min", type=float, default=0.10,
                   help="HSV yeşil onay eşiği 0-1 (varsayılan: 0.10)")
    # Problem 2: Adaptive threshold
    p.add_argument("--adaptive-thresh", action=argparse.BooleanOptionalAction, default=True,
                   help="[no-model] Görüntü bazlı adaptive confidence eşiği (varsayılan: açık)")
    # Problem 3: Y-only (+ nötr Cr/Cb) siyah
    p.add_argument("--y-only-black", action=argparse.BooleanOptionalAction, default=True,
                   help="[no-model] Siyah için Y-only + nötr Cr/Cb maskesi (varsayılan: açık)")
    p.add_argument("--y-black-max", type=int, default=45,
                   help="[no-model] Y-only siyah eşiği 0-255 (varsayılan: 45)")
    p.add_argument("--y-black-cr-tol", type=int, default=15,
                   help="[no-model] Siyah nötr Cr toleransı |Cr-128|< (varsayılan: 15)")
    p.add_argument("--y-black-cb-tol", type=int, default=15,
                   help="[no-model] Siyah nötr Cb toleransı |Cb-128|< (varsayılan: 15)")
    p.add_argument("--black-close-k", type=int, default=25,
                   help="[no-model] Siyah closing kernel boyutu (varsayılan: 25)")
    # Problem 1 (Yeşil): Fill ratio + doygunluk filtreleri
    p.add_argument("--green-fill-filter", action=argparse.BooleanOptionalAction, default=True,
                   help="[no-model] Yeşil doluluk oranı filtresi (varsayılan: açık)")
    p.add_argument("--green-fill-min", type=float, default=0.35,
                   help="[no-model] Yeşil min doluluk oranı 0–1 (varsayılan: 0.35)")
    p.add_argument("--green-sat-filter", action=argparse.BooleanOptionalAction, default=True,
                   help="[no-model] Yeşil doygunluk filtresi (varsayılan: açık)")
    p.add_argument("--green-sat-min", type=float, default=60.0,
                   help="[no-model] Yeşil min medyan HSV S eşiği 0–255 (varsayılan: 60)")
    p.add_argument("--json", action="store_true",
                   help="tespitler.json da oluştur")
    # Kutu(bbox) bazlı benchmark: annotate_buoy_folder.py'nin ürettiği ground truth
    p.add_argument("--ground-truth", default=None, metavar="JSON",
                   help="annotate_buoy_folder.py ile üretilmiş annotations.json -- verilirse "
                        "IoU eşleştirmeli bbox_benchmark_raporu.txt da yazılır")
    p.add_argument("--gt-iou-thresh", type=float, default=0.3,
                   help="Ground-truth eşleştirmesi için min IoU (varsayılan: 0.3)")
    return p


def main():
    args = _parser().parse_args()
    if not args.no_model and not args.weights:
        _parser().error("--weights gereklidir; model yoksa --no-model kullan.")
    run(args)


if __name__ == "__main__":
    main()
