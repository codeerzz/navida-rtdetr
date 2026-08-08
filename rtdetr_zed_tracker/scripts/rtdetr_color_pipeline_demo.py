#!/usr/bin/env python3
"""
rtdetr_color_pipeline_demo.py — RT-DETR şekil tespiti + YCrCb renk sınıflandırmasını
canlı kamerada İKİ AYRI, birleştirilmemiş aşama olarak göster.

Gerçek ROS boru hattımızdaki iki bağımsız aşamayı (tracker_node'un RT-DETR
tarafı + color_classification_node'un renk tarafı) ROS olmadan, tek pencerede
taklit eder. Her tespit kutusunun üstünde RT-DETR'in KENDİ tahminini, altında
YCrCb renk sınıflandırıcısının BAĞIMSIZ kararını ayrı satırlar halinde
gösteriyor -- kasıtlı olarak birleştirilmedi, ikisini tek tek değerlendirebilesin
diye. (İkisini gerçekten birleştiren mantık class_remap_node.py +
color_classification_node.py'de -- bkz. README.)

Kullanım:
  python3 rtdetr_color_pipeline_demo.py --weights /path/to/best.pt
  python3 rtdetr_color_pipeline_demo.py --weights /path/to/best.onnx --conf 0.4
  python3 rtdetr_color_pipeline_demo.py --weights best.pt --camera 1
  python3 rtdetr_color_pipeline_demo.py --weights best.pt --image ornek.jpg  # kamerasız

`--weights` hem .pt hem .onnx kabul eder -- ultralytics'in RTDETR sarmalayıcısı
ikisini de aynı arayüzle yükler. TensorRT `.plan` dosyaları BURADA ÇALIŞMAZ (o
formatı sadece Jetson/Isaac ROS çalıştırabilir); .onnx ya da .pt'ye ihtiyacın var.

Kameradayken:
  q — çık

Gereksinimler:
  pip3 install ultralytics opencv-python pyyaml numpy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

# rtdetr_zed_tracker/rtdetr_zed_tracker/color_classifier.py'yi colcon/pip kurulumu
# olmadan da import edebilmek icin paket klasorunun ustunu sys.path'e ekle.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rtdetr_zed_tracker.color_classifier import classify_color, load_color_ranges  # noqa: E402

DEFAULT_RANGES = Path(__file__).resolve().parent.parent / 'config' / 'color_ranges.yaml'
# tracker_node.py / class_labels.yaml ile ayni -- bkz. rtdetr_zed_tracker/tracker_node.py
DEFAULT_LABELS = {
    0: 'buoy', 1: 'east_buoy', 2: 'green_buoy', 3: 'north_buoy',
    4: 'red_buoy', 5: 'south_buoy', 6: 'west_buoy',
}
_COLOR_BGR = {'red': (0, 0, 255), 'green': (0, 200, 0)}


def load_labels(path: Path | None) -> dict:
    if path is None:
        return dict(DEFAULT_LABELS)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return {int(k): str(v) for k, v in raw.items()}


def load_detector(weights_path: str):
    """RT-DETR .pt/.onnx yukle. Ultralytics'in RTDETR sinifi ikisini de ayni
    arayuzle calistirir -- TensorRT .plan burada desteklenmez (bkz. modul docstring)."""
    from ultralytics import RTDETR
    return RTDETR(weights_path)


def _draw_two_stage(frame, box_xyxy, shape_name: str, det_conf: float, color_result):
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    color_label = color_result.label or 'belirsiz'
    color_bgr = _COLOR_BGR.get(color_result.label, (0, 200, 255))

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
    # Asama 1: RT-DETR'in KENDI tahmini (turuncu, ust satir)
    cv2.putText(frame, f'RT-DETR: {shape_name} ({det_conf:.2f})', (x1, max(16, y1 - 26)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2, cv2.LINE_AA)
    # Asama 2: renk siniflandiricinin BAGIMSIZ karari (kirmizi/yesil, alt satir)
    cv2.putText(frame, f'Renk: {color_label} ({color_result.confidence:.2f})', (x1, max(32, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2, cv2.LINE_AA)


def _process_frame(frame, model, color_ranges, labels, conf_thresh, roi_shrink):
    results = model.predict(frame, conf=conf_thresh, verbose=False)
    boxes = results[0].boxes if results else None
    detections = []
    if boxes is not None:
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            det_conf = float(box.conf[0])
            shape_name = labels.get(cls_id, f'unknown_{cls_id}')

            x1, y1, x2, y2 = [int(v) for v in xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            crop = frame[y1:y2, x1:x2]
            color_result = classify_color(crop, color_ranges, roi_shrink=roi_shrink)

            _draw_two_stage(frame, xyxy, shape_name, det_conf, color_result)
            detections.append((shape_name, det_conf, color_result))
    return detections


def run(weights_path: str, ranges_path: Path, camera_index: int, conf_thresh: float,
       labels: dict, roi_shrink: float):
    model = load_detector(weights_path)
    color_ranges = load_color_ranges(str(ranges_path))
    print(f'Model yuklendi: {weights_path}', flush=True)
    print(f'Renk config: {ranges_path}  ({list(color_ranges)})', flush=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f'HATA: kamera acilamadi (index={camera_index}). Baska bir --camera indexi dene.', flush=True)
        return 1

    window = 'RT-DETR (ust) + renk siniflandirici (alt) -- ayri ayri gosterim'
    cv2.namedWindow(window)

    while True:
        ok, frame = cap.read()
        if not ok:
            print('HATA: kameradan kare okunamadi.', flush=True)
            break

        _process_frame(frame, model, color_ranges, labels, conf_thresh, roi_shrink)
        cv2.putText(frame, 'q: cik', (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.imshow(window, frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def run_single_image(weights_path: str, ranges_path: Path, image_path: Path, conf_thresh: float,
                     labels: dict, roi_shrink: float):
    """Kamerasiz hizli dogrulama: tek bir goruntu dosyasi uzerinde calistir, sonuclari yazdir."""
    model = load_detector(weights_path)
    color_ranges = load_color_ranges(str(ranges_path))
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f'HATA: goruntu okunamadi: {image_path}')
        return 1

    detections = _process_frame(frame, model, color_ranges, labels, conf_thresh, roi_shrink)
    if not detections:
        print('Hicbir tespit bulunamadi (conf esigini dusurmeyi dene: --conf 0.1).')
        return 0
    for shape_name, det_conf, color_result in detections:
        print(f'RT-DETR: {shape_name} ({det_conf:.2f})  |  Renk: {color_result.label} '
              f'({color_result.confidence:.2f})  ratios={color_result.ratios}')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', type=str, required=True, help='RT-DETR agirlik dosyasi (.pt ya da .onnx)')
    p.add_argument('--camera', type=int, default=0, help='kamera index (varsayilan: 0)')
    p.add_argument('--image', type=Path, default=None,
                   help='kamera yerine tek bir goruntu dosyasi uzerinde test et (pencere acmaz)')
    p.add_argument('--ranges', type=Path, default=DEFAULT_RANGES, help='color_ranges.yaml yolu')
    p.add_argument('--labels', type=Path, default=None,
                   help='class_labels.yaml yolu (verilmezse built-in 7 sinif)')
    p.add_argument('--conf', type=float, default=0.4, help='RT-DETR guven esigi')
    p.add_argument('--roi-shrink', type=float, default=0.6, help='renk icin ROI ortasindan orneklenen oran')
    args = p.parse_args()

    if not args.ranges.exists():
        print(f'HATA: renk config dosyasi bulunamadi: {args.ranges}')
        return 1
    labels = load_labels(args.labels)

    if args.image is not None:
        return run_single_image(args.weights, args.ranges, args.image, args.conf, labels, args.roi_shrink)
    return run(args.weights, args.ranges, args.camera, args.conf, labels, args.roi_shrink)


if __name__ == '__main__':
    sys.exit(main())
