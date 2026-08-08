#!/usr/bin/env python3
"""
webcam_color_demo.py — color_classifier.py'yi kendi kameranla canlı test et.

ROS'a, ZED'e, robota gerek yok — sadece dizüstünün kamerası. Ortadaki kutucuğa
(ya da 's' ile seçtiğin bölgeye) tuttuğun nesnenin YCrCb rengini canlı gösterir;
kırmızı/yeşil bir nesneyi ışığa/gölgeye/parlamaya sokup etiketin sabit kalıp
kalmadığını gözle kontrol edebilirsin.

Kullanım:
  python3 webcam_color_demo.py                      # varsayılan kamera (0)
  python3 webcam_color_demo.py --camera 1            # başka bir kamera indexi
  python3 webcam_color_demo.py --ranges path/to.yaml # farklı bir renk config'i
  python3 webcam_color_demo.py --image ornek.jpg     # kamera yerine tek bir görsel

Kameradayken:
  s   — donan kareden fare ile bir dikdörtgen seç (ENTER ile onayla) -> o bölge
        artık takip edilen ROI olur. Elindeki nesneyi tam kapsamak için kullan.
  r   — ROI'yi ekranın ortasındaki varsayılan kutuya sıfırla
  q   — çık

Confidence düşük çıkıyorsa (config'teki değerler senin gerçek kameranla/ışığınla
uyuşmuyor demektir) kalibrasyon modunu kullan:
  python3 webcam_color_demo.py --calibrate red

  c   — o anki ROI'yi bir örnek olarak yakala (nesneyi farklı ışıkta/gölgede/
        parlamada birkaç kere 'c'ye basarak biriktir -> ne kadar çeşitli örnek,
        o kadar sağlam aralık). O örneğin KENDİ aralığını hemen terminale basar
        -- diğerlerinden çok farklıysa (arka plan/elin karıştıysa) hemen görürsün.
  z   — son yakalanan örneği geri al (kirli bir örneği fark edince kullan)
  p   — o ana kadar biriken örneklerden hesaplanan [y_min,y_max,cr_min,cr_max,
        cb_min,cb_max] aralığını terminale, color_ranges.yaml'a yapıştırmaya
        hazır şekilde bas (dosyayı OTOMATİK değiştirmez -- kendin gözden geçirip
        yapıştır). Aynı yaml'daki BAŞKA bir renkle çakışıyorsa uyarır -- bunu
        görmezden gelme, "her şeyi kırmızı algılama" gibi hataların sebebi bu.
  x   — biriken örnekleri temizle

Gereksinimler:
  pip3 install opencv-python pyyaml numpy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# rtdetr_zed_tracker/rtdetr_zed_tracker/color_classifier.py'yi colcon/pip kurulumu
# olmadan da import edebilmek için paket klasörünün üstünü sys.path'e ekle.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rtdetr_zed_tracker.color_classifier import (  # noqa: E402
    classify_color,
    load_color_ranges,
    ranges_overlap,
    suggest_range_from_samples,
)

DEFAULT_RANGES = Path(__file__).resolve().parent.parent / 'config' / 'color_ranges.yaml'


def _default_roi(frame_shape) -> tuple[int, int, int, int]:
    """Ekranın ortasında, genişliğin/yüksekliğin %35'i kadar bir kutu."""
    h, w = frame_shape[:2]
    bw, bh = int(w * 0.35), int(h * 0.35)
    x1 = (w - bw) // 2
    y1 = (h - bh) // 2
    return x1, y1, bw, bh


def _draw_overlay(frame, roi_xywh, result, calibrate_color: str | None, n_samples: int):
    x, y, w, h = roi_xywh
    label = result.label or 'belirsiz'
    color_bgr = {'red': (0, 0, 255), 'green': (0, 200, 0)}.get(result.label, (0, 200, 255))

    cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)

    info_h = 24 * (len(result.ratios) + 2)
    cv2.rectangle(frame, (0, 0), (380, info_h), (30, 30, 30), -1)
    cv2.putText(frame, f'label: {label}  (conf={result.confidence:.2f})', (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)
    for i, (color, ratio) in enumerate(sorted(result.ratios.items(), key=lambda kv: -kv[1])):
        cv2.putText(frame, f'  {color}: {ratio:.2f}', (10, 22 + 24 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    if calibrate_color is not None:
        cv2.putText(frame, f'KALIBRASYON [{calibrate_color}] -- ornekler: {n_samples}',
                    (10, info_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
        help_text = "s: bolge | c: yakala | z: geri al | p: araligi yazdir | x: temizle | q: cik"
    else:
        help_text = "s: bolge sec | r: sifirla | q: cik"
    cv2.putText(frame, help_text, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


# Bir aralik bandinin (percentile_high - percentile_low) bu degerlerden genisse,
# ROI muhtemelen tek renkli degil -- arka plan/golge siniri/baska nesne karismis.
# Gercek bir dubaya siki oturmus bir ROI'de Cr/Cb tipik olarak bundan cok daha dar
# cikar (bkz. test_color_classifier.py'deki gercek olcumler: tek bir ısık kosulunda
# Cr spani ~10-20 civari).
_CONTAMINATION_SPAN_THRESHOLD = {'y': 120, 'cr': 60, 'cb': 60}


def _print_single_sample_range(sample_index: int, crop, roi_shrink: float):
    """'c' basildiginda o TEK ornegin kendi araligini hemen goster -- diger
    orneklerden cok farkliysa (arka plan/el karismis olabilir) hemen belli olur."""
    y_min, y_max, cr_min, cr_max, cb_min, cb_max = suggest_range_from_samples([crop], roi_shrink, pad=0)
    print(f'  ornek #{sample_index}: Y[{y_min},{y_max}] Cr[{cr_min},{cr_max}] Cb[{cb_min},{cb_max}]', flush=True)

    spans = {'y': y_max - y_min, 'cr': cr_max - cr_min, 'cb': cb_max - cb_min}
    wide = {ch: span for ch, span in spans.items() if span > _CONTAMINATION_SPAN_THRESHOLD[ch]}
    if wide:
        print(f'  !! Bu ornek COK GENIS ({wide}) -- ROI muhtemelen tek renk degil (arka plan/golge '
              f'siniri/el karismis olabilir). "z" ile bunu geri al, "s" ile dubanin SADECE govdesini '
              f'kaplayan daha siki bir bolge sec, tekrar dene.', flush=True)


def _print_suggested_range(color_name: str, samples, roi_shrink: float, other_ranges: dict):
    if not samples:
        print('Once "c" ile en az bir ornek yakala.', flush=True)
        return
    learned = suggest_range_from_samples(samples, roi_shrink)
    y_min, y_max, cr_min, cr_max, cb_min, cb_max = learned
    print(f'\n--- {color_name}: {len(samples)} ornekten hesaplanan aralik '
          f'({"genis, cesitli isik" if len(samples) > 1 else "TEK ornek, daha fazla c ile cesitlendir"}) ---')
    print(f'{color_name}:')
    print(f'  - [{y_min}, {y_max}, {cr_min}, {cr_max}, {cb_min}, {cb_max}]')

    for other_color, ranges in other_ranges.items():
        if other_color == color_name:
            continue
        for other_range in ranges:
            if ranges_overlap(learned, other_range):
                print(f'  !! UYARI: bu aralik "{other_color}" ile cakisiyor {other_range} -- '
                      f'muhtemelen bir ornekte arka plan/baska nesne karisti. "z" ile supheli '
                      f'orneği geri al ya da tum ornekleri "x" ile temizleyip ROI\'yi daha siki '
                      f'secerek tekrar dene.')
    print('color_ranges.yaml icine yapistirmadan once mevcut satirlari gozden gecir.\n', flush=True)


def run(camera_index: int, ranges_path: Path, roi_shrink: float, min_confidence: float,
       calibrate_color: str | None):
    color_ranges = load_color_ranges(str(ranges_path))
    print(f'Yuklenen renkler: {list(color_ranges)}  ({ranges_path})', flush=True)
    if calibrate_color is not None:
        print(f'Kalibrasyon modu: "{calibrate_color}". c=yakala, p=araligi yazdir, x=temizle.', flush=True)
    else:
        print('Kalibrasyon modu KAPALI (--calibrate <renk> vermedin) -- c/p/x/z tuslari calismaz, '
              'sadece etiket goruntulenir.', flush=True)
    print('NOT: klavye kisayollari calismasi icin video PENCERESI aktif olmali -- tuslara basmadan '
          'once fare ile video penceresine tikla (terminale degil).', flush=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f'HATA: kamera acilamadi (index={camera_index}). Baska bir --camera indexi dene.', flush=True)
        return 1

    roi = None
    samples = []
    window = 'color_classifier canli test'
    cv2.namedWindow(window)

    while True:
        ok, frame = cap.read()
        if not ok:
            print('HATA: kameradan kare okunamadi.')
            break

        if roi is None:
            roi = _default_roi(frame.shape)
        x, y, w, h = roi
        crop = frame[y:y + h, x:x + w]

        result = classify_color(crop, color_ranges, roi_shrink=roi_shrink,
                                 min_confidence=min_confidence)
        _draw_overlay(frame, roi, result, calibrate_color, len(samples))
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:      # hicbir tusa basilmadi (cv2.waitKey'in "bos" donusu)
            pass
        elif key in (ord('q'), 27):
            break
        elif key == ord('s'):
            selected = cv2.selectROI(window, frame, showCrosshair=True)
            if selected[2] > 0 and selected[3] > 0:
                roi = selected
        elif key == ord('r'):
            roi = None
        elif key in (ord('c'), ord('z'), ord('p'), ord('x')) and calibrate_color is None:
            print(f'  "{chr(key)}" sadece kalibrasyon modunda calisir -- scripti '
                  f'"--calibrate red" (ya da baska bir renk) ile yeniden baslat.', flush=True)
        elif key == ord('c'):
            samples.append(crop.copy())
            _print_single_sample_range(len(samples), crop, roi_shrink)
        elif key == ord('z'):
            if samples:
                samples.pop()
                print(f'  son ornek geri alindi (kalan: {len(samples)})', flush=True)
            else:
                print('  geri alinacak ornek yok.', flush=True)
        elif key == ord('p'):
            _print_suggested_range(calibrate_color, samples, roi_shrink, color_ranges)
        elif key == ord('x'):
            samples = []
            print('  ornekler temizlendi.', flush=True)

    cap.release()
    cv2.destroyAllWindows()
    return 0


def run_single_image(image_path: Path, ranges_path: Path, roi_shrink: float, min_confidence: float):
    """Kamerasız hizli dogrulama: tek bir goruntu dosyasi uzerinde calistir."""
    color_ranges = load_color_ranges(str(ranges_path))
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f'HATA: goruntu okunamadi: {image_path}')
        return 1
    roi = _default_roi(frame.shape)
    x, y, w, h = roi
    result = classify_color(frame[y:y + h, x:x + w], color_ranges, roi_shrink, min_confidence)
    print(f'label={result.label} confidence={result.confidence:.3f} ratios={result.ratios}')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--camera', type=int, default=0, help='kamera index (varsayilan: 0)')
    p.add_argument('--image', type=Path, default=None,
                   help='kamera yerine tek bir goruntu dosyasi uzerinde test et (pencere acmaz)')
    p.add_argument('--ranges', type=Path, default=DEFAULT_RANGES, help='color_ranges.yaml yolu')
    p.add_argument('--roi-shrink', type=float, default=0.6, help='ROI ortasindan orneklenen oran')
    p.add_argument('--min-confidence', type=float, default=0.12, help='belirsiz sayilma esigi')
    p.add_argument('--calibrate', type=str, default=None, metavar='RENK',
                   help='kalibrasyon modu: bu renk icin ornek biriktir (orn. --calibrate red)')
    args = p.parse_args()

    if not args.ranges.exists():
        print(f'HATA: renk config dosyasi bulunamadi: {args.ranges}')
        return 1

    if args.image is not None:
        return run_single_image(args.image, args.ranges, args.roi_shrink, args.min_confidence)
    return run(args.camera, args.ranges, args.roi_shrink, args.min_confidence, args.calibrate)


if __name__ == '__main__':
    sys.exit(main())
