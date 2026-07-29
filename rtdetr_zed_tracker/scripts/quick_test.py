#!/usr/bin/env python3
"""
quick_test.py — YOLO-World manuel görsel test aracı.

Kullanım:
  python3 quick_test.py bilgisayar.jpg
  python3 quick_test.py bilgisayar.jpg --prompt "a laptop"
  python3 quick_test.py bilgisayar.jpg --prompt "a vessel" --conf 0.2
  python3 quick_test.py bilgisayar.jpg --save sonuc.jpg   # ekran açmak yerine kaydet

Gereksinimler:
  pip3 install ultralytics opencv-python
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def draw_results(img, results, prompt: str, conf_thresh: float):
    """Bounding box ve label'ları görüntüye çiz."""
    detections = []

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append((x1, y1, x2, y2, conf))

    # Arka plan bilgi kutusu
    info = f'Prompt: "{prompt}" | Conf >= {conf_thresh} | Tespit: {len(detections)}'
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), (30, 30, 30), -1)
    cv2.putText(img, info, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    for x1, y1, x2, y2, conf in detections:
        # Bounding box — turuncu
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 3)

        # Label arka planı
        label = f'{prompt}  {conf:.0%}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y = max(y1 - 10, th + 10)
        cv2.rectangle(img, (x1, label_y - th - 6), (x1 + tw + 6, label_y + 2),
                      (0, 165, 255), -1)
        cv2.putText(img, label, (x1 + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Köşe noktası (merkez)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)

    if not detections:
        cv2.putText(img, 'Hicbir sey bulunamadi — conf degerini dusur veya prompt degistir',
                    (10, img.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2, cv2.LINE_AA)

    return img, len(detections)


def main():
    parser = argparse.ArgumentParser(description='YOLO-World görsel test aracı')
    parser.add_argument('image', help='Test edilecek görüntü dosyası')
    parser.add_argument('--prompt', default='a laptop',
                        help='Aranacak nesne (örn: "a vessel", "a laptop", "a person")')
    parser.add_argument('--conf', type=float, default=0.2,
                        help='Minimum confidence eşiği (default: 0.2)')
    parser.add_argument('--model', default='yolov8s-worldv2.pt',
                        help='YOLO-World model dosyası')
    parser.add_argument('--save', default=None,
                        help='Sonucu bu dosyaya kaydet (default: ekranda göster)')
    args = parser.parse_args()

    # Görüntüyü oku
    img_path = Path(args.image)
    if not img_path.exists():
        print(f'HATA: Dosya bulunamadı: {img_path}')
        sys.exit(1)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f'HATA: Görüntü okunamadı: {img_path}')
        sys.exit(1)

    print(f'Görüntü: {img_path} ({img.shape[1]}x{img.shape[0]})')
    print(f'Prompt : "{args.prompt}"')
    print(f'Model  : {args.model}')
    print(f'Conf   : {args.conf}')
    print()

    # Model yükle
    try:
        from ultralytics import YOLO
    except ImportError:
        print('HATA: ultralytics kurulu değil.')
        print('Kur:  pip3 install ultralytics')
        sys.exit(1)

    print('Model yükleniyor...')
    model = YOLO(args.model)
    model.set_classes([args.prompt])

    # Inference
    print('Inference çalışıyor...')
    results = model.predict(img, conf=args.conf, verbose=False)

    # Sonuçları çiz
    output = img.copy()
    output, n = draw_results(output, results, args.prompt, args.conf)

    print(f'Tespit edilen nesne sayısı: {n}')
    if results and results[0].boxes is not None:
        for i, box in enumerate(results[0].boxes):
            conf = float(box.conf[0])
            if conf >= args.conf:
                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                print(f'  [{i+1}] box={xyxy}  conf={conf:.2%}')

    # Kaydet veya göster
    if args.save:
        cv2.imwrite(args.save, output)
        print(f'\nSonuç kaydedildi: {args.save}')
    else:
        # Büyük görüntüleri ekrana sığdır
        max_w, max_h = 1400, 900
        h, w = output.shape[:2]
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            output = cv2.resize(output, (int(w * scale), int(h * scale)))

        print('\nEkranda gösteriliyor — kapatmak için herhangi bir tuşa bas.')
        window_name = f'YOLO-World: "{args.prompt}"'
        cv2.imshow(window_name, output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
