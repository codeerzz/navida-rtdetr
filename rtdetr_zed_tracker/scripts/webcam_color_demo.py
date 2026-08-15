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

  # ZED'in kendisinden, pipeline CALISIRKEN (container icinde) -- asagiya bak:
  python3 webcam_color_demo.py --ros-topic /zed_node/left/image_rect_color --calibrate green

ZED'den kalibrasyon (--ros-topic)
---------------------------------
config/color_ranges.yaml'i GERCEK kameradan kalibre etmek icin tek dogru yol
budur. Sebebi: kalibrasyonun amaci kameranin kendi beyaz dengesi + pozlamasi
altinda rengin nereye dustugunu olcmek. Dizustunun webcam'i ile kalibre edersen
WEBCAM'in renk tepkisini olcmus olursun, ZED'inkini degil -- ve yaml robotta
kullanilir. ZED'in ham UVC karesi de olmaz: o, image_rect_color'in gectigi ZED
ISP/rectification hattindan gecmemistir.

Ayrica ZED'i zed_node aciyor, yani pipeline calisirken --camera ile ikinci bir
process onu zaten acamaz. --ros-topic bu yuzden var: goruntuyu kameradan degil,
zaten yayinlanan topic'ten alir, boylece stack calisirken kalibre edebilirsin.

CONTAINER ICINDE calistir (rclpy + cv_bridge orada) ve UDP profilini export et --
etmezsen HATA VERMEDEN sifir kare gelir (bkz. NOTES.md §7):

  docker exec -it -u admin -w /workspaces/isaac_ros-dev \
    isaac_ros_dev-aarch64-container bash -lc '
      source /opt/ros/humble/setup.bash
      source install/setup.bash
      export FASTRTPS_DEFAULT_PROFILES_FILE=$ISAAC_ROS_WS/src/rtdetr_zed_tracker/udp_only_profile.xml
      python3 src/rtdetr_zed_tracker/scripts/webcam_color_demo.py \
        --ros-topic /zed_node/left/image_rect_color --calibrate green'

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
import time
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


# ── kare kaynaklari ───────────────────────────────────────────────────────────
# Kalibrasyon dongusunun (ROI secimi, c/z/p/x, cakisma uyarisi) kareyi NEREDEN
# aldigi onemli degil -- tek ihtiyaci read(). Kaynagi ayirmak, ayni test edilmis
# akisin webcam'de de, ZED topic'inde de, durağan bir goruntude de birebir ayni
# calismasi demek; her kaynak icin ayri bir dongu kopyalanmis olsaydi ucu de
# zamanla birbirinden ayrisirdi.
class _CameraSource:
    """Yerel bir kamera (cv2.VideoCapture)."""

    def __init__(self, camera_index: int):
        self.cap = cv2.VideoCapture(camera_index)
        self.index = camera_index

    def ok(self) -> bool:
        return self.cap.isOpened()

    def why_not(self) -> str:
        return f'kamera acilamadi (index={self.index}). Baska bir --camera indexi dene.'

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class _StaticImageSource:
    """Tek bir goruntu dosyasi, sonsuza kadar ayni kare.

    Kalibrasyon icin de gecerli bir kaynak: kaydedilmis bir ZED karesinden ROI
    secip ornek yakalamak, canli akistan yakalamakla ayni seydir. Tek farki, tek
    isik kosulu -- birden fazla kosul icin birden fazla kare gerekir.
    """

    def __init__(self, image_path: Path):
        self.frame = cv2.imread(str(image_path))
        self.path = image_path

    def ok(self) -> bool:
        return self.frame is not None

    def why_not(self) -> str:
        return f'goruntu okunamadi: {self.path}'

    def read(self):
        return True, self.frame.copy()

    def release(self):
        pass


class _RosTopicSource:
    """Yayinlanmakta olan bir sensor_msgs/Image topic'i.

    ROS importlari MAHSUS burada, fonksiyon icinde: script'in webcam ve --image
    yollari ROS'suz bir dizustunde de calismali, ve modul seviyesinde bir rclpy
    importu bunu imkansiz kilardi.
    """

    def __init__(self, topic: str, timeout_s: float = 15.0):
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.qos import (
            DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
        )
        from sensor_msgs.msg import Image

        self._rclpy = rclpy
        self._bridge = CvBridge()
        self._latest = None
        self._topic = topic
        self._timeout_s = timeout_s
        self._warned_empty = False

        rclpy.init()
        self._node = rclpy.create_node('color_calibration_viewer')
        # RELIABLE, cunku zed_node RELIABLE yayinliyor -- BEST_EFFORT bir abone
        # ondan hicbir sey almaz. (overlay_node ve color_classification_node da
        # ayni sebeple RELIABLE.)
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=2,
                         durability=DurabilityPolicy.VOLATILE)
        self._node.create_subscription(Image, topic, self._on_image, qos)

    def _on_image(self, msg):
        try:
            self._latest = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:  # noqa: BLE001
            if not self._warned_empty:
                self._warned_empty = True
                print(f'HATA: cv_bridge kareyi cozemedi ({msg.encoding}): {e}', flush=True)

    def ok(self) -> bool:
        """Ilk kare gelene kadar bekle. Gelmezse sessizce beklemek yerine bunu
        rapor et -- bu senaryodaki en yaygin hata (UDP profili export edilmemis)
        HICBIR hata vermez, sadece sonsuza kadar bos akar."""
        print(f'"{self._topic}" bekleniyor (en fazla {self._timeout_s:.0f} s)...', flush=True)
        deadline = time.time() + self._timeout_s
        while time.time() < deadline and self._latest is None:
            self._rclpy.spin_once(self._node, timeout_sec=0.1)
        return self._latest is not None

    def why_not(self) -> str:
        return (f'"{self._topic}" uzerinden {self._timeout_s:.0f} s icinde kare gelmedi.\n'
                f'  Sirasiyla kontrol et:\n'
                f'   1) FASTRTPS_DEFAULT_PROFILES_FILE export edildi mi? Edilmediyse hata\n'
                f'      VERMEDEN sifir mesaj gelir -- bu senaryodaki en yaygin sebep budur.\n'
                f'   2) Pipeline calisiyor mu?  ros2 topic hz {self._topic}\n'
                f'   3) Topic adi dogru mu?     ros2 topic list | grep image\n'
                f'   4) Bu script container ICINDE mi calisiyor? (rclpy + cv_bridge orada)')

    def read(self):
        self._rclpy.spin_once(self._node, timeout_sec=0.05)
        if self._latest is None:
            return False, None
        return True, self._latest.copy()

    def release(self):
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


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


def run(source, ranges_path: Path, roi_shrink: float, min_confidence: float,
       calibrate_color: str | None):
    color_ranges = load_color_ranges(str(ranges_path))
    print(f'Yuklenen renkler: {list(color_ranges)}  ({ranges_path})', flush=True)
    if calibrate_color is not None:
        print(f'Kalibrasyon modu: "{calibrate_color}". c=yakala, p=araligi yazdir, x=temizle.', flush=True)
        if calibrate_color not in color_ranges:
            # Yeni bir renk eklemek gecerli bir kullanim, ama yazim hatasi da ayni
            # sekilde gorunur -- ve yanlis yazilmis bir renkte cakisma kontrolu
            # sessizce ise yaramaz hale gelir.
            print(f'  NOT: "{calibrate_color}" {ranges_path.name} icinde YOK. Yeni renk ekliyorsan '
                  f'sorun degil; degilse yazimi kontrol et (mevcutlar: {list(color_ranges)}).',
                  flush=True)
    else:
        print('Kalibrasyon modu KAPALI (--calibrate <renk> vermedin) -- c/p/x/z tuslari calismaz, '
              'sadece etiket goruntulenir.', flush=True)
    print('NOT: klavye kisayollari calismasi icin video PENCERESI aktif olmali -- tuslara basmadan '
          'once fare ile video penceresine tikla (terminale degil).', flush=True)

    if not source.ok():
        print(f'HATA: {source.why_not()}', flush=True)
        source.release()
        return 1

    roi = None
    samples = []
    window = 'color_classifier canli test'
    cv2.namedWindow(window)

    while True:
        ok, frame = source.read()
        if not ok:
            print('HATA: kaynaktan kare okunamadi.')
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

    source.release()
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
                   help='kamera yerine tek bir goruntu dosyasi (--calibrate yoksa pencere acmaz)')
    p.add_argument('--ros-topic', type=str, default=None, metavar='TOPIC',
                   help='kareleri yayinlanan bir sensor_msgs/Image topic ten al, orn. '
                        '/zed_node/left/image_rect_color -- ZED den kalibrasyon icin bunu kullan')
    p.add_argument('--ros-timeout', type=float, default=15.0,
                   help='--ros-topic ilk kareyi bu kadar saniye bekler (varsayilan: 15)')
    p.add_argument('--ranges', type=Path, default=DEFAULT_RANGES, help='color_ranges.yaml yolu')
    p.add_argument('--roi-shrink', type=float, default=0.6, help='ROI ortasindan orneklenen oran')
    p.add_argument('--min-confidence', type=float, default=0.12, help='belirsiz sayilma esigi')
    p.add_argument('--calibrate', type=str, default=None, metavar='RENK',
                   help='kalibrasyon modu: bu renk icin ornek biriktir (orn. --calibrate red)')
    args = p.parse_args()

    if not args.ranges.exists():
        print(f'HATA: renk config dosyasi bulunamadi: {args.ranges}')
        return 1

    if args.image is not None and args.ros_topic is not None:
        print('HATA: --image ve --ros-topic birlikte kullanilamaz -- kare kaynagi tek olmali.')
        return 2

    # --image, --calibrate YOKKEN tek seferlik ve penceresiz kalir (headless/CI icin).
    # --calibrate VARSA ayni interaktif donguye girer: eskiden bu kombinasyon
    # sessizce kalibrasyonsuz calisiyordu, yani kullanici ornek yakaladigini
    # saniyor ama c/z/p/x hicbir sey yapmiyordu.
    if args.image is not None and args.calibrate is None:
        return run_single_image(args.image, args.ranges, args.roi_shrink, args.min_confidence)

    if args.ros_topic is not None:
        try:
            source = _RosTopicSource(args.ros_topic, args.ros_timeout)
        except Exception as e:  # noqa: BLE001 -- ImportError (ROS yok), ama cv_bridge'in
            # numpy ABI uyusmazligi AttributeError atiyor ve rclpy.init() da kendi
            # hatalarini atabiliyor. Hepsinin cevabi ayni: yanlis yerde calistiriyorsun.
            print(f'HATA: --ros-topic icin rclpy + cv_bridge gerekli, hazirlanamadi: '
                  f'{type(e).__name__}: {e}\n'
                  '  Bu scripti CONTAINER ICINDE calistir (ROS orada kurulu):\n'
                  '    docker exec -it -u admin -w /workspaces/isaac_ros-dev \\\n'
                  '      isaac_ros_dev-aarch64-container bash -lc \'...\'')
            return 1
    elif args.image is not None:
        source = _StaticImageSource(args.image)
    else:
        source = _CameraSource(args.camera)

    return run(source, args.ranges, args.roi_shrink, args.min_confidence, args.calibrate)


if __name__ == '__main__':
    sys.exit(main())
