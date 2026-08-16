"""ROS 2 node: refine a detection's color label with YCrCb thresholding.

Bolted onto the pipeline as its own, independent enrichment stage -- same shape
as overlay_node's image<->tracks stamp matching -- so it never touches
depth_fusion_node. Reads the ZED color image + a Detection2DArray, classifies
each colorable detection's crop with color_classifier, and majority-votes the
result (label_vote.LabelVote) before publishing a refined copy of the message
with the color-bearing part of the label corrected.

Only labels in ``colorable_labels`` (default: buoy, red_buoy, green_buoy,
black_buoy) are touched -- cardinal-mark classes (east/north/south/west_buoy)
pass through unchanged, since a cardinal mark is decided by its topmark, not its
colour. ``buoy`` is included by default so this pairs directly with
class_remap_node: if RT-DETR's 7 trained classes are collapsed to a single
generic "buoy" upstream (no retraining needed -- see class_remap.yaml), this node
is what actually decides red vs green vs black.

Already-coloured labels are colorable too, so a decision can be revised while the
vote is still open; once the vote freezes, further frames cannot change it.

Per-colour confidence (``min_confidence_overrides``)
----------------------------------------------------
The colours are not equally risky to get wrong. Red and green sit far out on the
Cr axis, so a patch either has that chroma or it does not. Black sits AT neutral
chroma and is identified by low luma -- which any dark, washed-out, shadowed
patch also looks like. So black has to show more evidence: the default
``black:0.30`` raises its bar from the global 0.12.

Measured on the rig for context: a real black buoy scores ~0.64, a green one
~0.99, a red one ~0.30. So 0.30 leaves black about 2x headroom on real hardware
while refusing the thin, ambiguous cases.

Failing the bar is NOT a vote for another colour -- classify_color picks the
highest scorer first and only then applies that colour's threshold, so a
too-thin black becomes "uncertain" (no vote this frame), never "green".

Pixel-reading additions (``use_white_balance``/``use_clahe``/``use_glare_mask``/
``use_hsv_vote``)
----------------------------------------------------------------------------
Validated against real buoy photos + hand-labeled ground truth
(scripts/test_buoy_folder.py --ground-truth) before landing here. None of them
touch color_ranges.yaml's calibrated numbers -- they only change how pixels are
read before those numbers are applied. All default ON. See color_classifier.py's
module docstring for exactly what each one does; short version: white
balance + CLAHE run on the whole frame in ``on_image`` (before cropping), glare
exclusion + the HSV rescue/confirm vote run inside ``classify_color`` on each
crop. Every one of them is also a parameter, so they can be switched off
individually without a code change if a real deployment ever disagrees with the
benchmark; the previous call sites are kept as comments for a full revert.

Name space vs index space (``class_labels_file``)
-------------------------------------------------
This node decides in NAME space -- ``colorable_labels`` and ``<color>_buoy`` are
names. RT-DETR, however, puts the numeric class INDEX in ``class_id`` ("4"), and
depth_fusion_node downstream does ``int(class_id)``. The index->name step used to
happen in tracker_node, upstream of here; that node is gone and the mapping now
lives downstream instead, so this node has to bridge the two itself:

  class_labels_file EMPTY  -> class_id is read and written as a name. Standalone
                              mode; this is what the unit tests exercise.
  class_labels_file SET    -> class_id is read as an index, translated to a name
                              to decide, and the refined name is written back as
                              an index. This is what tracking.launch.py uses.

The bridge fails safe in both directions: an index with no configured name is
treated as not-colorable (passes through), and a refined name with no configured
index is left alone rather than written out as a name into an index-space topic,
which would make depth_fusion_node's int() raise on every frame.

Vote keying (``vote_key``)
--------------------------
LabelVote needs something to key a vote on. It was written against tracker_node's
per-track ``Detection2D.id``, which no longer exists -- raw RT-DETR detections
carry an empty id, so keying on it would merge every box in a frame into one
shared vote. Hence:

  id    Key on ``Detection2D.id`` (the original behaviour). Only meaningful if
        something upstream actually assigns ids.
  grid  Key on the box centre quantised to ``vote_cell_px``. A buoy holds its
        cell for far longer than the ~3 frames a vote needs at 45 Hz, and a cell
        that goes unseen for a single frame is garbage-collected, so a frozen
        vote cannot outlive the object that produced it by more than one frame.
        This is the launch default.
  none  No voting; decide from the current frame alone.

``grid`` substitutes proximity for identity, which is weaker than a real track
id: two same-cell objects in consecutive frames share a vote. The right long-term
home for label voting is buoy_mapper_node on the host, which has genuine
world-frame identity -- LabelVote is deliberately pure Python with zero ROS
imports so it can move there unchanged.
"""
from __future__ import annotations

from collections import deque

import rclpy
import yaml
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from .color_classifier import (
    classify_color, clahe_luma, gray_world_white_balance, load_color_ranges,
)
from .label_vote import LabelVote

VOTE_KEYS = ('id', 'grid', 'none')


def _desc(text):
    return ParameterDescriptor(description=text)


class ColorClassificationNode(Node):
    def __init__(self, **kwargs):
        super().__init__('color_classification_node', **kwargs)
        p = self.declare_parameter
        self.buffer_len = p('image_buffer', 15, _desc('images buffered for stamp matching')).value
        self.match_tol_ms = p('match_tolerance_ms', 100.0,
                              _desc('max stamp diff to accept an image<->tracks match (ms)')).value
        self.roi_shrink = p('roi_shrink', 0.6, _desc('central crop fraction sampled for color')).value
        self.min_confidence = p('min_confidence', 0.12,
                                _desc('min in-range pixel ratio to trust a color observation')).value
        overrides_raw = p('min_confidence_overrides', 'black:0.30',
                          _desc('per-colour confidence overrides, "colour:threshold" separated by '
                                'commas; empty = min_confidence applies to every colour')).value
        # Ön işleme (tam kareye uygulanır, on_image'da) + classify_color'ın glare/HSV
        # eklentileri (crop'a uygulanır, _decide_colour'da) -- hepsi varsayılan AÇIK,
        # renk aralıklarına (color_ranges_file) DOKUNMAZLAR, sadece pikseller nasıl
        # okunuyor onu değiştirirler. bkz. color_classifier.py'nin modül docstring'i.
        self.use_white_balance = p('use_white_balance', True,
                                   _desc('gray-world white balance on each frame before cropping')).value
        self.use_clahe = p('use_clahe', True,
                           _desc('CLAHE on the Y channel only, before cropping')).value
        self.use_glare_mask = p('use_glare_mask', True,
                                _desc('exclude near-blown-out (specular highlight) pixels from '
                                      'every color mask')).value
        self.glare_y_thresh = p('glare_y_thresh', 245,
                                _desc('Y value at/above which a pixel counts as glare')).value
        self.use_hsv_vote = p('use_hsv_vote', True,
                              _desc('HSV hue check alongside YCrCb: rescue/confirm for red, '
                                    'confirm-only for green')).value
        self.hsv_red_min_ratio = p('hsv_red_min_ratio', 0.10,
                                   _desc('min HSV red-hue ratio for the red rescue/confirm vote')).value
        self.hsv_green_min_ratio = p('hsv_green_min_ratio', 0.10,
                                     _desc('min HSV green-hue ratio to confirm a YCrCb green pick')).value
        self.min_votes = p('min_votes', 3, _desc('observations before a track color is frozen')).value
        color_ranges_file = p('color_ranges_file', '',
                              _desc('YAML: color -> [[y_min,y_max,cr_min,cr_max,cb_min,cb_max], ...]')).value
        colorable = p('colorable_labels', ['buoy', 'red_buoy', 'green_buoy', 'black_buoy'],
                     _desc('class names eligible for color refinement; others pass through untouched')).value
        self.colorable = set(colorable)
        labels_file = p('class_labels_file', '',
                        _desc('YAML index->name. Empty = class_id is a name; set = class_id is '
                              'a numeric index and is translated in and out')).value
        self.vote_key_mode = str(p('vote_key', 'id',
                                   _desc('id|grid|none — what a colour vote is keyed on')).value).lower()
        self.vote_cell_px = float(p('vote_cell_px', 64.0,
                                    _desc('grid cell size (px) when vote_key=grid')).value)

        if self.vote_key_mode not in VOTE_KEYS:
            raise RuntimeError(
                f'vote_key must be one of {VOTE_KEYS} (got: {self.vote_key_mode!r}) -- a typo here '
                'silently falling back to per-frame decisions would look like the vote working')
        if self.vote_key_mode == 'grid' and self.vote_cell_px <= 0.0:
            raise RuntimeError(f'vote_cell_px must be > 0 (got: {self.vote_cell_px})')

        if not color_ranges_file:
            raise RuntimeError("color_classification_node requires the 'color_ranges_file' parameter")
        self.color_ranges = load_color_ranges(color_ranges_file)
        # 'red' -> 'red_buoy' -- assumes the "<color>_buoy" naming already used by class_labels.yaml.
        self.color_to_label = {c: f'{c}_buoy' for c in self.color_ranges}

        # Per-colour confidence, resolved once so classify_color just looks it up.
        self.min_confidence_per_color = self._parse_overrides(overrides_raw)

        # Empty dicts mean "class_id is already a name" -- see the module docstring.
        self.index_to_name, self.name_to_index = self._load_labels(labels_file)

        self.bridge = CvBridge()
        self.images = deque(maxlen=int(self.buffer_len))   # (stamp_ns, bgr image)
        self.votes: dict[str, LabelVote] = {}               # vote key -> LabelVote
        self._warned_unmapped_index = set()
        self._warned_unmapped_name = set()

        img_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                             depth=int(self.buffer_len), durability=DurabilityPolicy.VOLATILE)
        track_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                               depth=10, durability=DurabilityPolicy.VOLATILE)
        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                             depth=10, durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(Image, '~/image', self.on_image, img_qos)
        self.create_subscription(Detection2DArray, '~/detections_input', self.on_tracks, track_qos)
        self.pub = self.create_publisher(Detection2DArray, '~/detections_color_refined', pub_qos)
        self._miss = 0
        space = f'index ({len(self.index_to_name)} labels)' if self.index_to_name else 'name'
        vote = (f'{self.vote_key_mode}/{self.vote_cell_px:.0f}px'
                if self.vote_key_mode == 'grid' else self.vote_key_mode)
        self.get_logger().info(
            f'color_classification_node up: colors={list(self.color_ranges)} '
            f'colorable={sorted(self.colorable)} class_id_space={space} '
            f'vote_key={vote} min_votes={self.min_votes} '
            f'roi_shrink={self.roi_shrink} '
            f'min_confidence={ {c: round(t, 3) for c, t in sorted(self.min_confidence_per_color.items())} } '
            f'white_balance={self.use_white_balance} clahe={self.use_clahe} '
            f'glare_mask={self.use_glare_mask}(Y>={self.glare_y_thresh}) '
            f'hsv_vote={self.use_hsv_vote}(red>={self.hsv_red_min_ratio} green>={self.hsv_green_min_ratio})')

    # ---------------------------------------------------------------- confidence
    def _parse_overrides(self, raw: str) -> dict[str, float]:
        """"black:0.30,red:0.2" -> {every colour: its threshold}.

        Every configured colour gets an entry, so classify_color never has to fall
        back to a module default that might disagree with this node's
        ``min_confidence``.

        Malformed syntax raises -- "red", "red:high" and "red:30" are unambiguous
        mistakes, and an override exists to make a colour HARDER to claim, so one
        that silently reverts to the low global bar would be invisible in exactly
        the case it guards.

        A well-formed override for a colour the config does not define only WARNS.
        It has to: the default is ``black:0.30``, and a perfectly valid
        color_ranges.yaml with just red and green would otherwise stop the node
        from starting. A default must never break a valid configuration -- so this
        one degrades to a log line that still catches a typo.
        """
        thresholds = {c: float(self.min_confidence) for c in self.color_ranges}
        for entry in (e.strip() for e in str(raw).split(',')):
            if not entry:
                continue
            colour, _, value = entry.partition(':')
            colour = colour.strip()
            if not _:
                raise RuntimeError(
                    f'min_confidence_overrides entry {entry!r} is not "colour:threshold"')
            try:
                threshold = float(value)
            except ValueError:
                raise RuntimeError(
                    f'min_confidence_overrides: {value!r} in {entry!r} is not a number') from None
            if not 0.0 <= threshold <= 1.0:
                raise RuntimeError(
                    f'min_confidence_overrides: {threshold} in {entry!r} is outside [0, 1] -- '
                    'it is an in-range PIXEL RATIO, not a percentage')
            if colour not in self.color_ranges:
                self.get_logger().warn(
                    f'min_confidence_overrides names colour {colour!r}, which is not in '
                    f'color_ranges_file (has: {sorted(self.color_ranges)}) -- ignoring it. '
                    'Expected if this config simply has no such colour; check the spelling '
                    'if you did mean to raise its bar.')
                continue
            thresholds[colour] = threshold
        return thresholds

    # ---------------------------------------------------------------- label space
    def _load_labels(self, path):
        """(index->name, name->index), or two empty dicts when no file is configured.

        A failed load returns empty dicts, which drops the node back to name space.
        That is the safe direction: in name space nothing matches a numeric class_id,
        so the node degrades to a pass-through instead of writing names onto an
        index-space topic that depth_fusion_node would then fail to int().
        """
        if not path:
            return {}, {}
        try:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            forward = {int(k): str(v) for k, v in raw.items()}
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(
                f'labels load failed {path}: {e}; falling back to name-space class_id '
                '(this node will pass numeric class ids through unrefined)')
            return {}, {}
        self.get_logger().info(f'loaded {len(forward)} class labels from {path}')
        return forward, {v: k for k, v in forward.items()}

    def _to_name(self, class_id):
        """Wire class_id -> label name, or None if it can't be named (never colorable)."""
        if not self.index_to_name:
            return class_id
        try:
            index = int(class_id)
        except (ValueError, TypeError):
            return None
        name = self.index_to_name.get(index)
        if name is None and index not in self._warned_unmapped_index:
            self._warned_unmapped_index.add(index)
            self.get_logger().warn(
                f'class index {index} not in class_labels_file -> not colorable, passing through')
        return name

    def _to_class_id(self, name, original_class_id):
        """Label name -> wire class_id, falling back to the ORIGINAL value.

        A colour whose ``<colour>_buoy`` name has no configured index (a colour added
        to color_ranges.yaml but not to class_labels.yaml) must not be written out as
        a name here -- downstream reads this field with int(). Keeping the original
        index loses the refinement, which is strictly better than breaking the frame.
        """
        if not self.name_to_index:
            return name
        index = self.name_to_index.get(name)
        if index is None:
            if name not in self._warned_unmapped_name:
                self._warned_unmapped_name.add(name)
                self.get_logger().warn(
                    f'refined colour "{name}" has no index in class_labels_file -> keeping the '
                    'original class id (add it there to let this refinement through)')
            return original_class_id
        return str(index)

    # ---------------------------------------------------------------- vote keying
    def _vote_key(self, d):
        """Key this detection's colour vote, or None when voting is disabled.

        ``grid`` quantises the box centre: consecutive frames of the same buoy land
        in the same cell and therefore share a vote, which is what stands in for the
        track id LabelVote was originally keyed on.
        """
        if self.vote_key_mode == 'id':
            return d.id
        if self.vote_key_mode == 'grid':
            cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
            return f'{int(cx // self.vote_cell_px)}:{int(cy // self.vote_cell_px)}'
        return None

    def _seen_keys(self, msg: Detection2DArray) -> set:
        """Vote keys this message accounts for -- what _gc_votes keeps.

        Built from EVERY detection, not just the colorable ones: a non-colorable
        detection still occupies its grid cell, and keeping that cell's entry alive
        is the honest reading of "something is still there".
        """
        return {k for k in (self._vote_key(d) for d in msg.detections) if k is not None}

    # ---------------------------------------------------------------- image buffer
    @staticmethod
    def _ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def on_image(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'cv_bridge failed ({msg.encoding}): {e}')
            return

        # ESKİ (ön işlemesiz) davranış -- geri dönmek istersen bunu aç, alttaki
        # ön işleme bloğunu kapat. Ya da kod değiştirmeden use_white_balance=false
        # use_clahe=false parametreleriyle aynı sonuca ulaşırsın:
        # self.images.append((self._ns(msg.header.stamp), img))

        # YENİ: gray-world beyaz dengesi + CLAHE(Y) -- TAM karede, crop'lanmadan
        # ÖNCE. classify_color'ın crop-bazlı glare/HSV eklentilerinin aksine bunlar
        # tüm kareyi görmeye ihtiyaç duyar, o yüzden burada, buffer'a girmeden yapılır.
        if self.use_white_balance:
            img = gray_world_white_balance(img)
        if self.use_clahe:
            img = clahe_luma(img)
        self.images.append((self._ns(msg.header.stamp), img))

    def _match_image(self, stamp_ns: int):
        """Return the buffered image whose stamp equals (or is closest within
        tolerance to) stamp_ns. Same matching rule as overlay_node."""
        if not self.images:
            return None
        best, best_d = None, None
        for ns, img in self.images:
            if ns == stamp_ns:
                return img
            d = abs(ns - stamp_ns)
            if best_d is None or d < best_d:
                best, best_d = img, d
        return best if best_d is not None and best_d <= self.match_tol_ms * 1e6 else None

    # ---------------------------------------------------------------- main callback
    def _decide_colour(self, d, image):
        """Colour for one detection, or None if undecided.

        ``image`` None means no frame matched this message's stamp, so there is no
        new observation to make. That is NOT a reason to throw away a decision the
        vote already reached: surviving a dropped frame is precisely what freezing
        a vote is for. Republishing the detector's own colour guess instead made
        the published label flicker between the detector's class and the refined
        one at exactly the rate the image stream dropped frames (~16 % of frames
        on the Jetson, where four nodes share the full-res colour image over UDP).
        """
        key = self._vote_key(d)
        if image is None:
            vote = self.votes.get(key)
            return vote.current_best if vote is not None else None

        h_img, w_img = image.shape[:2]
        cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
        w, h = d.bbox.size_x, d.bbox.size_y
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(w_img, int(cx + w / 2))
        y2 = min(h_img, int(cy + h / 2))
        crop = image[y1:y2, x1:x2]

        # ESKİ çağrı (glare maskesi/HSV oylaması yok) -- geri dönmek istersen bunu
        # aç, alttakini kapat: result = classify_color(crop, self.color_ranges,
        #     self.roi_shrink, self.min_confidence_per_color)
        result = classify_color(crop, self.color_ranges, self.roi_shrink,
                                self.min_confidence_per_color,
                                use_glare_mask=self.use_glare_mask,
                                glare_y_thresh=self.glare_y_thresh,
                                use_hsv_vote=self.use_hsv_vote,
                                hsv_red_min_ratio=self.hsv_red_min_ratio,
                                hsv_green_min_ratio=self.hsv_green_min_ratio)
        if key is None:                      # vote_key='none' -- this frame decides alone
            return result.label
        vote = self.votes.setdefault(key, LabelVote(min_votes=self.min_votes))
        vote.add(result.label)
        return vote.current_best

    def on_tracks(self, msg: Detection2DArray):
        image = self._match_image(self._ns(msg.header.stamp))
        if image is None:
            self._miss += 1
            if self._miss % 30 == 1:
                self.get_logger().warn(
                    f'no image within tolerance for a tracks stamp (count={self._miss}); '
                    'applying frozen votes only, adding none')

        out = Detection2DArray()
        out.header = msg.header
        for d in msg.detections:
            original_class_id = d.results[0].hypothesis.class_id if d.results else None
            original_label = self._to_name(original_class_id) if d.results else None
            if original_label not in self.colorable:
                out.detections.append(d)
                continue

            refined_color = self._decide_colour(d, image)
            if refined_color is not None:
                refined_label = self.color_to_label.get(refined_color)
                if refined_label is not None and refined_label != original_label:
                    d.results[0].hypothesis.class_id = self._to_class_id(
                        refined_label, original_class_id)
            out.detections.append(d)

        # GC on both paths. The detections message is authoritative about what is
        # still present whether or not an image turned up to classify it, and
        # skipping this leaked a vote per vanished object for as long as the image
        # stream was unhealthy -- exactly when votes are most stale.
        self._gc_votes(self._seen_keys(msg))
        self.pub.publish(out)

    def _gc_votes(self, seen: set):
        """Drop vote state for every key that this frame did not produce.

        With vote_key='grid' this is what stops a frozen vote from outliving the
        object that produced it: a cell the objects have moved out of is cleared on
        the very next frame, so a different buoy that later enters that cell starts
        its own vote from scratch rather than inheriting a stale frozen colour.
        """
        for key in [k for k in self.votes if k not in seen]:
            del self.votes[key]


def main(args=None):
    rclpy.init(args=args)
    node = ColorClassificationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
