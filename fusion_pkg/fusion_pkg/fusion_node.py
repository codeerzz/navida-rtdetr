#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import message_filters
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
import cv_bridge
import numpy as np


def iou(box_a, box_b):
    # box format: (x1, y1, x2, y2)
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


class SimpleTracker:
    """Basit IoU tabanli multi-object tracker.
    Her nesneye kalici bir ID atar, birkac frame kaybolursa ID'yi tutar.
    """

    def __init__(self, iou_threshold=0.3, max_missed=10):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = {}  # id -> {"box": (x1,y1,x2,y2), "missed": int, "class_id": str}
        self.next_id = 0

    def update(self, detections):
        # detections: list of dicts {"box": (x1,y1,x2,y2), "class_id": str}
        unmatched_tracks = set(self.tracks.keys())
        unmatched_dets = list(range(len(detections)))
        matches = []

        # basit greedy eslestirme (en yuksek IoU'dan basla)
        pairs = []
        for tid, track in self.tracks.items():
            for di, det in enumerate(detections):
                score = iou(track["box"], det["box"])
                if score >= self.iou_threshold:
                    pairs.append((score, tid, di))
        pairs.sort(reverse=True, key=lambda p: p[0])

        used_tracks = set()
        used_dets = set()
        for score, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            matches.append((tid, di))
            used_tracks.add(tid)
            used_dets.add(di)

        unmatched_tracks -= used_tracks
        unmatched_dets = [d for d in unmatched_dets if d not in used_dets]

        # eslesenleri guncelle
        for tid, di in matches:
            self.tracks[tid]["box"] = detections[di]["box"]
            self.tracks[tid]["class_id"] = detections[di]["class_id"]
            self.tracks[tid]["missed"] = 0

        # eslesmeyen track'lerin missed sayacini arttir
        for tid in unmatched_tracks:
            self.tracks[tid]["missed"] += 1

        # cok kaybolanlari sil
        to_delete = [tid for tid, t in self.tracks.items() if t["missed"] > self.max_missed]
        for tid in to_delete:
            del self.tracks[tid]

        # eslesmeyen detection'lar icin yeni track ac
        new_ids_for_dets = {}
        for di in unmatched_dets:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "box": detections[di]["box"],
                "class_id": detections[di]["class_id"],
                "missed": 0,
            }
            new_ids_for_dets[di] = tid

        # her detection index'i icin son atanan track id'yi don
        result = {}
        for tid, di in matches:
            result[di] = tid
        for di, tid in new_ids_for_dets.items():
            result[di] = tid

        return result  # {detection_index: track_id}


class FusionNode(Node):
    QUEUE_SIZE = 10

    def __init__(self):
        super().__init__('fusion_node')
        self._bridge = cv_bridge.CvBridge()
        self.tracker = SimpleTracker(iou_threshold=0.3, max_missed=10)

        self._det_sub = message_filters.Subscriber(
            self, Detection2DArray, 'detections_output')
        self._depth_sub = message_filters.Subscriber(
            self, Image, '/zed_node/depth/depth_registered')

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self._det_sub, self._depth_sub], self.QUEUE_SIZE, slop=0.1)
        self.sync.registerCallback(self.callback)

        self.get_logger().info('Fusion node baslatildi (tracker + depth)')

    def callback(self, detections_msg, depth_msg):
        depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(
            depth_msg.height, depth_msg.width)

        det_list = []
        for det in detections_msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            box = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
            class_id = det.results[0].hypothesis.class_id if det.results else "unknown"
            det_list.append({"box": box, "class_id": class_id, "cx": cx, "cy": cy})

        id_map = self.tracker.update(det_list)

        for di, det in enumerate(det_list):
            track_id = id_map.get(di, -1)
            u = int(det["cx"])
            v = int(det["cy"])

            z = float('nan')
            if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
                z = depth[v, u]

            self.get_logger().info(
                f'Track ID={track_id} class={det["class_id"]} depth={z:.2f}m')


def main():
    rclpy.init()
    node = FusionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
