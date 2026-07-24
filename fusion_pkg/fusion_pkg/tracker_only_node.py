#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray


def iou(box_a, box_b):
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
    def __init__(self, iou_threshold=0.3, max_missed=10):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = {}
        self.next_id = 0

    def update(self, detections):
        unmatched_tracks = set(self.tracks.keys())
        pairs = []
        for tid, track in self.tracks.items():
            for di, det in enumerate(detections):
                score = iou(track["box"], det["box"])
                if score >= self.iou_threshold:
                    pairs.append((score, tid, di))
        pairs.sort(reverse=True, key=lambda p: p[0])

        used_tracks = set()
        used_dets = set()
        matches = []
        for score, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            matches.append((tid, di))
            used_tracks.add(tid)
            used_dets.add(di)

        unmatched_tracks -= used_tracks
        unmatched_dets = [d for d in range(len(detections)) if d not in used_dets]

        for tid, di in matches:
            self.tracks[tid]["box"] = detections[di]["box"]
            self.tracks[tid]["class_id"] = detections[di]["class_id"]
            self.tracks[tid]["missed"] = 0

        for tid in unmatched_tracks:
            self.tracks[tid]["missed"] += 1

        to_delete = [tid for tid, t in self.tracks.items() if t["missed"] > self.max_missed]
        for tid in to_delete:
            del self.tracks[tid]

        result = {}
        for tid, di in matches:
            result[di] = tid
        for di in unmatched_dets:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "box": detections[di]["box"],
                "class_id": detections[di]["class_id"],
                "missed": 0,
            }
            result[di] = tid

        return result


class TrackerOnlyNode(Node):
    def __init__(self):
        super().__init__('tracker_only_node')
        self.tracker = SimpleTracker(iou_threshold=0.3, max_missed=10)
        self.sub = self.create_subscription(
            Detection2DArray, 'detections_output', self.callback, 10)
        self.get_logger().info('Tracker-only node baslatildi (depth YOK, sadece RGB tracking)')

    def callback(self, msg):
        det_list = []
        for det in msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            box = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
            class_id = det.results[0].hypothesis.class_id if det.results else "unknown"
            det_list.append({"box": box, "class_id": class_id})

        id_map = self.tracker.update(det_list)

        for di, det in enumerate(det_list):
            track_id = id_map.get(di, -1)
            self.get_logger().info(f'Track ID={track_id} class={det["class_id"]}')


def main():
    rclpy.init()
    node = TrackerOnlyNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
