import cv2
import numpy as np
from ultralytics import YOLO
import json
import time
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from collections import deque
import gc
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
import config  # Centralized Configuration module
import logging

# Suppress the redundant 'source' warning from Ultralytics
class SourceWarningFilter(logging.Filter):
    def filter(self, record):
        return "'source' is missing" not in record.getMessage()
logging.getLogger("ultralytics").addFilter(SourceWarningFilter())

# ─────────────────────────────────────────────────────────────────────────────
# IDStabilizer — Post-processing Anti-ID-Switch Layer
# ─────────────────────────────────────────────────────────────────────────────
class IDStabilizer:
    def __init__(
        self,
        max_lost_frames: int = 300,
        iou_thresh: float = 0.05,
        dist_thresh: float = 500,
        history_len: int = 15,
        anchor_radius: float = 350,
    ):
        self.max_lost_frames = max_lost_frames
        self.iou_thresh       = iou_thresh
        self.dist_thresh      = dist_thresh
        self.history_len      = history_len
        self.anchor_radius    = anchor_radius

        self.track_history: dict[int, deque] = {}
        self.lost_tracks: dict[int, dict]    = {}
        self.id_map: dict[int, int]          = {}
        self._sid_cls: dict[int, int]        = {}
        self._anchor: dict[int, list]        = {}

        self.next_stable_id: int = 1

    @staticmethod
    def _iou(a, b) -> float:
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter  = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0: return 0.0
        areaA  = (a[2]-a[0]) * (a[3]-a[1])
        areaB  = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (areaA + areaB - inter + 1e-6)

    @staticmethod
    def _center(box):
        return (box[0]+box[2])/2, (box[1]+box[3])/2

    @staticmethod
    def _center_dist(a, b) -> float:
        cxA, cyA = (a[0]+a[2])/2, (a[1]+a[3])/2
        cxB, cyB = (b[0]+b[2])/2, (b[1]+b[3])/2
        return ((cxA-cxB)**2 + (cyA-cyB)**2) ** 0.5

    def _update_anchor(self, sid: int, box):
        cx, cy = self._center(box)
        if sid not in self._anchor:
            self._anchor[sid] = [cx, cy, 1]
        else:
            acx, acy, n = self._anchor[sid]
            n = min(n + 1, 60)
            self._anchor[sid] = [acx + (cx - acx) / n, acy + (cy - acy) / n, n]

    def _anchor_dist(self, sid: int, box) -> float:
        if sid not in self._anchor: return float('inf')
        acx, acy, _ = self._anchor[sid]
        cx, cy = self._center(box)
        return ((cx - acx)**2 + (cy - acy)**2) ** 0.5

    def _avg_velocity(self, history: deque):
        pts = list(history)
        if len(pts) < 2: return (0.0, 0.0, 0.0, 0.0)
        n   = len(pts) - 1
        vx  = ((pts[-1][0]+pts[-1][2])/2 - (pts[0][0]+pts[0][2])/2) / n
        vy  = ((pts[-1][1]+pts[-1][3])/2 - (pts[0][1]+pts[0][3])/2) / n
        dwx = ((pts[-1][2]-pts[-1][0]) - (pts[0][2]-pts[0][0])) / (2*n)
        dwy = ((pts[-1][3]-pts[-1][1]) - (pts[0][3]-pts[0][1])) / (2*n)
        return (vx, vy, dwx, dwy)

    def _predict_box(self, last_box, vel, frames_ahead: int):
        vx, vy = vel[0], vel[1]
        dwx, dwy = (vel[2], vel[3]) if len(vel) > 3 else (0.0, 0.0)
        fx, fy = vx * frames_ahead, vy * frames_ahead
        return (last_box[0]+fx-dwx*frames_ahead, last_box[1]+fy-dwy*frames_ahead,
                last_box[2]+fx+dwx*frames_ahead, last_box[3]+fy+dwy*frames_ahead)

    def update(self, raw_ids, boxes, cls_indices) -> list[int]:
        stable_ids = []
        current_raw_set = set(raw_ids)

        for sid in list(self.lost_tracks):
            self.lost_tracks[sid]["frames_lost"] += 1
            if self.lost_tracks[sid]["frames_lost"] > self.max_lost_frames:
                print(f"[IDStabilizer] Expiring Stable #{sid} (Out of frame for {self.max_lost_frames} frames)")
                del self.lost_tracks[sid]
                for r, s in list(self.id_map.items()):
                    if s == sid: del self.id_map[r]

        lost_by_class: dict[int, list] = {}
        for lost_sid, info in self.lost_tracks.items():
            lost_by_class.setdefault(info["cls_idx"], []).append(lost_sid)

        for raw_id, box, cls_idx in zip(raw_ids, boxes, cls_indices):
            box = tuple(float(v) for v in box)
            if raw_id in self.id_map:
                sid = self.id_map[raw_id]
                # print(f"[IDStabilizer] Active: Raw #{raw_id} == Stable #{sid}")
            else:
                best_sid, best_score, match_reason = None, -1.0, "new"
                same_class_lost = lost_by_class.get(cls_idx, [])

                # Layer 3: zone-anchor
                for lost_sid in same_class_lost:
                    adist = self._anchor_dist(lost_sid, box)
                    if adist < self.anchor_radius:
                        score = 1.0 - adist / self.anchor_radius
                        if score > best_score:
                            best_score, best_sid = score, lost_sid
                            match_reason = f"zone-anchor(d={adist:.0f}px)"

                # Layer 4: trajectory prediction
                for lost_sid in same_class_lost:
                    info = self.lost_tracks[lost_sid]
                    pred = self._predict_box(info["last_box"], info["vel"], info["frames_lost"])
                    iou  = self._iou(box, pred)
                    dist = self._center_dist(box, pred)
                    if dist > self.dist_thresh and iou < self.iou_thresh: continue
                    score = iou * 0.6 + max(0.0, 1.0 - dist / self.dist_thresh) * 0.4
                    if score > best_score:
                        best_score, best_sid = score, lost_sid
                        match_reason = f"trajectory(iou={iou:.2f},d={dist:.0f}px)"

                if best_sid is not None:
                    sid = best_sid
                    del self.lost_tracks[best_sid]
                    if cls_idx in lost_by_class and best_sid in lost_by_class[cls_idx]:
                        lost_by_class[cls_idx].remove(best_sid)
                    print(f"[IDStabilizer] Match Found: Raw #{raw_id} → Stable #{sid} via {match_reason}")
                else:
                    sid = self.next_stable_id
                    self.next_stable_id += 1
                    print(f"[IDStabilizer] New Object: Raw #{raw_id} → Assigned Stable #{sid}")
                self.id_map[raw_id] = sid

            if sid not in self.track_history:
                self.track_history[sid] = deque(maxlen=self.history_len)
            self.track_history[sid].append(box)
            self._update_anchor(sid, box)
            stable_ids.append(sid)

        for raw_id, sid in list(self.id_map.items()):
            if raw_id not in current_raw_set and sid not in self.lost_tracks:
                hist = self.track_history.get(sid, deque())
                vel, last = self._avg_velocity(hist), (hist[-1] if hist else (0.0,0.0,0.0,0.0))
                self.lost_tracks[sid] = {"last_box": last, "vel": vel, "frames_lost": 0, "cls_idx": self._sid_cls.get(sid, -1)}
        return stable_ids

    def register_cls(self, sid: int, cls_idx: int):
        self._sid_cls[sid] = cls_idx

try:
    from boxmot import StrongSORT, DeepOCSORT
    HAS_BOXMOT = True
except ImportError as e:
    HAS_BOXMOT, BOXMOT_ERR = False, str(e)

class PolygonSelector:
    def __init__(self, win_name, frame):
        self.win_name, self.frame, self.points = win_name, frame.copy(), []
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, self.on_mouse)
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4: self.points.append((x, y))
    def select(self):
        while True:
            img_draw = self.frame.copy()
            for i, pt in enumerate(self.points):
                cv2.circle(img_draw, pt, 6, (0, 255, 255), -1)
                if i > 0: cv2.line(img_draw, self.points[i-1], self.points[i], (0, 255, 255), 2)
            if len(self.points) == 4: cv2.line(img_draw, self.points[3], self.points[0], (0, 255, 255), 3)
            cv2.imshow(self.win_name, img_draw)
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(self.points) == 4: break
            if key == ord('c'): self.points = []
            if key == 27: self.points = []; break
        cv2.destroyWindow(self.win_name)
        return self.points

class TrackingAnalyzer:
    def __init__(self, model_path=config.MODEL_PATH):
        self.model_path, self.model = model_path, YOLO(model_path)
        self.stabilizer = IDStabilizer(
            max_lost_frames=config.STABILIZER_MAX_LOST,
            iou_thresh=config.STABILIZER_IOU,
            dist_thresh=config.STABILIZER_DIST,
            history_len=config.STABILIZER_HISTORY,
            anchor_radius=config.STABILIZER_ANCHOR,
        )
        self.boxmot_tracker, self.current_tracker_file = None, config.TRACKER_TYPE

    def _get_boxmot_tracker(self):
        if not HAS_BOXMOT: return None
        if self.boxmot_tracker is None or self.current_tracker_file != config.TRACKER_TYPE:
            try:
                if config.TRACKER_TYPE == 'strongsort.yaml':
                    self.boxmot_tracker = StrongSORT(model_weights=Path('osnet_x0_25_msmt17.pt'), device=config.DEVICE, fp16=True)
                elif config.TRACKER_TYPE == 'deepocsort.yaml':
                    self.boxmot_tracker = DeepOCSORT(model_weights=Path('osnet_x0_25_msmt17.pt'), device=config.DEVICE, fp16=True)
                self.current_tracker_file = config.TRACKER_TYPE
            except Exception: return None
        return self.boxmot_tracker

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return
        w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
        prev_time = time.time()
        out_path = f"out/tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        Path("out").mkdir(exist_ok=True)
        video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        if config.USE_ROI:
            if config.ROI and not isinstance(config.ROI[0], (list, tuple)): config.save_roi(None); config.ROI = None
            if config.ROI is None:
                ret, first_frame = cap.read()
                if ret:
                    pts = PolygonSelector("Select 4-Point ROI", first_frame).select()
                    if len(pts) == 4: config.ROI = pts; config.save_roi(pts)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                stable_ids = [] # Reset for every frame
                
                if (frame_id := int(cap.get(cv2.CAP_PROP_POS_FRAMES))) % 30 == 0:
                    config.load_live_settings()
                    self.stabilizer.max_lost_frames = config.STABILIZER_MAX_LOST
                    self.stabilizer.iou_thresh = config.STABILIZER_IOU
                    self.stabilizer.dist_thresh = config.STABILIZER_DIST
                    self.stabilizer.history_len = config.STABILIZER_HISTORY
                    self.stabilizer.anchor_radius = config.STABILIZER_ANCHOR

                proc_frame = frame
                if config.USE_ROI and config.ROI and len(config.ROI) == 4:
                    mask = np.zeros(frame.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(config.ROI, dtype=np.int32).reshape((-1,1,2))], (255,255,255))
                    proc_frame = cv2.bitwise_and(frame, mask)

                if config.TRACKER_TYPE != self.current_tracker_file:
                    if hasattr(self, 'model'): del self.model
                    self.boxmot_tracker = None
                    gc.collect()
                    if HAS_TORCH and torch.cuda.is_available(): torch.cuda.empty_cache()
                    self.model = YOLO(self.model_path); self.current_tracker_file = config.TRACKER_TYPE

                if config.TRACKER_TYPE in ['strongsort.yaml', 'deepocsort.yaml'] and HAS_BOXMOT:
                    res = self.model.predict(source=proc_frame, conf=0.1, imgsz=config.YOLO_IMGSZ, classes=config.TRACKED_CLASSES, verbose=False, device=config.DEVICE)
                    tracks = self._get_boxmot_tracker().update(res[0].boxes.data.cpu().numpy(), proc_frame)
                    raw_boxes, raw_ids, cls_indices = (tracks[:, 0:4], tracks[:, 4].astype(int), tracks[:, 6].astype(int)) if tracks.size > 0 else ([],[],[])
                else:
                    res = self.model.track(source=proc_frame, persist=True, tracker=config.TRACKER_TYPE if config.TRACKER_TYPE in ['botsort.yaml', 'bytetrack.yaml'] else 'botsort.yaml', conf=0.1, iou=config.YOLO_IOU, imgsz=config.YOLO_IMGSZ, classes=config.TRACKED_CLASSES, agnostic_nms=True, verbose=False, device=config.DEVICE)
                    raw_boxes, raw_ids, cls_indices = (res[0].boxes.xyxy.cpu().numpy(), res[0].boxes.id.cpu().numpy().astype(int), res[0].boxes.cls.cpu().numpy().astype(int)) if res[0].boxes.id is not None else ([],[],[])

                # ── Post-processing: ID Stabilization ────────────────────────
                # Always update (even if empty) to age 'lost' tracks correctly
                stable_ids = self.stabilizer.update(
                    raw_ids.tolist() if isinstance(raw_ids, np.ndarray) else raw_ids,
                    raw_boxes.tolist() if isinstance(raw_boxes, np.ndarray) else raw_boxes,
                    cls_indices.tolist() if isinstance(cls_indices, np.ndarray) else cls_indices
                )
                
                if len(raw_ids) > 0:
                    for box, obj_id, cls_idx in zip(raw_boxes, stable_ids, cls_indices):
                        x1, y1, x2, y2 = map(int, box)
                        color = (255, 0, 0) if cls_idx == 1 else (0, 180, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{self.model.names[cls_idx]} #{obj_id}", (x1, max(y1-8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                if config.USE_ROI and config.ROI and len(config.ROI) == 4:
                    cv2.polylines(frame, [np.array(config.ROI, dtype=np.int32).reshape((-1,1,2))], True, (0, 255, 255), 2)

                curr_t = time.time(); elapsed = curr_t - prev_time; prev_time = curr_t
                cv2.putText(frame, f"Tracker: {config.TRACKER_TYPE.upper()} | FPS: {1.0/elapsed if elapsed>0 else 0:.1f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                # Row 1: Active (Visible)
                cv2.putText(frame, f"Active IDs: {sorted(stable_ids)}", (15, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Row 2: Memory (Waiting/Hidden)
                hidden_ids = sorted(list(self.stabilizer.lost_tracks.keys()))
                cv2.putText(frame, f"Memory (Hidden): {hidden_ids}", (15, 75), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
                cv2.imshow("EagleVision", frame); video_writer.write(frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('r'): config.save_roi(None); break
        finally:
            cap.release(); video_writer.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    gui_p = subprocess.Popen([sys.executable, "setting.py"])
    try:
        analyzer = TrackingAnalyzer()
        if Path("videos/v.mp4").exists(): analyzer.analyze_video("videos/v.mp4")
    finally:
        time.sleep(0.5); gui_p.terminate()