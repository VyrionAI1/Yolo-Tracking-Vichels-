import json
import os

# EagleVision Configuration Parameters

# --- Device & Hardware ---
DEVICE = 0                 # 0 = first GPU, 'cpu' = CPU

# --- YOLO Settings ---
MODEL_PATH = 'models/yolo26m.pt'
YOLO_IOU = 0.1             # NMS IoU threshold for detection
YOLO_IMGSZ = 640          # High resolution for high-speed/tiny cars

# --- Tracker Selection ---
TRACK_BUFFER = 8000
TRACKER_TYPE = 'deepocsort.yaml'
TRACKED_CLASSES = [2, 5, 7]  # COC0: 2=car, 5=bus, 7=truck

# --- ROI (Area of Interest) ---
USE_ROI = True
ROI = None  # Expected format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

# --- ID Stabilizer Settings ---
STABILIZER_MAX_LOST = 30   # Tightened for dense traffic
STABILIZER_IOU = 0.2       # Tightened for dense traffic
STABILIZER_DIST = 600      # Tightened to avoid ID merging
STABILIZER_HISTORY = 40
STABILIZER_ANCHOR = 350

# --- Live Tuning Logic ---
def load_live_settings():
    """Real-time parameter loading from settings.json."""
    global TRACK_BUFFER, YOLO_IOU, TRACKER_TYPE, TRACKED_CLASSES, ROI, USE_ROI
    global STABILIZER_MAX_LOST, STABILIZER_IOU, STABILIZER_DIST, STABILIZER_HISTORY, STABILIZER_ANCHOR
    settings_file = "settings.json"
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                data = json.load(f)
                TRACK_BUFFER = int(data.get("TRACK_BUFFER", TRACK_BUFFER))
                YOLO_IOU = float(data.get("YOLO_IOU", YOLO_IOU))
                TRACKER_TYPE = data.get("TRACKER_TYPE", TRACKER_TYPE)
                TRACKED_CLASSES = data.get("TRACKED_CLASSES", TRACKED_CLASSES)
                ROI = data.get("ROI", None)
                USE_ROI = data.get("USE_ROI", True)
                
                STABILIZER_MAX_LOST = int(data.get("STABILIZER_MAX_LOST", STABILIZER_MAX_LOST))
                STABILIZER_IOU = float(data.get("STABILIZER_IOU", STABILIZER_IOU))
                STABILIZER_DIST = float(data.get("STABILIZER_DIST", STABILIZER_DIST))
                STABILIZER_HISTORY = int(data.get("STABILIZER_HISTORY", STABILIZER_HISTORY))
                STABILIZER_ANCHOR = float(data.get("STABILIZER_ANCHOR", STABILIZER_ANCHOR))
        except Exception:
            pass  # Use defaults if there's an error

def save_roi(roi_coords):
    """Save selected ROI to settings.json."""
    settings_file = "settings.json"
    data = {}
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                data = json.load(f)
        except: pass
    
    data["ROI"] = roi_coords
    try:
        with open(settings_file, "w") as f:
            json.dump(data, f, indent=4)
    except: pass

# Initialize with live settings on import
load_live_settings()
