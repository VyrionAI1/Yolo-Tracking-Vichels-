import tkinter as tk
from tkinter import ttk
import json
import os
from pathlib import Path

# --- Configuration Persistence ---
def load_settings():
    defaults = {
        "TRACK_BUFFER": 8000,
        "YOLO_IOU": 0.1,
        "TRACKER_TYPE": "deepocsort.yaml",
        "STABILIZER_MAX_LOST": 30,
        "STABILIZER_IOU": 0.2,
        "STABILIZER_DIST": 600,
        "STABILIZER_HISTORY": 40,
        "STABILIZER_ANCHOR": 350,
        "USE_ROI": True
    }
    settings_file = "settings.json"
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                settings = json.load(f)
                for k, v in defaults.items():
                    if k not in settings: settings[k] = v
                return settings
        except: return defaults
    return defaults

def save_settings(settings):
    """Smart Save: Merges GUI settings with existing file data (preserves ROI)."""
    current = {}
    settings_file = "settings.json"
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                current = json.load(f)
        except: pass
    
    current.update(settings)
    
    with open(settings_file, "w") as f:
        json.dump(current, f, indent=4)

def save_tracker_yaml(buffer_size):
    for yaml_file in ["botsort.yaml", "bytetrack.yaml"]:
        if os.path.exists(yaml_file):
            with open(yaml_file, "r") as f: lines = f.readlines()
            with open(yaml_file, "w") as f:
                for line in lines:
                    if "track_buffer:" in line: f.write(f"track_buffer: {buffer_size}\n")
                    else: f.write(line)

class TuningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EagleVision Tuning")
        self.root.geometry("450x700")
        self.settings = load_settings()
        self.controls = {}
        self.create_widgets()

    def create_widgets(self):
        header = tk.Frame(self.root, bg='#2c3e50', height=60)
        header.pack(fill="x")
        tk.Label(header, text="EagleVision Live Tuning", font=('Helvetica', 14, 'bold'), fg='white', bg='#2c3e50').pack(pady=15)

        canvas = tk.Canvas(self.root, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Groups
        det_g = ttk.LabelFrame(scrollable_frame, text=" 1. YOLO Detection ", padding="10")
        det_g.pack(fill="x", pady=5, padx=10)
        self.add_slider(det_g, "YOLO_IOU", "NMS IOU", 0.01, 1.0, 0.05, False, "Detector sensitivity")

        mem_g = ttk.LabelFrame(scrollable_frame, text=" 2. Memory ", padding="10")
        mem_g.pack(fill="x", pady=5, padx=10)
        self.add_slider(mem_g, "STABILIZER_MAX_LOST", "Max Lost Frames", 1, 1000, 1, True, "ID Memory")
        self.add_slider(mem_g, "STABILIZER_HISTORY", "History Depth", 2, 100, 1, True, "Motion vector frames")

        sens_g = ttk.LabelFrame(scrollable_frame, text=" 3. Sensitivity ", padding="10")
        sens_g.pack(fill="x", pady=5, padx=10)
        self.add_slider(sens_g, "STABILIZER_IOU", "Box Match Thresh", 0.01, 0.5, 0.01, False, "Higher = stricter")
        self.add_slider(sens_g, "STABILIZER_DIST", "Max Jump Distance", 10, 2000, 10, True, "In pixels")
        self.add_slider(sens_g, "STABILIZER_ANCHOR", "Anchor Radius", 10, 1000, 10, True, "Entry/Exit zone")

        alg_g = ttk.LabelFrame(scrollable_frame, text=" 4. Algorithm ", padding="10")
        alg_g.pack(fill="x", pady=5, padx=10)
        self.tracker_var = tk.StringVar(value=self.settings.get("TRACKER_TYPE", "deepocsort.yaml"))
        for t in [("BoT-SORT", "botsort.yaml"), ("StrongSORT++", "strongsort.yaml"), ("Deep-OC-SORT", "deepocsort.yaml")]:
            ttk.Radiobutton(alg_g, text=t[0], variable=self.tracker_var, value=t[1], command=self.update_tracker_type).pack(anchor="w")

        spec_g = ttk.LabelFrame(scrollable_frame, text=" 5. Special Actions ", padding="10")
        spec_g.pack(fill="x", pady=5, padx=10)
        self.use_roi_var = tk.BooleanVar(value=self.settings.get("USE_ROI", True))
        ttk.Checkbutton(spec_g, text="Enable Area of Interest (ROI)", variable=self.use_roi_var, command=self.update_use_roi).pack(anchor="w")
        ttk.Button(spec_g, text="Reset ROI", command=self.reset_roi).pack(fill="x", pady=5)

        ttk.Button(scrollable_frame, text="Reset to Defaults", command=self.reset_defaults).pack(pady=20)

    def add_slider(self, parent, key, label, min_v, max_v, res, is_int, note):
        container = ttk.Frame(parent); container.pack(fill="x", pady=5)
        head = ttk.Frame(container); head.pack(fill="x")
        ttk.Label(head, text=label, font=('Helvetica', 10, 'bold')).pack(side="left")
        v_lab = ttk.Label(head, text=f"{self.settings.get(key)}"); v_lab.pack(side="right")
        v_var = tk.DoubleVar(value=self.settings.get(key))
        def on_change(v):
            val = int(float(v)) if is_int else round(float(v), 3)
            v_lab.config(text=str(val)); self.settings[key] = val; save_settings(self.settings)
        ttk.Scale(container, from_=min_v, to=max_v, variable=v_var, orient="horizontal", command=on_change).pack(fill="x")
        ttk.Label(container, text=note, font=('Helvetica', 8, 'italic'), foreground='gray').pack(anchor="w")
        self.controls[key] = (v_var, v_lab)

    def update_tracker_type(self):
        self.settings["TRACKER_TYPE"] = self.tracker_var.get(); save_settings(self.settings)

    def update_use_roi(self):
        self.settings["USE_ROI"] = self.use_roi_var.get(); save_settings(self.settings)

    def reset_roi(self):
        self.settings["ROI"] = None; save_settings(self.settings)

    def reset_defaults(self):
        # Default numeric/toggle settings (No ROI in here)
        d = {
            "TRACK_BUFFER": 8000, 
            "YOLO_IOU": 0.1, 
            "TRACKER_TYPE": "deepocsort.yaml", 
            "STABILIZER_MAX_LOST": 30, 
            "STABILIZER_IOU": 0.2, 
            "STABILIZER_DIST": 600, 
            "STABILIZER_HISTORY": 40, 
            "STABILIZER_ANCHOR": 350, 
            "USE_ROI": True
        }
        self.settings = d
        save_settings(d)
        save_tracker_yaml(8000)
        
        # Update UI components
        for k, v in d.items():
            if k in self.controls: 
                self.controls[k][0].set(v)
                self.controls[k][1].config(text=str(v))
        
        self.tracker_var.set("deepocsort.yaml")
        self.use_roi_var.set(True)
        print("[Settings] Reset to optimized defaults. ROI preserved.")

if __name__ == "__main__":
    root = tk.Tk(); TuningGUI(root); root.mainloop()