import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path


def _to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_payload(data):
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")

    if isinstance(data.get("touch_events"), list):
        return {
            "source_type": "touch_events",
            "video_name": data.get("video_name", "N/A"),
            "frames_processed": data.get("frames_processed", 0),
            "fps": data.get("fps", 0),
            "events": [ev for ev in data.get("touch_events", []) if isinstance(ev, dict)],
            "filter_pinky_default": True,
        }

    if isinstance(data.get("events"), list):
        meta = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}
        normalized_events = []
        for ev in data.get("events", []):
            if not isinstance(ev, dict):
                continue
            finger_type = str(ev.get("finger_type") or "").strip().lower()
            fingertip = str(ev.get("fingertip") or (f"{finger_type}_tip" if finger_type else "unknown"))
            timestamp_sec = ev.get("timestamp_sec", ev.get("time_sec", 0.0))
            struck_id = ev.get("struck_string_id", ev.get("string_id"))
            normalized_events.append(
                {
                    "time_sec": timestamp_sec,
                    "timestamp_sec": timestamp_sec,
                    "frame_index": ev.get("frame_index"),
                    "hand": str(ev.get("hand") or ev.get("hand_side") or "right"),
                    "hand_side": str(ev.get("hand_side") or ev.get("hand") or "right"),
                    "fingertip": fingertip,
                    "finger_type": finger_type,
                    "string_id": struck_id,
                    "beat_label": ev.get("beat_label"),
                    "confidence": ev.get("confidence"),
                    "confidence_label": ev.get("confidence_label"),
                    "strategy": ev.get("strategy"),
                }
            )
        source_video = meta.get("source_video") or meta.get("video_name") or "right_av_strike_events"
        return {
            "source_type": "av_strike_events",
            "video_name": str(Path(str(source_video)).name),
            "frames_processed": meta.get("frames_processed", 0),
            "fps": meta.get("fps", 0),
            "events": normalized_events,
            "filter_pinky_default": False,
        }

    raise ValueError("Unsupported JSON schema. Expected `touch_events` or `events`.")

class TouchEventsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Touch Events Analysis Dashboard")
        self.root.geometry("1100x850")
        
        # Beat Analysis Variables
        self.bpm = tk.DoubleVar(value=120.0)
        self.tolerance = tk.DoubleVar(value=0.1)
        self.raw_data = None
        self.normalized_payload = None
        
        # Configure overall style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colors & Fonts
        self.bg_color = "#121212"
        self.fg_color = "#e0e0e0"
        self.accent_color = "#3b82f6"
        self.card_bg = "#1e1e1e"
        self.font_main = ("Segoe UI", 10)
        self.font_title = ("Segoe UI", 16, "bold")
        self.font_subtitle = ("Segoe UI", 12, "bold")
        
        self.root.configure(bg=self.bg_color)
        
        style.configure("TFrame", background=self.bg_color)
        style.configure("Card.TFrame", background=self.card_bg, relief="flat")
        style.configure("TLabel", background=self.bg_color, foreground=self.fg_color, font=self.font_main)
        style.configure("Card.TLabel", background=self.card_bg, foreground=self.fg_color, font=self.font_main)
        style.configure("Title.TLabel", foreground=self.accent_color, font=self.font_title, background=self.bg_color)
        style.configure("SubTitle.TLabel", foreground=self.fg_color, font=self.font_subtitle, background=self.card_bg)
        style.configure("Accent.TButton", background=self.accent_color, foreground="white", font=("Segoe UI", 10, "bold"), padding=10)
        style.map("Accent.TButton", background=[("active", "#2563eb")])
        
        # Customizing Treeview Colors
        style.configure("Treeview", 
                        background=self.bg_color,
                        foreground=self.fg_color,
                        fieldbackground=self.bg_color,
                        bordercolor=self.card_bg,
                        rowheight=25)
        style.map('Treeview', background=[('selected', self.accent_color)])
        style.configure("Treeview.Heading",
                        background=self.card_bg,
                        foreground=self.fg_color,
                        font=("Segoe UI", 10, "bold"))
        style.map("Treeview.Heading", background=[('active', '#2d2d2d')])

        self.setup_ui()
        
        # Trace changes to BPM/Tolerance
        self.bpm.trace_add("write", lambda *args: self.reprocess())
        self.tolerance.trace_add("write", lambda *args: self.reprocess())
        
        # Data
        self.all_events = []

    def setup_ui(self):
        # Header Frame
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=20, pady=20)
        
        title = ttk.Label(header_frame, text="Events Analysis Dashboard", style="Title.TLabel")
        title.pack(side=tk.LEFT)
        
        # Controls Group
        ctrl_frame = ttk.Frame(header_frame)
        ctrl_frame.pack(side=tk.RIGHT)
        
        ttk.Label(ctrl_frame, text="BPM:", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(10, 2))
        bpm_spin = tk.Spinbox(ctrl_frame, from_=20, to=300, textvariable=self.bpm, width=5, 
                             bg=self.card_bg, fg=self.fg_color, insertbackground=self.fg_color, bd=0)
        bpm_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(ctrl_frame, text="Tolerance (s):", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(10, 2))
        tol_spin = tk.Spinbox(ctrl_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.tolerance, 
                             width=5, bg=self.card_bg, fg=self.fg_color, insertbackground=self.fg_color, bd=0)
        tol_spin.pack(side=tk.LEFT, padx=5)
        
        btn_load = ttk.Button(ctrl_frame, text="Load JSON", style="Accent.TButton", command=self.load_file)
        btn_load.pack(side=tk.LEFT, padx=15)
        
        # Main Content container
        self.content_frame = ttk.Frame(self.root)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Top Grid for Stats
        self.stats_frame = ttk.Frame(self.content_frame)
        self.stats_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Configure columns for stats grid (now 5 columns)
        for i in range(5):
            self.stats_frame.columnconfigure(i, weight=1)
            
        self.lbl_video = self.create_stat_card(self.stats_frame, 0, "Video Name", "-")
        self.lbl_frames = self.create_stat_card(self.stats_frame, 1, "Frames", "-")
        self.lbl_fps = self.create_stat_card(self.stats_frame, 2, "FPS", "-")
        self.lbl_total = self.create_stat_card(self.stats_frame, 3, "Total Events", "-")
        self.lbl_accuracy = self.create_stat_card(self.stats_frame, 4, "Beat Accuracy", "-")
        
        # Middle Grid for Classifications
        self.class_frame = ttk.Frame(self.content_frame)
        self.class_frame.pack(fill=tk.X, pady=(0, 20))
        
        for i in range(3):
            self.class_frame.columnconfigure(i, weight=1)
            
        self.hand_text = self.create_list_card(self.class_frame, 0, "By Hand")
        self.finger_text = self.create_list_card(self.class_frame, 1, "By Fingertip")
        self.string_text = self.create_list_card(self.class_frame, 2, "Top String IDs")

        # Bottom Area for Table
        self.table_frame = ttk.Frame(self.content_frame, style="Card.TFrame")
        self.table_frame.pack(fill=tk.BOTH, expand=True)
        
        table_lbl_frame = ttk.Frame(self.table_frame, style="Card.TFrame")
        table_lbl_frame.pack(fill=tk.X, padx=15, pady=10)
        
        table_lbl = ttk.Label(table_lbl_frame, text="Detailed Event Log", style="SubTitle.TLabel")
        table_lbl.pack(side=tk.LEFT)
        
        self.lbl_page = ttk.Label(table_lbl_frame, text="", style="Card.TLabel", font=("Segoe UI", 9))
        self.lbl_page.pack(side=tk.RIGHT)
        
        # Treeview (Table)
        columns = ("time", "frame", "hand", "fingertip", "string", "beat")
        self.tree = ttk.Treeview(self.table_frame, columns=columns, show="headings")
        self.tree.heading("time", text="Time (s)")
        self.tree.heading("frame", text="Frame")
        self.tree.heading("hand", text="Hand")
        self.tree.heading("fingertip", text="Fingertip")
        self.tree.heading("string", text="String ID")
        self.tree.heading("beat", text="Beat Status")
        
        self.tree.column("time", width=80, anchor=tk.CENTER)
        self.tree.column("frame", width=80, anchor=tk.CENTER)
        self.tree.column("hand", width=80, anchor=tk.CENTER)
        self.tree.column("fingertip", width=120, anchor=tk.W)
        self.tree.column("string", width=80, anchor=tk.CENTER)
        self.tree.column("beat", width=100, anchor=tk.CENTER)
        
        # Tags for coloring
        self.tree.tag_configure('onbeat', foreground="#4ade80")
        self.tree.tag_configure('offbeat', foreground="#f87171")
        
        # Scrollbar for table
        vsb = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(15, 0), pady=(0, 15))
        vsb.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 15), pady=(0, 15))

    def create_stat_card(self, parent, col, title, value):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=0, column=col, sticky="nsew", padx=5)
        
        lbl_title = ttk.Label(frame, text=title.upper(), style="Card.TLabel", foreground="#888888", font=("Segoe UI", 9, "bold"))
        lbl_title.pack(anchor=tk.W, padx=15, pady=(15, 5))
        
        lbl_value = ttk.Label(frame, text=value, style="Card.TLabel", font=("Segoe UI", 16, "bold"))
        lbl_value.pack(anchor=tk.W, padx=15, pady=(0, 15))
        
        return lbl_value
        
    def create_list_card(self, parent, col, title):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=0, column=col, sticky="nsew", padx=5)
        
        lbl_title = ttk.Label(frame, text=title, style="SubTitle.TLabel")
        lbl_title.pack(anchor=tk.W, padx=15, pady=(15, 5))
        
        text_widget = tk.Text(frame, bg=self.card_bg, fg=self.fg_color, bd=0, 
                             highlightthickness=0, font=("Courier New", 10), height=8)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        text_widget.config(state=tk.DISABLED)
        
        return text_widget

    def update_text_widget(self, widget, content):
        widget.config(state=tk.NORMAL)
        widget.delete(1.0, tk.END)
        widget.insert(tk.END, content)
        widget.config(state=tk.DISABLED)

    def load_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Touch/AV Events JSON file",
            filetypes=[("JSON Files", "*.json")]
        )
        
        if not filepath:
            return
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            self.normalized_payload = _normalize_payload(self.raw_data)
            self.process_data(self.raw_data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file\n{str(e)}")

    def reprocess(self):
        if self.raw_data:
            self.process_data(self.raw_data)

    def process_data(self, data):
        # Clear existing table data
        for item in self.tree.get_children():
            self.tree.delete(item)

        if self.normalized_payload is None:
            self.normalized_payload = _normalize_payload(data)
        payload = self.normalized_payload

        raw_events = payload.get('events', [])
        filter_pinky = bool(payload.get("filter_pinky_default", True))
        self.all_events = [ev for ev in raw_events if not (filter_pinky and ev.get('fingertip') == 'pinky_tip')]
        
        total_events = len(self.all_events)
        
        # Update Top Stats
        video_name = payload.get('video_name', 'N/A')
        self.lbl_video.config(text=video_name[:15] + '...' if len(video_name) > 15 else video_name)
        self.lbl_frames.config(text=f"{int(payload.get('frames_processed', 0) or 0):,}")
        self.lbl_fps.config(text=f"{payload.get('fps', 0)}")
        self.lbl_total.config(text=f"{total_events:,}")
        
        if total_events == 0:
            self.lbl_accuracy.config(text="-")
            return
            
        # Tally Classifications & Beat Analysis
        hands = {}
        fingers = {}
        strings = {}
        on_beat_count = 0
        beat_source_stats = {"json": 0, "computed": 0}
        
        # Get Current BPM and Tolerance
        try:
            bpm = self.bpm.get()
            tol = self.tolerance.get()
        except:
            bpm = 120.0
            tol = 0.1
            
        beat_interval = 60.0 / bpm if bpm > 0 else 1.0

        for ev in self.all_events:
            h = ev.get('hand', 'unknown')
            hands[h] = hands.get(h, 0) + 1
            
            f = ev.get('fingertip', 'unknown')
            fingers[f] = fingers.get(f, 0) + 1
            
            s = ev.get('string_id', 'unknown')
            strings[s] = strings.get(s, 0) + 1
            
            # Beat Logic
            beat_label = str(ev.get('beat_label') or '').strip().lower()
            if beat_label in {"on_beat", "off_beat"}:
                is_on = beat_label == "on_beat"
                ev['beat_source'] = 'json'
            else:
                time_sec = _to_float(ev.get('time_sec', 0.0), 0.0)
                remainder = time_sec % beat_interval
                is_on = (remainder < tol) or (beat_interval - remainder < tol)
                ev['beat_source'] = 'computed'
            beat_source_stats[ev['beat_source']] = beat_source_stats.get(ev['beat_source'], 0) + 1
            ev['is_on_beat'] = is_on
            if is_on:
                on_beat_count += 1
            
        # Accuracy Header
        acc_pct = (on_beat_count / total_events) * 100
        self.lbl_accuracy.config(text=f"{acc_pct:.1f}%")
        
        # Format Text for Lists
        def format_stats(tally_dict, max_items=None):
            sorted_items = sorted(tally_dict.items(), key=lambda x: x[1], reverse=True)
            if max_items:
                sorted_items = sorted_items[:max_items]
                
            lines = []
            for k, v in sorted_items:
                pct = (v / total_events) * 100
                label = str(k).replace('_', ' ').capitalize()
                if "Str" not in str(k) and isinstance(k, int):
                    label = f"String {k}"
                lines.append(f"{label:<12} {v:>5} ({pct:0.1f}%)")
            return "\n".join(lines)
            
        self.update_text_widget(self.hand_text, format_stats(hands))
        self.update_text_widget(self.finger_text, format_stats(fingers))
        self.update_text_widget(self.string_text, format_stats(strings, 8))
        
        # Populate Table (Limit for performance)
        display_events = self.all_events[:1000] 
        source_type = str(payload.get("source_type", "events"))
        self.lbl_page.config(text=f"{source_type} | Showing first 1000 events of {total_events:,} | beat: json={beat_source_stats['json']} computed={beat_source_stats['computed']}")
        
        for ev in display_events:
            time_str = f"{_to_float(ev.get('time_sec', 0.0), 0.0):.3f}"
            frame = ev.get('frame_index', '')
            hand = str(ev.get('hand', '')).capitalize()
            fingertip = str(ev.get('fingertip', '')).replace('_', ' ').capitalize()
            string_id = f"Str {ev.get('string_id', '')}"
            beat_tag = "ON" if ev.get('is_on_beat') else "OFF"
            tag = 'onbeat' if ev.get('is_on_beat') else 'offbeat'
            
            self.tree.insert("", "end", values=(time_str, frame, hand, fingertip, string_id, beat_tag), tags=(tag,))

if __name__ == "__main__":
    root = tk.Tk()
    
    # Try Immersive Dark Mode
    try:
        import ctypes
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        set_window_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
        get_parent = ctypes.windll.user32.GetParent
        hwnd = get_parent(root.winfo_id())
        rendering_policy = DWMWA_USE_IMMERSIVE_DARK_MODE
        value = 2
        value = ctypes.c_int(value)
        set_window_attribute(hwnd, rendering_policy, ctypes.byref(value), ctypes.sizeof(value))
    except:
        pass

    app = TouchEventsApp(root)
    
    # Auto-load existing path
    default_path = r"c:\Users\ASUS\Downloads\Telegram Desktop\91287cd719ea492e943e77c88504383d_touch_events.json"
    if os.path.exists(default_path):
        try:
            with open(default_path, 'r', encoding='utf-8') as f:
                app.raw_data = json.load(f)
            app.process_data(app.raw_data)
        except:
            pass
            
    root.mainloop()
