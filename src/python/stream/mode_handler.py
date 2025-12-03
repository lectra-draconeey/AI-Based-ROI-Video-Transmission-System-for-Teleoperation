import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import cv2
import time
import socket
import struct
import pickle
import numpy as np
from PIL import Image, ImageTk
from src.python.stream.display import StreamDisplay


class BaseHandler:
    """Base class for all processors"""
    def __init__(self, root, camera_stream, ai_processor):
        self.root = root
        self.camera_stream = camera_stream
        self.ai_processor = ai_processor
        self.running = False
        self.thread = None

        # Create UI elements
        self.setup_ui()

    def setup_ui(self):
        """Sets up the basic UI interface"""
        # Main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Video display area
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        # Original video label frame
        self.original_frame = ttk.LabelFrame(self.video_frame, text="Original Video")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.original_label = ttk.Label(self.original_frame)
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Processed video label frame
        self.processed_frame = ttk.LabelFrame(self.video_frame, text="Processed Video")
        self.processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.processed_label = ttk.Label(self.processed_frame)
        self.processed_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Control area
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=10)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5)

        # Add progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(
            self.control_frame,
            variable=self.progress_var,
            orient=tk.HORIZONTAL,
            length=200,
            mode='determinate'
        )
        self.progress.pack(side=tk.LEFT, padx=10)

    def update_status(self, message):
        """Updates the status information"""
        self.status_var.set(message)

    def update_progress(self, value):
        """Updates the progress bar value"""
        self.progress_var.set(value)

    def display_frame(self, label, frame):
        """Displays a video frame on the label"""
        if frame is None:
            return

        # Resize image to fit the label
        h, w = frame.shape[:2]
        max_size = (400, 300)  # Maximum display size

        # Calculate scale factor
        scale = min(max_size[0]/w, max_size[1]/h)
        new_size = (int(w*scale), int(h*scale))

        # Scale the image
        resized = cv2.resize(frame, new_size)

        # Convert color format
        cv2_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_image)
        tk_image = ImageTk.PhotoImage(image=pil_image)

        # Update label image
        label.imgtk = tk_image
        label.config(image=tk_image)

    def start(self):
        """Starts the processing thread"""
        self.running = True
        self.thread = threading.Thread(target=self.process_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stops the processing thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def process_loop(self):
        """Processing loop, must be implemented by subclass"""
        raise NotImplementedError("Subclass must implement the process_loop method")


class LocalHandler(BaseHandler):
    """Local processing mode handler"""
    def __init__(self, root, camera_stream, ai_processor):
        super().__init__(root, camera_stream, ai_processor)

        # Add UI elements specific to local mode
        self.add_local_controls()

    def add_local_controls(self):
        """Adds local mode control buttons"""
        # Add QP value adjustment sliders
        self.qp_frame = ttk.LabelFrame(self.control_frame, text="Encoding Quality Control")
        self.qp_frame.pack(side=tk.RIGHT, padx=10)

        # ROI Region QP Value
        ttk.Label(self.qp_frame, text="ROI Region QP Value:").grid(row=0, column=0, padx=5, pady=3)
        self.roi_qp_var = tk.IntVar(value=15)  # Low QP = High Quality
        self.roi_qp_scale = ttk.Scale(
            self.qp_frame,
            from_=5,
            to=40,
            orient=tk.HORIZONTAL,
            variable=self.roi_qp_var,
            length=100
        )
        self.roi_qp_scale.grid(row=0, column=1, padx=5, pady=3)
        ttk.Label(self.qp_frame, textvariable=self.roi_qp_var).grid(row=0, column=2, padx=5
