import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import numpy as np


class StreamDisplay:
    """Video stream display class, responsible for displaying video streams in the GUI"""

    def __init__(self, root, camera_stream, ai_processor):
        """
        Initializes the video stream display

        Args:
            root: Tkinter root window
            camera_stream: Camera stream object
            ai_processor: AI processor object
        """
        self.root = root
        self.camera_stream = camera_stream
        self.ai_processor = ai_processor
        self.running = False
        self.thread = None

        # Create display labels
        self.original_label = tk.Label(root)
        self.original_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.original_label.imgtk = None

        self.processed_label = tk.Label(root)
        self.processed_label.pack(side=tk.RIGHT, padx=10, pady=10)
        self.processed_label.imgtk = None

        # State variables
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()

        # Status display
        self.status_frame = tk.Frame(root)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(self.status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.fps_var = tk.StringVar(value="FPS: 0.0")
        self.fps_label = tk.Label(self.status_frame, textvariable=self.fps_var)
        self.fps_label.pack(side=tk.RIGHT, padx=10)

    def update_frames(self):
        """Loop for updating video frames"""
        while self.running:
            try:
                # Get the original frame
                frame = self.camera_stream.get_frame()

                if frame is not None:
                    # Display the original frame
                    self.display_frame(self.original_label, frame.copy())

                    # Process the frame
                    try:
                        processed_frame = self.ai_processor.process_frame(frame.copy())

                        # Display the processed frame
                        if processed_frame is not None:
                            self.display_frame(self.processed_label, processed_frame)
                    except Exception as e:
                        self.status_var.set(f"Processing Error: {e}")

                    # Update FPS count
                    self.frame_count += 1
                    current_time = time.time()
                    time_diff = current_time - self.last_time

                    # Update FPS display every second
                    if time_diff >= 1.0:
                        self.fps = self.frame_count / time_diff
                        self.fps_var.set(f"FPS: {self.fps:.1f}")
                        self.frame_count = 0
                        self.last_time = current_time

            except Exception as e:
                self.status_var.set(f"Display Error: {e}")

            # Reduce CPU usage
            time.sleep(0.01)

    def start_stream(self):
        """Starts the video stream display"""
        if self.running:
            return

        self.running = True
        self.camera_stream.running = True
        self.thread = threading.Thread(target=self.update_frames)
        self.thread.daemon = True
        self.thread.start()
        self.status_var.set("Video Stream Running")

    def stop_stream(self):
        """Stops the video stream display"""
        self.running = False

        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

        self.status_var.set("Video Stream Stopped")

    def display_frame(self, label, frame):
        """
        Displays a video frame on the label

        Args:
            label: The Tkinter label on which to display the frame
            frame: The video frame to display
        """
        if frame is None:
            return

        try:
            # Resize image
            h, w = frame.shape[:2]
            max_size = (640, 480)  # Maximum display size

            # Calculate scale factor, maintaining aspect ratio
            scale = min(max_size[0]/w, max_size[1]/h)
            new_size = (int(w*scale), int(h*scale))

            # Scale the image
            if scale != 1.0:
                frame = cv2.resize(frame, new_size)

            # Convert color format BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL image
            pil_image = Image.fromarray(rgb_frame)

            # Convert to Tkinter-compatible image
            imgtk = ImageTk.PhotoImage(image=pil_image)

            # Update the label
            label.imgtk = imgtk
            label.config(image=imgtk)

        except Exception as e:
            print(f"Error displaying frame: {e}")

    def update_status(self, message):
        """Updates the status display"""
        self.status_var.set(message)
