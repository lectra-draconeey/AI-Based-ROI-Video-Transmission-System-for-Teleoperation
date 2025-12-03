"""
Python Interface for C++ Components

This module provides an interface for Python to call the RTMP push streaming and RTSP client functionalities implemented in C++.
The ctypes library is used to load the dynamic library and call its functions.

Author: PycharmProjects
Date: 2023
"""

import os
import sys
import ctypes
import numpy as np
from typing import Callable, Optional, List, Tuple, Dict
import threading
import platform


# Determine the dynamic library file extension
def get_lib_ext():
    """Gets the dynamic library extension based on the operating system"""
    if platform.system() == 'Windows':
        return '.dll'
    elif platform.system() == 'Darwin':  # macOS
        return '.dylib'
    else:  # Linux and other UNIX systems
        return '.so'


# Find the path to the library file
def find_library(name: str) -> str:
    """
    Finds the path to the dynamic library file

    Args:
        name: Library file name (without extension)

    Returns:
        The full path to the library file

    Raises:
        FileNotFoundError: If the library file is not found
    """
    # Possible paths
    lib_ext = get_lib_ext()
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'lib', f'lib{name}{lib_ext}'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'build', f'lib{name}{lib_ext}'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'lib', f'{name}{lib_ext}'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'build', f'{name}{lib_ext}'),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)

    # If the library is not found, print detailed error information
    error_msg = f"Library file not found: {name}{lib_ext}\n"
    error_msg += "Search paths:\n"
    for path in possible_paths:
        error_msg += f"  - {os.path.abspath(path)}\n"
    error_msg += "Please ensure the C++ library has been compiled and the library file is in the correct location."

    raise FileNotFoundError(error_msg)


# Define the ROI region class
class ROIRegion:
    """Represents an ROI (Region of Interest) in the video"""

    def __init__(self, x: int, y: int, width: int, height: int, qp: int = 15):
        """
        Initializes the ROI region

        Args:
            x: Top-left x-coordinate
            y: Top-left y-coordinate
            width: Width
            height: Height
            qp: QP value (Quantization Parameter, lower value means higher quality)
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.qp = qp

    def __repr__(self) -> str:
        return f"ROIRegion(x={self.x}, y={self.y}, width={self.width}, height={self.height}, qp={self.qp})"


# RTMP Streamer class
class RTMPStreamer:
    """RTMP Streamer, used to push video streams to a streaming media server"""

    def __init__(self, url: str, width: int, height: int, fps: int = 30,
                 bitrate: int = 1000000, gop: int = 30, qp: int = 23):
        """
        Initializes the RTMP streamer

        Args:
            url: RTMP server URL
            width: Video width
            height: Video height
            fps: Frame rate
            bitrate: Bitrate (bps)
            gop: GOP size
            qp: Default QP value
        """
        try:
            # Load dynamic library
            self.lib_path = find_library('rtmp_streamer')
            self.lib = ctypes.CDLL(self.lib_path)

            # Set function arguments and return types
            self._setup_functions()

            # Create streamer instance
            self.url_bytes = url.encode('utf-8')
            self.streamer = self.lib.create_rtmp_streamer(
                self.url_bytes, width, height, fps, bitrate, gop, qp
            )

            if not self.streamer:
                raise RuntimeError("Failed to create RTMP streamer")

            # Save parameters
            self.width = width
            self.height = height
            self.channels = 3  # RGB format (Note: BGR is typical for OpenCV)
            self.is_initialized = False
            self.is_streaming = False

            # Create frame data structure
            self.frame_data = self.lib.create_frame_data(width, height, self.channels)
            if not self.frame_data:
                raise RuntimeError("Failed to create frame data structure")

            print(f"RTMP Streamer successfully created, target: {url}")

        except Exception as e:
            print(f"Failed to initialize RTMP Streamer: {e}")
            self._cleanup()
            raise

    def _setup_functions(self):
        """Sets up C++ function arguments and return types"""
        # Create and destroy streamer
        self.lib.create_rtmp_streamer.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.create_rtmp_streamer.restype = ctypes.c_void_p

        self.lib.destroy_rtmp_streamer.argtypes = [ctypes.c_void_p]
        self.lib.destroy_rtmp_streamer.restype = None

        # Initialization and control
        self.lib.initialize_streamer.argtypes = [ctypes.c_void_p]
        self.lib.initialize_streamer.restype = ctypes.c_bool

        self.lib.start_streaming.argtypes = [ctypes.c_void_p]
        self.lib.start_streaming.restype = ctypes.c_bool

        self.lib.stop_streaming.argtypes = [ctypes.c_void_p]
        self.lib.stop_streaming.restype = None

        # Frame data operations
        self.lib.create_frame_data.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.create_frame_data.restype = ctypes.c_void_p

        self.lib.destroy_frame_data.argtypes = [ctypes.c_void_p]
        self.lib.destroy_frame_data.restype = None

        self.lib.add_roi_region.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.add_roi_region.restype = None

        self.lib.clear_roi_regions.argtypes = [ctypes.c_void_p]
        self.lib.clear_roi_regions.restype = None

        self.lib.set_frame_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
                                            ctypes.c_int, ctypes.c_int64]
        self.lib.set_frame_data.restype = ctypes.c_bool

        self.lib.push_frame.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.push_frame.restype = ctypes.c_bool

    def initialize(self) -> bool:
        """
        Initializes the streamer

        Returns:
            Whether initialization was successful
        """
        if not self.streamer:
            return False

        success = self.lib.initialize_streamer(self.streamer)
        self.is_initialized = success
        return success

    def start(self) -> bool:
        """
        Starts the streaming

        Returns:
            Whether starting the stream was successful
        """
        if not self.streamer:
            return False

        if not self.is_initialized:
            if not self.initialize():
                return False

        success = self.lib.start_streaming(self.streamer)
        self.is_streaming = success
        return success

    def stop(self):
        """Stops the streaming"""
        if self.streamer and self.is_streaming:
            self.lib.stop_streaming(self.streamer)
            self.is_streaming = False

    def push_frame(self, frame: np.ndarray, rois: Optional[List[ROIRegion]] = None) -> bool:
        """
        Pushes a video frame

        Args:
            frame: Video frame (NumPy array in BGR format)
            rois: List of ROI regions

        Returns:
            Whether pushing was successful
        """
        if not self.streamer or not self.is_streaming or not self.frame_data:
            return False

        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            print(f"Frame dimensions mismatch: Expected {self.width}x{self.height}, but received {frame.shape[1]}x{frame.shape[0]}")
            return False

        # Ensure frame is in contiguous memory
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        # Clear previous ROI regions
        self.lib.clear_roi_regions(self.frame_data)

        # Add new ROI regions
        if rois:
            for roi in rois:
                self.lib.add_roi_region(
                    self.frame_data, roi.x, roi.y, roi.width, roi.height, roi.qp
                )

        # Set frame data
        timestamp = int(time.time() * 1000)  # Millisecond timestamp
        ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        size = frame.size

        success = self.lib.set_frame_data(self.frame_data, ptr, size, timestamp)
        if not success:
            return False

        # Push frame
        return self.lib.push_frame(self.streamer, self.frame_data)

    def _cleanup(self):
        """Cleans up resources"""
        if hasattr(self, 'frame_data') and self.frame_data:
            self.lib.destroy_frame_data(self.frame_data)
            self.frame_data = None

        if hasattr(self, 'streamer') and self.streamer:
            self.stop()
            self.lib.destroy_rtmp_streamer(self.streamer)
            self.streamer = None

    def __del__(self):
        """Destructor"""
        self._cleanup()


# RTSP client callback function type
RTSP_FRAME_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int, ctypes.c_int, ctypes.c_int64
)


# RTSP Client class
class RTSPClient:
    """RTSP Client, used to fetch video streams from IP cameras"""

    def __init__(self, url: str, width: int = 0, height: int = 0):
        """
        Initializes the RTSP client

        Args:
            url: RTSP server URL
            width: Output image width (0 means use original width)
            height: Output image height (0 means use original height)
        """
        try:
            # Load dynamic library
            self.lib_path = find_library('rtsp_client')
            self.lib = ctypes.CDLL(self.lib_path)

            # Set function arguments and return types
            self._setup_functions()

            # Create client instance
            self.url_bytes = url.encode('utf-8')
            self.client = self.lib.create_rtsp_client(self.url_bytes, width, height)

            if not self.client:
                raise RuntimeError("Failed to create RTSP client")

            # Initialize callback related members
            self.callback = None
            self.user_callback = None
            self.callback_lock = threading.Lock()

            # State
            self.is_initialized = False
            self.is_running = False

            print(f"RTSP Client successfully created, source: {url}")

        except Exception as e:
            print(f"Failed to initialize RTSP Client: {e}")
            self._cleanup()
            raise

    def _setup_functions(self):
        """Sets up C++ function arguments and return types"""
        # Create and destroy client
        self.lib.create_rtsp_client.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        self.lib.create_rtsp_client.restype = ctypes.c_void_p

        self.lib.destroy_rtsp_client.argtypes = [ctypes.c_void_p]
        self.lib.destroy_rtsp_client.restype = None

        # Initialization and control
        self.lib.initialize_client.argtypes = [ctypes.c_void_p]
        self.lib.initialize_client.restype = ctypes.c_bool

        self.lib.start_client.argtypes = [ctypes.c_void_p]
        self.lib.start_client.restype = ctypes.c_bool

        self.lib.stop_client.argtypes = [ctypes.c_void_p]
        self.lib.stop_client.restype = None

        # Get information
        self.lib.get_fps.argtypes = [ctypes.c_void_p]
        self.lib.get_fps.restype = ctypes.c_double

        self.lib.get_width.argtypes = [ctypes.c_void_p]
        self.lib.get_width.restype = ctypes.c_int

        self.lib.get_height.argtypes = [ctypes.c_void_p]
        self.lib.get_height.restype = ctypes.c_int

        # Set callback
        self.lib.set_frame_callback.argtypes = [
            ctypes.c_void_p, RTSP_FRAME_CALLBACK, ctypes.c_void_p
        ]
        self.lib.set_frame_callback.restype = None

    def initialize(self) -> bool:
        """
        Initializes the client

        Returns:
            Whether initialization was successful
        """
        if not self.client:
            return False

        success = self.lib.initialize_client(self.client)
        self.is_initialized = success

        if success:
            # Get dimensions
            self.width = self.lib.get_width(self.client)
            self.height = self.lib.get_height(self.client)
            print(f"RTSP Client successfully initialized, resolution: {self.width}x{self.height}")

        return success

    def start(self) -> bool:
        """
        Starts receiving and processing the RTSP stream

        Returns:
            Whether starting was successful
        """
        if not self.client:
            return False

        if not self.is_initialized:
            if not self.initialize():
                return False

        success = self.lib.start_client(self.client)
        self.is_running = success
        return success

    def stop(self):
        """Stops receiving and processing the RTSP stream"""
        if self.client and self.is_running:
            self.lib.stop_client(self.client)
            self.is_running = False

    def set_frame_callback(self, callback: Callable[[np.ndarray, int], None]):
        """
        Sets the frame callback function

        Args:
            callback: The callback function, accepting frame (NumPy array) and timestamp arguments
        """
        if not self.client:
            return

        with self.callback_lock:
            self.user_callback = callback

            # C++ callback wrapper function
            @RTSP_FRAME_CALLBACK
            def frame_callback(user_data, data, width, height, timestamp):
                if self.user_callback:
                    try:
                        # Convert C++ pointer data to NumPy array
                        # Create a new NumPy array with its own memory to avoid pointer invalidation issues
                        size = width * height * 3  # RGB format
                        buffer = (ctypes.c_uint8 * size).from_address(ctypes.addressof(data.contents))
                        frame = np.frombuffer(buffer, dtype=np.uint8, count=size).copy()
                        frame = frame.reshape((height, width, 3))

                        # Call the user-provided callback function
                        self.user_callback(frame, timestamp)

                    except Exception as e:
                        print(f"RTSP frame callback error: {e}")

            # Save callback reference to prevent garbage collection
            self.callback = frame_callback

            # Set the callback on the C++ side
            self.lib.set_frame_callback(self.client, self.callback, None)

    def get_fps(self) -> float:
        """
        Gets the current FPS (Frames Per Second)

        Returns:
            The current frame rate
        """
        if not self.client:
            return 0.0

        return self.lib.get_fps(self.client)

    def _cleanup(self):
        """Cleans up resources"""
        if hasattr(self, 'client') and self.client:
            self.stop()
            self.lib.destroy_rtsp_client(self.client)
            self.client = None

    def __del__(self):
        """Destructor"""
        self._cleanup()


# Simple Test
if __name__ == "__main__":
    import time
    import cv2

    print("RTSP/RTMP Interface Test")

    try:
        # Test RTSP Client
        rtsp_url = "rtsp://admin:admin@192.168.1.100:554/stream"
        rtsp_client = RTSPClient(rtsp_url)

        # Set frame callback
        def on_frame(frame, timestamp):
            print(f"Received RTSP frame: {frame.shape}, Timestamp: {timestamp}")
            cv2.imshow("RTSP Frame", frame)
            cv2.waitKey(1)

        rtsp_client.set_frame_callback(on_frame)

        # Initialize and start
        if rtsp_client.initialize() and rtsp_client.start():
            print("RTSP Client started")

            # Run for 10 seconds
            for i in range(10):
                time.sleep(1)
                print(f"RTSP FPS: {rtsp_client.get_fps():.2f}")

            rtsp_client.stop()

        # Test RTMP Streamer
        rtmp_url = "rtmp://localhost/live/stream"
        rtmp_streamer = RTMPStreamer(rtmp_url, 640, 480)

        # Initialize and start
        if rtmp_streamer.initialize() and rtmp_streamer.start():
            print("RTMP Streamer started")

            # Create a test frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(test_frame, "RTMP Test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Push 10 frames
            for i in range(10):
                cv2.putText(test_frame, f"Frame {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                rtmp_streamer.push_frame(test_frame)
                time.sleep(1)

            rtmp_streamer.stop()

        print("Test completed")

    except Exception as e:
        print(f"Test Error: {e}")

    finally:
        cv2.destroyAllWindows()
