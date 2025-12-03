import cv2
import time
import threading
import re
import urllib.parse


class CameraStream:
    """
    Camera stream processing class, supporting local cameras and RTSP video streams.
    """

    def __init__(self, source, resolution=(640, 480)):
        """
        Initializes the camera stream.

        Args:
            source: Video source, can be an integer (local camera index) or a string (RTSP URL).
            resolution: Video resolution, format is (width, height).
        """
        self.source = source
        self.resolution = resolution
        self.cap = None
        self.running = False
        self.reconnect_thread = None
        self.lock = threading.Lock()    # Used for thread-safe access
        self.last_frame = None          # Caches the last frame, prevents returning None on failure
        self.frame_count = 0            # Number of frames processed
        self.fps = 0                    # Estimated FPS
        self.last_time = time.time()    # Time of the last FPS calculation

        # Open the camera
        self.open_camera()

    def open_camera(self):
        """Opens the camera or video stream."""
        try:
            # If there is already an active camera, close it first
            if self.cap is not None:
                self.cap.release()

            # Open the appropriate video stream based on the source type
            if isinstance(self.source, int):
                # Local camera
                self.cap = cv2.VideoCapture(self.source)
                print(f"Opened local camera {self.source}")

            elif isinstance(self.source, str):
                # Check if it is an RTSP or RTMP URL
                if self.source.lower().startswith(('rtsp://', 'rtmp://')):
                    # Use RTSP stream
                    # Set parameters for the RTSP stream
                    self.cap = cv2.VideoCapture(self.source)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Set small buffer to reduce latency
                    print(f"Opened network stream: {self._sanitize_url(self.source)}")

                elif self.source.lower().startswith(('http://', 'https://')):
                    # Regular HTTP stream
                    self.cap = cv2.VideoCapture(self.source)
                    print(f"Opened HTTP stream: {self._sanitize_url(self.source)}")

                else:
                    # Attempt to open as a video file
                    self.cap = cv2.VideoCapture(self.source)
                    print(f"Opened video file: {self.source}")
            else:
                raise ValueError(f"Unsupported video source type: {type(self.source)}")

            # Set resolution
            if self.resolution:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Check if successfully opened
            if not self.cap.isOpened():
                raise IOError(f"Unable to open video source: {self.source}")

            # Read one frame to test if it's working
            ret, _ = self.cap.read()
            if not ret:
                raise IOError(f"Unable to read from video source: {self.source}")

            self.running = True
            return True

        except Exception as e:
            print(f"Failed to open video source: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False

    def _sanitize_url(self, url):
        """Removes sensitive information (like username and password) from the URL for printing."""
        try:
            # Parse the URL
            parsed = urllib.parse.urlparse(url)

            # Check for username and password
            if '@' in parsed.netloc:
                # Replace sensitive information
                netloc = re.sub(r'[^@]+@', '***:***@', parsed.netloc)
                # Rebuild the URL
                sanitized = urllib.parse.urlunparse(
                    (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
                )
                return sanitized
            return url
        except:
            # If parsing fails, return a summary of the original URL
            return url[:8] + "..." + url[-8:] if len(url) > 20 else url

    def get_frame(self):
        """
        Gets one video frame.

        Returns:
            The video frame or None (if retrieval fails).
        """
        if not self.running or self.cap is None or not
