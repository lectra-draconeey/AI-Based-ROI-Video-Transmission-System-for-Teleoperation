import cv2
import numpy as np
import os
from ultralytics import YOLO
import sys
import time


class Processor:
    """AI Processor class, responsible for intelligent processing of video frames"""

    def __init__(self, model_weights="yolo11n", model_cfg=None, class_names=None):
        """
        Initializes the AI Processor
        

        Args:
            model_weights: YOLO model name (without suffix, e.g., 'yolo11n') or custom model path
            model_cfg: Deprecated, kept for backward compatibility
            class_names: Deprecated, kept for backward compatibility
        """
        # Performance metrics
        self.inference_times = []      # list of inference durations in ms
        self.total_frames = 0          # total frames submitted for processing
        self.failed_frames = 0         # frames where inference raised an exception

        try:
            # Ensure the models directory exists
            models_dir = os.path.abspath('models')
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
                print("Created models directory")

            # Note: The environment variable is set in main.py
            # os.environ['YOLO_CONFIG_DIR'] = os.path.abspath('models')

            # Check whether to download the model or use a local model
            model_path = model_weights
            if not os.path.exists(model_weights) and not model_weights.startswith(models_dir):
                # If it's not an absolute path, try to find it in the models directory
                potential_model_path = os.path.join(models_dir, f"{model_weights}.pt")
                if os.path.exists(potential_model_path):
                    model_path = potential_model_path
                    print(f"Using local model: {model_path}")


            # Load model using ultralytics
            # Prioritize automatic selection of the most suitable model version for the system
            self.net = YOLO(model_path)
            print(f"Successfully loaded YOLOv11 model: {model_weights}")

            # Get the list of classes supported by the model
            self.classes = self.net.names
            print(f"Number of classes supported by the model: {len(self.classes)}")

        except Exception as e:
            raise Exception(f"Failed to load model: {e}")

        # Set detection parameters
        self.conf_threshold = 0.5  # Confidence threshold
        self.nms_threshold = 0.4   # Non-Maximum Suppression threshold

    def process_frame(self, frame, return_rois=False, roi_qp=15, non_roi_qp=35):
        """
        Processes the video frame, performs object detection, and applies differential encoding

        Args:
            frame: Input video frame
            return_rois: Whether to return the detected ROI region information
            roi_qp: QP value for the ROI region (Lower value = Higher Quality)
            non_roi_qp: QP value for the non-ROI region (Higher value = Lower Quality)

        Returns:
            Processed frame, and ROI region information (if return_rois=True)
        """
        if frame is None:
            if return_rois:
                return None, []
            return None

        # Save a copy of the original frame
        original_frame = frame.copy()

        # Get image dimensions
        height, width, _ = frame.shape

        self.total_frames += 1

        try:
            # Perform object detection using YOLOv11
            start_time = time.perf_counter()
            results = self.net(frame, conf=self.conf_threshold, iou=self.nms_threshold, verbose=False)
            end_time = time.perf_counter()
            inference_ms = (end_time - start_time) * 1000
            self.inference_times.append(inference_ms)
        except Exception as e:
            self.failed_frames += 1
            print(f"Error during inference: {e}")
            if return_rois:
                return original_frame, []
            return original_frame    


        # Create a mask image to mark ROI regions
        mask = np.zeros((height, width), dtype=np.uint8)

        # Parse detection results
        rois = []  # List of ROI regions

        # Process each detection result
        for result in results:
            boxes = result.boxes  # Get bounding boxes

            # Process each detected object
            for i in range(len(boxes)):
                # Get bounding box coordinates
                box = boxes[i].xyxy[0].cpu().numpy()  # Convert to numpy array
                x1, y1, x2, y2 = box
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)

                # Get class and confidence
                cls_id = int(boxes[i].cls[0].item())
                conf = float(boxes[i].conf[0].item())

                # Ensure coordinates do not exceed image boundaries
                x = max(0, x)
                y = max(0, y)
                right = min(width, x + w)
                bottom = min(height, y + h)

                # Mark the detected object area as ROI
                cv2.rectangle(mask, (x, y), (right, bottom), 255, -1)  # Fill the ROI region

                # Draw bounding box and label on the frame
                self.draw_prediction(frame, cls_id, conf, x, y, right, bottom)

                # Add to ROI list
                roi_info = {
                    'class': self.classes[cls_id],
                    'confidence': conf,
                    'box': [x, y, w, h]
                }
                rois.append(roi_info)

        # Apply differential encoding (simulating encoding with different QP values)
        # In a real application, this should call the encoder API to set QP values for different regions
        # Here we use JPEG compression to simulate different quality encoding effects
        roi_encoded = self.simulate_encoding(original_frame, mask, roi_qp, non_roi_qp)

        if return_rois:
            return roi_encoded, rois
        return roi_encoded

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        """
        Draws detection results on the image

        Args:
            img: Image to draw on
            class_id: Class ID
            confidence: Confidence score
            x, y: Top-left coordinates
            x_plus_w, y_plus_h: Bottom-right coordinates
        """
        # Get class name
        label = self.classes[class_id] if class_id in self.classes else f"Unknown Class-{class_id}"

        # Format confidence
        conf_text = f"{confidence:.2f}"

        # Draw bounding box
        color = self.get_color_for_class(class_id)
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

        # Draw class label background
        text_size, _ = cv2.getTextSize(f"{label} {conf_text}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x, y - text_size[1] - 5), (x + text_size[0], y), color, -1)

        # Draw class label text
        cv2.putText(img, f"{label} {conf_text}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def get_color_for_class(self, class_id):
        """
        Generates different colors for different classes

        Args:
            class_id: Class ID

        Returns:
            BGR color value
        """
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (127, 255, 0),  # Yellow-Green
            (255, 127, 0),  # Blue-Violet
            (127, 0, 255),  # Red-Violet
            (255, 255, 255) # White
        ]
        return colors[class_id % len(colors)]

    def simulate_encoding(self, original_frame, mask, roi_qp, non_roi_qp):
        """
        Simulates the effect of encoding different regions with different QP values

        Args:
            original_frame: Original video frame
            mask: ROI region mask
            roi_qp: QP value for the ROI region (Lower value = Higher Quality)
            non_roi_qp: QP value for the non-ROI region (Higher value = Lower Quality)

        Returns:
            Differentially encoded frame
        """
        # Convert QP values to JPEG quality parameters (0-100, inverse of QP value, higher value = higher quality)
        roi_quality = max(5, min(100, int(100 - (roi_qp * 2.5))))
        non_roi_quality = max(5, min(100, int(100 - (non_roi_qp * 2.5))))

        # Process ROI region (High Quality)
        roi_frame = original_frame.copy()
        # Encode ROI region (using high quality)
        _, roi_encoded = cv2.imencode('.jpg', roi_frame, [cv2.IMWRITE_JPEG_QUALITY, roi_quality])
        roi_decoded = cv2.imdecode(roi_encoded, cv2.IMREAD_COLOR)

        # Process Non-ROI region (Low Quality)
        non_roi_frame = original_frame.copy()
        # Encode Non-ROI region (using low quality)
        _, non_roi_encoded = cv2.imencode('.jpg', non_roi_frame, [cv2.IMWRITE_JPEG_QUALITY, non_roi_quality])
        non_roi_decoded = cv2.imdecode(non_roi_encoded, cv2.IMREAD_COLOR)

        # Merge the two regions
        # Convert mask to a suitable format for merging
        mask_3ch = cv2.merge([mask, mask, mask])

        # Use the mask to merge the two frames of different quality
        result = np.where(mask_3ch > 0, roi_decoded, non_roi_decoded)

        # Add text description
        cv2.putText(result, f"ROI QP: {roi_qp}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(result, f"Non-ROI QP: {non_roi_qp}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return result.astype(np.uint8)
    
    def get_inference_metrics(self):
        """
        Returns aggregated inference performance metrics.
        
        Returns:
            dict: Contains avg_inference_ms, fps, stability, total_frames, failed_frames
        """
        if not self.inference_times:
            return {
                "avg_inference_ms": 0.0,
                "fps": 0.0,
                "stability": 0.0,
                "total_frames": self.total_frames,
                "failed_frames": self.failed_frames
            }

        avg_inf = sum(self.inference_times) / len(self.inference_times)
        fps = 1000.0 / avg_inf
        stability = (self.total_frames - self.failed_frames) / self.total_frames if self.total_frames > 0 else 0.0

        return {
            "avg_inference_ms": round(avg_inf, 2),
            "fps": round(fps, 2),
            "stability": round(stability, 4),
            "total_frames": self.total_frames,
            "failed_frames": self.failed_frames
        }

    # Example usage
    # ai_processor = Processor('yolov3.weights', 'yolov3.cfg', 'coco.names')
