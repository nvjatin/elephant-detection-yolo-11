"""
Elephant Detection System using YOLOv8
Supports webcam, multi-camera streams, and video file processing.
Optimized for Apple Silicon with external SSD storage.

Usage:
    python detect.py --mode webcam
    python detect.py --mode multi-camera
    python detect.py --mode video --input video.mp4 --output output.mp4
"""

import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from threading import Thread, Lock
from pathlib import Path
import time
import json
import sys

# Add config to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import (
        MODEL_PATH,
        TRAINED_MODEL_PATH,
        PRETRAINED_MODEL_PATH,
        OUTPUT_PATH,
        VIDEO_OUTPUT_PATH,
        DETECTION_LOG_PATH,
        ELEPHANT_CLASS_ID,
        INFERENCE_CONFIG,
        DEFAULT_CAMERAS,
        get_device,
        ensure_directories,
        verify_ssd_mounted,
        get_model_path,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Fallback defaults
    ELEPHANT_CLASS_ID = 22
    MODEL_PATH = Path("/Volumes/Extended Storage/Elephant-Detection/models")
    OUTPUT_PATH = Path("/Volumes/Extended Storage/Elephant-Detection/outputs")
    VIDEO_OUTPUT_PATH = OUTPUT_PATH / "videos"
    DETECTION_LOG_PATH = OUTPUT_PATH / "detections"


class ElephantDetector:
    """Core detection engine - shared across all modes."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to ensure model is loaded only once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ElephantDetector._model is None:
            # Ensure directories exist
            if CONFIG_AVAILABLE:
                try:
                    verify_ssd_mounted()
                    ensure_directories()
                except Exception as e:
                    print(f"Warning: {e}")
            
            # Determine device
            device = get_device() if CONFIG_AVAILABLE else "mps"
            print(f"Using device: {device}")
            
            # Load model from SSD
            model_path = self._get_model_path()
            print(f"Loading YOLOv8 model from: {model_path}")
            ElephantDetector._model = YOLO(str(model_path))
            print("Model loaded successfully!")
    
    def _get_model_path(self) -> Path:
        """Get model path, preferring trained model on SSD."""
        if CONFIG_AVAILABLE:
            return get_model_path()
        
        # Fallback: check common locations
        paths = [
            TRAINED_MODEL_PATH / "best.pt",
            MODEL_PATH / "pretrained" / "yolov8n.pt",
            Path("yolov8n.pt"),
        ]
        for p in paths:
            if p.exists():
                return p
        return Path("yolov8n.pt")  # Will download if needed
    
    @property
    def model(self) -> YOLO:
        return ElephantDetector._model
    
    def detect(self, frame: np.ndarray, draw: bool = True) -> tuple[np.ndarray, list[dict]]:
        """
        Run elephant detection on a frame.
        
        Returns:
            tuple: (annotated_frame, list of detections)
        """
        detections = []
        results = self.model(frame, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id == ELEPHANT_CLASS_ID:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": confidence
                    })
                    
                    if draw:
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Draw label
                        label = f"Elephant: {confidence:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw alert banner if detected
        if draw and detections:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
            cv2.putText(frame, "ALERT: Elephant Detected!", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame, detections


class WebcamDetector:
    """Single webcam detection mode."""
    
    def __init__(self, source: int = 0):
        self.source = source
        self.detector = ElephantDetector()
    
    def start(self):
        """Start webcam detection."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("=" * 50)
        print("Webcam Elephant Detection Started")
        print("Press 'q' to quit")
        print("=" * 50)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break
            
            frame, detections = self.detector.detect(frame)
            
            for det in detections:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] DETECTED: Elephant (confidence: {det['confidence']:.2%})")
            
            cv2.imshow("Elephant Detection - Webcam", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nDetection stopped.")


class CameraProcessor:
    """Processes a single camera stream for elephant detection."""
    
    def __init__(self, camera_id: int, source: str, location: str, detector: ElephantDetector):
        self.camera_id = camera_id
        self.source = source
        self.location = location
        self.detector = detector
        
        self.cap = None
        self.frame = None
        self.processed_frame = None
        self.elephant_detected = False
        self.is_running = False
        self.is_connected = False
        self.lock = Lock()
        
        self.reconnect_delay = 5
        self.last_reconnect_attempt = 0
    
    def connect(self) -> bool:
        """Attempt to connect to the camera source."""
        try:
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                self.is_connected = True
                print(f"[Camera {self.camera_id}] Connected: {self.location}")
                return True
            else:
                self.is_connected = False
                print(f"[Camera {self.camera_id}] Failed to connect: {self.location}")
                return False
        except Exception as e:
            self.is_connected = False
            print(f"[Camera {self.camera_id}] Connection error: {e}")
            return False
    
    def reconnect(self):
        """Attempt to reconnect after disconnection."""
        current_time = time.time()
        if current_time - self.last_reconnect_attempt >= self.reconnect_delay:
            self.last_reconnect_attempt = current_time
            print(f"[Camera {self.camera_id}] Attempting reconnection: {self.location}")
            if self.cap:
                self.cap.release()
            self.connect()
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """Run detection on a frame."""
        frame, detections = self.detector.detect(frame)
        
        for det in detections:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] ALERT - Camera {self.camera_id} ({self.location}): "
                  f"Elephant detected (confidence: {det['confidence']:.2%})")
        
        # Add camera label at bottom
        label_text = f"Camera {self.camera_id} - {self.location}"
        cv2.rectangle(frame, (0, frame.shape[0] - 30), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
        cv2.putText(frame, label_text, (10, frame.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, len(detections) > 0
    
    def start(self):
        """Start the camera processing thread."""
        self.is_running = True
        self.connect()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def _run(self):
        """Main processing loop."""
        while self.is_running:
            if not self.is_connected:
                self.reconnect()
                if not self.is_connected:
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "DISCONNECTED", (180, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(placeholder, f"Camera {self.camera_id} - {self.location}",
                                (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    with self.lock:
                        self.processed_frame = placeholder
                        self.elephant_detected = False
                    time.sleep(1)
                    continue
            
            ret, frame = self.cap.read()
            if not ret:
                self.is_connected = False
                print(f"[Camera {self.camera_id}] Disconnected: {self.location}")
                continue
            
            processed, detected = self.process_frame(frame)
            
            with self.lock:
                self.processed_frame = processed
                self.elephant_detected = detected
    
    def get_frame(self) -> tuple[np.ndarray, bool]:
        """Get the latest processed frame."""
        with self.lock:
            if self.processed_frame is not None:
                return self.processed_frame.copy(), self.elephant_detected
            return None, False
    
    def stop(self):
        """Stop the camera processing."""
        self.is_running = False
        if self.cap:
            self.cap.release()


class MultiCameraDetector:
    """Manages multiple camera processors and displays grid view."""
    
    def __init__(self, cameras: list[dict], grid_size: tuple[int, int] = (2, 2)):
        self.grid_size = grid_size
        self.cell_width = 640
        self.cell_height = 480
        
        self.detector = ElephantDetector()
        
        self.processors = []
        for i, cam in enumerate(cameras):
            processor = CameraProcessor(
                camera_id=i + 1,
                source=cam["source"],
                location=cam["location"],
                detector=self.detector
            )
            self.processors.append(processor)
    
    def create_grid(self) -> np.ndarray:
        """Create a grid view of all camera feeds."""
        rows, cols = self.grid_size
        grid = np.zeros((rows * self.cell_height, cols * self.cell_width, 3), dtype=np.uint8)
        
        for i, processor in enumerate(self.processors):
            row = i // cols
            col = i % cols
            
            frame, _ = processor.get_frame()
            if frame is not None:
                resized = cv2.resize(frame, (self.cell_width, self.cell_height))
                
                y_start = row * self.cell_height
                y_end = (row + 1) * self.cell_height
                x_start = col * self.cell_width
                x_end = (col + 1) * self.cell_width
                
                grid[y_start:y_end, x_start:x_end] = resized
        
        return grid
    
    def start(self):
        """Start all camera processors and display loop."""
        print("=" * 60)
        print("Multi-Camera Elephant Detection System Started")
        print("Press 'q' to quit")
        print("=" * 60)
        
        for processor in self.processors:
            processor.start()
        
        time.sleep(2)
        
        while True:
            grid = self.create_grid()
            cv2.imshow("Elephant Detection - Multi-Camera View", grid)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        for processor in self.processors:
            processor.stop()
        cv2.destroyAllWindows()
        print("\nDetection system stopped.")


class VideoFileDetector:
    """Process video files for elephant detection."""
    
    def __init__(self, input_path: str, output_path: str = None):
        self.input_path = Path(input_path)
        
        # Use SSD output path if not specified
        if output_path:
            self.output_path = Path(output_path)
        else:
            # Save to SSD outputs folder
            VIDEO_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
            self.output_path = VIDEO_OUTPUT_PATH / f"{self.input_path.stem}_detected.mp4"
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Detection log goes to SSD
        DETECTION_LOG_PATH.mkdir(parents=True, exist_ok=True)
        self.log_path = DETECTION_LOG_PATH / f"{self.input_path.stem}_detections.json"
        
        self.detector = ElephantDetector()
        self.detection_log = []
    
    def format_timestamp(self, frame_num: int, fps: float) -> str:
        """Convert frame number to timestamp string."""
        total_seconds = frame_num / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    
    def process(self, show_preview: bool = True):
        """Process the video file."""
        if not self.input_path.exists():
            print(f"Error: Input file not found: {self.input_path}")
            return
        
        cap = cv2.VideoCapture(str(self.input_path))
        if not cap.isOpened():
            print(f"Error: Could not open video: {self.input_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        
        print("=" * 60)
        print("Video File Elephant Detection")
        print(f"Input:  {self.input_path}")
        print(f"Output: {self.output_path}")
        print(f"Log:    {self.log_path}")
        print(f"Resolution: {width}x{height} @ {fps:.2f} FPS")
        print(f"Total frames: {total_frames}")
        if show_preview:
            print("Press 'q' to cancel, 's' to skip preview")
        print("=" * 60)
        
        frame_num = 0
        show_window = show_preview
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            timestamp = self.format_timestamp(frame_num, fps)
            
            # Run detection
            processed_frame, detections = self.detector.detect(frame)
            
            # Add timestamp overlay
            cv2.putText(processed_frame, timestamp, (width - 150, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Log detections
            if detections:
                log_entry = {
                    "frame": frame_num,
                    "timestamp": timestamp,
                    "detections": [
                        {
                            "confidence": det["confidence"],
                            "bbox": det["bbox"]
                        }
                        for det in detections
                    ]
                }
                self.detection_log.append(log_entry)
                
                print(f"[{timestamp}] Frame {frame_num}: "
                      f"{len(detections)} elephant(s) detected")
            
            # Write to output
            out.write(processed_frame)
            
            # Progress update
            if frame_num % 100 == 0:
                progress = (frame_num / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_num}/{total_frames})")
            
            # Show preview
            if show_window:
                preview = cv2.resize(processed_frame, (width // 2, height // 2))
                cv2.imshow("Video Processing Preview", preview)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nProcessing cancelled by user.")
                    break
                elif key == ord('s'):
                    show_window = False
                    cv2.destroyAllWindows()
                    print("Preview disabled. Processing continues...")
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Save detection log
        self._save_log(fps, total_frames)
        
        print("\n" + "=" * 60)
        print("Processing Complete!")
        print(f"Total detections: {len(self.detection_log)} frames with elephants")
        print(f"Output video: {self.output_path}")
        print(f"Detection log: {self.log_path}")
        print("=" * 60)
    
    def _save_log(self, fps: float, total_frames: int):
        """Save detection log to JSON file."""
        log_data = {
            "input_file": str(self.input_path),
            "output_file": str(self.output_path),
            "video_info": {
                "fps": fps,
                "total_frames": total_frames,
                "duration": self.format_timestamp(total_frames, fps)
            },
            "detection_count": len(self.detection_log),
            "detections": self.detection_log
        }
        
        with open(self.log_path, 'w') as f:
            json.dump(log_data, f, indent=2)


def get_default_cameras() -> list[dict]:
    """Return default camera configuration."""
    if CONFIG_AVAILABLE:
        return DEFAULT_CAMERAS
    return [
        {"source": 0, "location": "Village A"},
        {"source": "http://192.168.1.101:8080/video", "location": "Village B"},
        {"source": "http://192.168.1.102:8080/video", "location": "Forest Edge"},
        {"source": "rtsp://192.168.1.103:554/stream", "location": "Highway"},
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Elephant Detection System using YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam mode (default)
  python detect.py --mode webcam
  
  # Multi-camera mode
  python detect.py --mode multi-camera
  
  # Video file mode
  python detect.py --mode video --input wildlife.mp4
  python detect.py --mode video --input wildlife.mp4 --output result.mp4
  python detect.py --mode video --input wildlife.mp4 --no-preview
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["webcam", "multi-camera", "video"],
        default="webcam",
        help="Detection mode (default: webcam)"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input video file path (required for video mode)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output video file path (optional, auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview window during video processing"
    )
    
    parser.add_argument(
        "--webcam-id",
        type=int,
        default=0,
        help="Webcam device ID (default: 0)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "webcam":
        detector = WebcamDetector(source=args.webcam_id)
        detector.start()
    
    elif args.mode == "multi-camera":
        cameras = get_default_cameras()
        detector = MultiCameraDetector(cameras, grid_size=(2, 2))
        detector.start()
    
    elif args.mode == "video":
        if not args.input:
            parser.error("--input is required for video mode")
        
        detector = VideoFileDetector(
            input_path=args.input,
            output_path=args.output
        )
        detector.process(show_preview=not args.no_preview)


if __name__ == "__main__":
    main()
