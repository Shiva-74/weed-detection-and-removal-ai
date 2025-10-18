
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import argparse
from pathlib import Path
from collections import deque
import threading
import queue
import json
from datetime import datetime

class PrecisionWeedLaserDetector:
    def __init__(self, model_path, source, output_path=None, confidence=0.25, iou=0.5):
        # Load trained model
        self.model = YOLO(model_path)
        self.source = source
        self.output_path = output_path
        self.confidence = confidence
        self.iou = iou

        # Detection classes (adjust based on your model)
        self.class_names = self.model.names if hasattr(self.model, 'names') else ['crop', 'weed']
        self.weed_classes = [1]  # Adjust if your weed class index is different
        self.crop_classes = [0]  # Adjust if your crop class index is different

        # Laser action system
        self.laser_queue = queue.Queue()
        self.laser_actions = []
        self.setup_laser_system()

        # Performance tracking
        self.fps_history = deque(maxlen=60)
        self.inference_times = deque(maxlen=60)
        self.detection_history = []

        # Colors for visualization
        self.colors = {
            'weed': (0, 0, 255),      # Red for weeds
            'crop': (0, 255, 0),      # Green for crops
            'laser': (0, 255, 255),   # Yellow for laser targeting
            'crosshair': (255, 0, 255) # Magenta for crosshair
        }

        print(f"🎯 Precision Weed Laser Detector Initialized")
        print(f"Model: {model_path}")
        print(f"Confidence: {confidence}")
        print(f"Source: {source}")
        print(f"Classes: {self.class_names}")

    def setup_laser_system(self):
        """Initialize simulated laser targeting system"""
        self.laser_available = True
        self.laser_power_levels = {'low': 0.3, 'medium': 0.6, 'high': 1.0}

        # Start laser control thread
        self.laser_thread = threading.Thread(target=self.laser_control_worker, daemon=True)
        self.laser_thread.start()

        print("🔴 Laser targeting system ready")

    def laser_control_worker(self):
        """Worker thread for laser actions"""
        while True:
            try:
                laser_command = self.laser_queue.get(timeout=1)
                self.execute_laser_action(laser_command)
                self.laser_queue.task_done()
            except queue.Empty:
                continue

    def calculate_laser_parameters(self, bbox_area, confidence):
        """Calculate optimal laser power and duration"""
        # Power based on weed size
        if bbox_area < 0.01:
            power = self.laser_power_levels['low']
        elif bbox_area < 0.05:
            power = self.laser_power_levels['medium']
        else:
            power = self.laser_power_levels['high']

        # Duration based on confidence
        duration = max(0.1, confidence * 0.4)

        return power, duration

    def execute_laser_action(self, laser_command):
        """Execute precision laser targeting"""
        x, y, bbox_area, confidence, class_name, timestamp = laser_command

        power, duration = self.calculate_laser_parameters(bbox_area, confidence)

        laser_action = {
            'timestamp': timestamp,
            'target_x': x,
            'target_y': y,
            'bbox_area': bbox_area,
            'confidence': confidence,
            'class_name': class_name,
            'laser_power': power,
            'laser_duration': duration,
            'status': 'LASER_FIRED'
        }

        self.laser_actions.append(laser_action)

        print(f"🔴 LASER: {class_name} at ({x},{y}) | Conf: {confidence:.2f} | Power: {power:.1f} | Duration: {duration:.2f}s")

        # Simulate laser firing time
        time.sleep(duration)

    def draw_targeting_grid(self, frame):
        """Draw precision targeting grid"""
        h, w = frame.shape[:2]

        # Draw 5x5 targeting grid
        for i in range(1, 5):
            # Vertical lines
            x = i * w // 5
            cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 1, cv2.LINE_AA)
            # Horizontal lines
            y = i * h // 5
            cv2.line(frame, (0, y), (w, y), (100, 100, 100), 1, cv2.LINE_AA)

        # Zone labels
        zones = [f"Z{i+1:02d}" for i in range(25)]
        for i, zone in enumerate(zones):
            row, col = i // 5, i % 5
            x_pos = col * w // 5 + 5
            y_pos = row * h // 5 + 20
            cv2.putText(frame, zone, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        return frame

    def calculate_detection_metrics(self, x1, y1, x2, y2, frame_w, frame_h):
        """Calculate precise detection metrics"""
        # Bounding box metrics
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = (bbox_width * bbox_height) / (frame_w * frame_h)  # Normalized

        # Center coordinates
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Aspect ratio
        aspect_ratio = bbox_width / max(bbox_height, 1)

        return {
            'center_x': center_x,
            'center_y': center_y,
            'bbox_area': bbox_area,
            'aspect_ratio': aspect_ratio,
            'width': bbox_width,
            'height': bbox_height
        }

    def process_detections(self, frame, results):
        """Process detections with precision laser targeting"""
        h, w = frame.shape[:2]
        frame = self.draw_targeting_grid(frame)

        detection_counts = {'weeds': 0, 'crops': 0, 'total_detections': 0}
        frame_detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    if conf > self.confidence:
                        detection_counts['total_detections'] += 1

                        # Calculate detection metrics
                        metrics = self.calculate_detection_metrics(x1, y1, x2, y2, w, h)
                        class_name = self.class_names[cls] if cls < len(self.class_names) else f'class_{cls}'

                        # Determine if weed or crop
                        is_weed = cls in self.weed_classes
                        is_crop = cls in self.crop_classes

                        if is_weed:
                            detection_counts['weeds'] += 1
                            color = self.colors['weed']
                            label_prefix = "WEED"

                            # Queue laser action for weeds
                            laser_command = (
                                metrics['center_x'], metrics['center_y'],
                                metrics['bbox_area'], conf, class_name,
                                time.time()
                            )

                            try:
                                self.laser_queue.put_nowait(laser_command)
                            except queue.Full:
                                print("⚠️ Laser queue full - skipping target")

                            # Draw laser targeting elements
                            cv2.drawMarker(frame, (metrics['center_x'], metrics['center_y']), 
                                         self.colors['laser'], cv2.MARKER_CROSS, 25, 3)

                            # Draw targeting circle
                            radius = max(8, int(metrics['bbox_area'] * 200))
                            cv2.circle(frame, (metrics['center_x'], metrics['center_y']), 
                                     radius, self.colors['crosshair'], 2)

                        elif is_crop:
                            detection_counts['crops'] += 1
                            color = self.colors['crop']
                            label_prefix = "CROP"

                        else:
                            continue

                        # Draw precise bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Create detailed label
                        main_label = f"{label_prefix}: {class_name}"
                        detail_label = f"Conf: {conf:.2f} | Area: {metrics['bbox_area']:.3f}"

                        # Draw label background
                        label_h = 40
                        cv2.rectangle(frame, (x1, y1 - label_h), (x1 + 250, y1), color, -1)

                        # Draw labels
                        cv2.putText(frame, main_label, (x1 + 5, y1 - 25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(frame, detail_label, (x1 + 5, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                        # Store detection for analysis
                        detection_data = {
                            'timestamp': time.time(),
                            'class': class_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'center': [metrics['center_x'], metrics['center_y']],
                            'area': metrics['bbox_area'],
                            'is_weed': is_weed
                        }
                        frame_detections.append(detection_data)

        # Store frame detections
        self.detection_history.append({
            'timestamp': time.time(),
            'detections': frame_detections,
            'counts': detection_counts.copy()
        })

        # Draw comprehensive statistics
        self.draw_detection_statistics(frame, detection_counts)

        return frame

    def draw_detection_statistics(self, frame, detection_counts):
        """Draw comprehensive detection statistics"""
        h, w = frame.shape[:2]

        # Statistics background
        stats_bg_y = h - 120
        cv2.rectangle(frame, (10, stats_bg_y), (400, h - 10), (0, 0, 0), -1)

        # Current frame stats
        y_offset = stats_bg_y + 20
        cv2.putText(frame, f"Weeds Detected: {detection_counts['weeds']}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['weed'], 2)

        y_offset += 20
        cv2.putText(frame, f"Crops Protected: {detection_counts['crops']}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['crop'], 2)

        y_offset += 20
        cv2.putText(frame, f"Total Detections: {detection_counts['total_detections']}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y_offset += 20
        cv2.putText(frame, f"Laser Actions: {len(self.laser_actions)}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['laser'], 2)

        y_offset += 20
        cv2.putText(frame, f"Queue Size: {self.laser_queue.qsize()}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def draw_performance_metrics(self, frame):
        """Draw real-time performance metrics"""
        if self.fps_history and self.inference_times:
            avg_fps = np.mean(self.fps_history)
            avg_inference = np.mean(self.inference_times) * 1000

            # Performance background
            cv2.rectangle(frame, (10, 10), (350, 110), (0, 0, 0), -1)

            # Performance metrics
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Inference: {avg_inference:.1f}ms", (15, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {self.confidence:.2f}", (15, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Model info
            cv2.putText(frame, f"Model: YOLO Weed Detector", (200, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", (200, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def save_detection_report(self, output_dir="detection_reports"):
        """Save detailed detection report"""
        Path(output_dir).mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detection history
        detection_report = {
            'session_info': {
                'timestamp': timestamp,
                'model_path': str(self.model),
                'confidence_threshold': self.confidence,
                'iou_threshold': self.iou,
                'total_frames_processed': len(self.detection_history),
                'total_laser_actions': len(self.laser_actions)
            },
            'detection_history': self.detection_history,
            'laser_actions': self.laser_actions
        }

        report_path = Path(output_dir) / f"detection_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(detection_report, f, indent=2)

        print(f"📊 Detection report saved: {report_path}")
        return str(report_path)

    def run_detection(self):
        """Main detection loop"""
        # Open video source
        if str(self.source).isdigit():
            cap = cv2.VideoCapture(int(self.source))
        else:
            cap = cv2.VideoCapture(str(self.source))

        if not cap.isOpened():
            print(f"❌ Error: Cannot open video source {self.source}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📹 Video Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        if total_frames > 0:
            print(f"  Total frames: {total_frames}")
            print(f"  Duration: {total_frames/fps:.1f}s")

        # Setup video writer for output
        out = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
            print(f"💾 Output will be saved to: {self.output_path}")

        print("\n🚀 Starting precision weed detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'r' - Save detection report")
        print("  'p' - Pause/Resume")
        print("  '+'/'-' - Increase/Decrease confidence")
        print("="*60)

        frame_count = 0
        start_time = time.time()
        paused = False

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("📹 End of video stream")
                        break

                    # Run inference
                    inference_start = time.time()
                    results = self.model(
                        frame,
                        conf=self.confidence,
                        iou=self.iou,
                        verbose=False
                    )
                    inference_time = time.time() - inference_start
                    self.inference_times.append(inference_time)

                    # Process detections
                    processed_frame = self.process_detections(frame, results)
                    processed_frame = self.draw_performance_metrics(processed_frame)

                    # Calculate FPS
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    self.fps_history.append(current_fps)

                    # Save frame to output video
                    if out:
                        out.write(processed_frame)

                    # Progress update
                    if total_frames > 0 and frame_count % 60 == 0:
                        progress = (frame_count / total_frames) * 100
                        elapsed_min = elapsed_time / 60
                        weeds_per_min = len(self.laser_actions) / max(elapsed_min, 0.1)
                        print(f"Progress: {progress:.1f}% | Weeds/min: {weeds_per_min:.1f} | FPS: {current_fps:.1f}")

                else:
                    processed_frame = frame.copy()
                    cv2.putText(processed_frame, "⏸️ PAUSED - Press 'p' to resume", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display frame
                cv2.imshow('🎯 Precision Weed Laser Detection System', processed_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Stopping detection...")
                    break
                elif key == ord('s'):
                    save_path = f"precision_frame_{frame_count:06d}.jpg"
                    cv2.imwrite(save_path, processed_frame)
                    print(f"💾 Frame saved: {save_path}")
                elif key == ord('r'):
                    report_path = self.save_detection_report()
                    print(f"📊 Report saved: {report_path}")
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
                elif key == ord('+') or key == ord('='):
                    self.confidence = min(0.95, self.confidence + 0.05)
                    print(f"🔺 Confidence: {self.confidence:.2f}")
                elif key == ord('-'):
                    self.confidence = max(0.05, self.confidence - 0.05)
                    print(f"🔻 Confidence: {self.confidence:.2f}")

        except KeyboardInterrupt:
            print("\n⚠️ Detection interrupted by user")

        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

            # Generate final report
            self.generate_final_report(frame_count, time.time() - start_time)

    def generate_final_report(self, total_frames, total_time):
        """Generate comprehensive final report"""
        print("\n" + "="*80)
        print("🎯 PRECISION WEED LASER DETECTION - FINAL REPORT")
        print("="*80)

        # Basic statistics
        print(f"📊 Session Statistics:")
        print(f"  Total frames processed: {total_frames:,}")
        print(f"  Processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"  Average FPS: {total_frames/total_time:.2f}")

        # Detection statistics
        total_weeds = sum(1 for action in self.laser_actions)
        total_crops = sum(frame_data['counts']['crops'] for frame_data in self.detection_history)

        print(f"\n🌿 Detection Results:")
        print(f"  Total weeds detected: {total_weeds:,}")
        print(f"  Total crops protected: {total_crops:,}")
        print(f"  Weeds per minute: {total_weeds/(total_time/60):.1f}")

        # Laser system performance
        if self.laser_actions:
            avg_confidence = np.mean([action['confidence'] for action in self.laser_actions])
            avg_laser_time = np.mean([action['laser_duration'] for action in self.laser_actions])
            total_laser_time = sum([action['laser_duration'] for action in self.laser_actions])

            print(f"\n🔴 Laser System Performance:")
            print(f"  Total laser actions: {len(self.laser_actions):,}")
            print(f"  Average confidence: {avg_confidence:.3f}")
            print(f"  Average laser duration: {avg_laser_time:.3f}s")
            print(f"  Total laser time: {total_laser_time:.1f}s")

        # Performance metrics
        if self.fps_history and self.inference_times:
            avg_fps = np.mean(self.fps_history)
            avg_inference = np.mean(self.inference_times) * 1000

            print(f"\n⚡ Performance Metrics:")
            print(f"  Average FPS: {avg_fps:.2f}")
            print(f"  Average inference time: {avg_inference:.1f}ms")
            print(f"  Peak FPS: {np.max(self.fps_history):.2f}")
            print(f"  Min inference time: {np.min(self.inference_times)*1000:.1f}ms")

        # Save final report
        report_path = self.save_detection_report()
        if self.output_path:
            print(f"\n💾 Files saved:")
            print(f"  Video output: {self.output_path}")
            print(f"  Detection report: {report_path}")

        print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Precision Weed Laser Detection System')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                       help='Video source (0 for webcam, path to video file)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (optional)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for NMS (0.0-1.0)')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.model).exists():
        print(f"❌ Error: Model file not found: {args.model}")
        return

    if not str(args.source).isdigit() and not Path(args.source).exists():
        print(f"❌ Error: Video source not found: {args.source}")
        return

    # Initialize and run detector
    detector = PrecisionWeedLaserDetector(
        model_path=args.model,
        source=args.source,
        output_path=args.output,
        confidence=args.conf,
        iou=args.iou
    )

    detector.run_detection()

if __name__ == "__main__":
    main()
