import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import yaml
from pathlib import Path
from collections import deque
import threading
import queue
import argparse
import math

class PrecisionLaserWeedSystem:
    def __init__(self, model_path, config_path, source, output_path=None):
        self.model = YOLO(model_path)
        self.source = source
        self.output_path = output_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Precision detection parameters
        self.conf_threshold = self.config['deployment']['confidence']
        self.iou_threshold = self.config['deployment']['iou_threshold']
        self.max_detections = self.config['deployment']['max_detections']
        
        # Class configuration
        self.class_names = self.config['dataset']['names']
        self.weed_classes = set(self.config.get('weed_classes', [1]))
        self.crop_classes = set(self.config.get('crop_classes', [0]))
        
        # Precision laser system
        self.laser_queue = queue.Queue()
        self.laser_actions = []
        self.setup_precision_laser()
        
        # Performance tracking
        self.fps_history = deque(maxlen=60)
        self.inference_times = deque(maxlen=60)
        self.detection_history = deque(maxlen=100)
        
        # Colors for visualization
        self.weed_color = (0, 0, 255)      # Red for weeds
        self.crop_color = (0, 255, 0)      # Green for crops
        self.laser_color = (0, 255, 255)   # Yellow for laser
        self.target_color = (255, 0, 255)  # Magenta for targets
        
        print("🎯 Precision Laser Weed System Initialized")
        print(f"Model: {model_path}")
        print(f"Weed classes: {self.weed_classes}")
        print(f"Crop classes: {self.crop_classes}")
    
    def setup_precision_laser(self):
        """Setup precision laser control system"""
        self.laser_available = True
        self.laser_calibrated = True
        
        # Laser specifications (adjustable)
        self.laser_power_levels = {
            'low': 0.3,     # Small weeds
            'medium': 0.6,  # Medium weeds  
            'high': 1.0     # Large weeds
        }
        
        # Start precision laser thread
        self.laser_thread = threading.Thread(target=self.precision_laser_worker, daemon=True)
        self.laser_thread.start()
        
        print("🔴 Precision laser system ready")
    
    def precision_laser_worker(self):
        """Precision laser control worker"""
        while True:
            try:
                laser_command = self.laser_queue.get(timeout=1)
                self.execute_precision_laser(laser_command)
                self.laser_queue.task_done()
            except queue.Empty:
                continue
    
    def calculate_laser_power(self, bbox_area, confidence):
        """Calculate optimal laser power based on weed size and confidence"""
        # Larger bounding boxes need more power
        if bbox_area < 0.01:  # Small weed
            base_power = self.laser_power_levels['low']
        elif bbox_area < 0.05:  # Medium weed
            base_power = self.laser_power_levels['medium']
        else:  # Large weed
            base_power = self.laser_power_levels['high']
        
        # Adjust based on confidence
        power_modifier = 0.7 + (confidence * 0.3)  # 0.7 to 1.0
        
        return min(base_power * power_modifier, 1.0)
    
    def calculate_laser_duration(self, bbox_area, confidence):
        """Calculate optimal laser duration"""
        base_duration = 0.1  # Base 100ms
        
        # Larger weeds need longer exposure
        area_modifier = min(bbox_area * 10, 2.0)  # Up to 2x duration
        
        # Higher confidence allows longer exposure
        conf_modifier = 0.5 + confidence * 0.5  # 0.5x to 1x
        
        return base_duration * area_modifier * conf_modifier
    
    def execute_precision_laser(self, laser_command):
        """Execute precision laser targeting"""
        x, y, bbox_area, confidence, weed_class = laser_command
        
        # Calculate optimal laser parameters
        power = self.calculate_laser_power(bbox_area, confidence)
        duration = self.calculate_laser_duration(bbox_area, confidence)
        
        action = {
            'timestamp': time.time(),
            'x': x, 'y': y,
            'bbox_area': bbox_area,
            'confidence': confidence,
            'weed_class': weed_class,
            'laser_power': power,
            'laser_duration': duration,
            'action': 'PRECISION_LASER_FIRED'
        }
        self.laser_actions.append(action)
        
        print(f"🔴 PRECISION LASER: {self.class_names[weed_class]} at ({x},{y}) "
              f"conf={confidence:.2f} power={power:.1f} duration={duration:.2f}s")
        
        # Simulate laser firing
        if self.laser_available:
            time.sleep(duration)
    
    def draw_precision_grid(self, frame):
        """Draw precision targeting grid"""
        h, w = frame.shape[:2]
        
        # 5x5 precision grid
        for i in range(1, 5):
            # Vertical lines
            x = i * w // 5
            cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 1, cv2.LINE_AA)
            # Horizontal lines
            y = i * h // 5
            cv2.line(frame, (0, y), (w, y), (100, 100, 100), 1, cv2.LINE_AA)
        
        # Zone labels (25 zones for precision)
        zones = [f"Z{i+1:02d}" for i in range(25)]
        for i, zone in enumerate(zones):
            row, col = i // 5, i % 5
            x_pos = col * w // 5 + 5
            y_pos = row * h // 5 + 20
            cv2.putText(frame, zone, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame
    
    def calculate_weed_metrics(self, x1, y1, x2, y2, w, h):
        """Calculate precise weed metrics"""
        # Bounding box metrics
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = (bbox_width * bbox_height) / (w * h)  # Normalized area
        
        # Center point for laser targeting
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Aspect ratio (shape analysis)
        aspect_ratio = bbox_width / max(bbox_height, 1)
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'bbox_area': bbox_area,
            'aspect_ratio': aspect_ratio,
            'width': bbox_width,
            'height': bbox_height
        }
    
    def draw_precision_detections(self, frame, results):
        """Draw precise detections with detailed information"""
        h, w = frame.shape[:2]
        frame = self.draw_precision_grid(frame)
        
        detection_counts = {'weeds': 0, 'crops': 0}
        laser_targets = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if conf > self.conf_threshold:
                        # Calculate weed metrics
                        metrics = self.calculate_weed_metrics(x1, y1, x2, y2, w, h)
                        
                        is_weed = cls in self.weed_classes
                        is_crop = cls in self.crop_classes
                        
                        if is_weed:
                            detection_counts['weeds'] += 1
                            color = self.weed_color
                            label_prefix = "WEED"
                            
                            # Queue precision laser action
                            laser_targets.append((
                                metrics['center_x'], metrics['center_y'],
                                metrics['bbox_area'], conf, cls
                            ))
                            
                            # Draw laser crosshair
                            cv2.drawMarker(frame, (metrics['center_x'], metrics['center_y']), 
                                         self.laser_color, cv2.MARKER_CROSS, 15, 2)
                            
                            # Draw targeting circle based on weed size
                            radius = max(5, int(metrics['bbox_area'] * 100))
                            cv2.circle(frame, (metrics['center_x'], metrics['center_y']), 
                                     radius, self.target_color, 2)
                        
                        elif is_crop:
                            detection_counts['crops'] += 1
                            color = self.crop_color
                            label_prefix = "CROP"
                        
                        else:
                            continue
                        
                        # Draw precise bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Enhanced label with metrics
                        class_name = self.class_names[cls]
                        label = f"{label_prefix}: {class_name}"
                        detail = f"Conf:{conf:.2f} Area:{metrics['bbox_area']:.3f}"
                        
                        # Label background
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 25), 
                                    (x1 + max(label_size[0], 200), y1), color, -1)
                        
                        # Draw labels
                        cv2.putText(frame, label, (x1, y1 - 15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(frame, detail, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        # Draw size indicator
                        if is_weed:
                            size_indicator = "●" if metrics['bbox_area'] < 0.01 else "●●" if metrics['bbox_area'] < 0.05 else "●●●"
                            cv2.putText(frame, size_indicator, (x2 - 30, y1 + 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.laser_color, 2)
        
        # Send laser commands
        for target in laser_targets:
            try:
                self.laser_queue.put_nowait(target)
            except queue.Full:
                print("⚠️ Laser queue full!")
        
        # Draw comprehensive statistics
        stats_y = h - 120
        cv2.rectangle(frame, (10, stats_y - 10), (300, h - 10), (0, 0, 0), -1)
        
        cv2.putText(frame, f"Weeds Detected: {detection_counts['weeds']}", 
                   (15, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.weed_color, 2)
        cv2.putText(frame, f"Crops Protected: {detection_counts['crops']}", 
                   (15, stats_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.crop_color, 2)
        cv2.putText(frame, f"Laser Shots: {len(self.laser_actions)}", 
                   (15, stats_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.laser_color, 2)
        cv2.putText(frame, f"Queue: {self.laser_queue.qsize()}", 
                   (15, stats_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def draw_performance_metrics(self, frame):
        """Draw comprehensive performance metrics"""
        if self.fps_history and self.inference_times:
            avg_fps = np.mean(self.fps_history)
            avg_inf = np.mean(self.inference_times) * 1000
            
            # Performance box
            cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
            
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Inference: {avg_inf:.1f}ms", (15, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Model: Precision YOLOv8m", (15, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", 
                       (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run_precision_system(self):
        """Run precision weed detection and laser system"""
        # Open video source
        if str(self.source).isdigit():
            cap = cv2.VideoCapture(int(self.source))
        else:
            cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print(f"❌ Cannot open source: {self.source}")
            return
        
        # Set high resolution for precision
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {w}x{h} @ {fps}FPS")
        if total_frames > 0:
            print(f"Total frames: {total_frames}")
        
        # Video writer
        out = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
            print(f"💾 Recording: {self.output_path}")
        
        print("\n🎯 Starting precision weed laser system...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save frame")
        print("  'p' - Pause/Resume")
        print("  'r' - Reset statistics")
        print("="*60)
        
        frame_count = 0
        start_time = time.time()
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("📹 Video completed")
                    break
                
                # High-precision inference
                inf_start = time.time()
                results = self.model(
                    frame, 
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_detections,
                    device=self.device if hasattr(self, 'device') else 0,
                    verbose=False
                )
                inf_time = time.time() - inf_start
                self.inference_times.append(inf_time)
                
                # Process with precision
                processed_frame = self.draw_precision_detections(frame, results)
                processed_frame = self.draw_performance_metrics(processed_frame)
                
                # FPS calculation
                frame_count += 1
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                self.fps_history.append(current_fps)
                
                # Save video
                if out:
                    out.write(processed_frame)
                
                # Progress indicator
                if total_frames > 0 and frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Weeds eliminated: {len(self.laser_actions)}")
            
            else:
                processed_frame = frame.copy()
                cv2.putText(processed_frame, "⏸️ PAUSED - Press 'p' to resume", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display
            cv2.imshow('🎯 Precision Weed Laser System', processed_frame)
            
            # Control handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"precision_frame_{frame_count:06d}.jpg"
                cv2.imwrite(save_path, processed_frame)
                print(f"💾 Saved: {save_path}")
            elif key == ord('p'):
                paused = not paused
                print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
            elif key == ord('r'):
                self.laser_actions.clear()
                print("🔄 Statistics reset")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Final comprehensive report
        self.generate_final_report(frame_count, time.time() - start_time)
    
    def generate_final_report(self, total_frames, total_time):
        """Generate comprehensive final report"""
        print("\n" + "="*80)
        print("🎯 PRECISION WEED LASER SYSTEM - FINAL REPORT")
        print("="*80)
        
        print(f"📊 Performance Metrics:")
        print(f"  Total frames processed: {total_frames:,}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"  Average FPS: {total_frames/total_time:.2f}")
        
        if self.fps_history:
            print(f"  Peak FPS: {np.max(self.fps_history):.2f}")
        if self.inference_times:
            print(f"  Avg inference: {np.mean(self.inference_times)*1000:.1f}ms")
        
        print(f"\n🔴 Laser System Performance:")
        print(f"  Total laser shots: {len(self.laser_actions):,}")
        print(f"  Shots per minute: {len(self.laser_actions)/(total_time/60):.1f}")
        
        # Analyze laser actions
        if self.laser_actions:
            powers = [a['laser_power'] for a in self.laser_actions]
            durations = [a['laser_duration'] for a in self.laser_actions]
            
            print(f"  Avg laser power: {np.mean(powers):.2f}")
            print(f"  Avg laser duration: {np.mean(durations):.3f}s")
            print(f"  Total laser time: {np.sum(durations):.1f}s")
        
        print(f"\n📈 Detection Analysis:")
        weed_sizes = {'small': 0, 'medium': 0, 'large': 0}
        for action in self.laser_actions:
            area = action['bbox_area']
            if area < 0.01:
                weed_sizes['small'] += 1
            elif area < 0.05:
                weed_sizes['medium'] += 1
            else:
                weed_sizes['large'] += 1
        
        total_weeds = sum(weed_sizes.values())
        if total_weeds > 0:
            print(f"  Small weeds: {weed_sizes['small']} ({weed_sizes['small']/total_weeds*100:.1f}%)")
            print(f"  Medium weeds: {weed_sizes['medium']} ({weed_sizes['medium']/total_weeds*100:.1f}%)")
            print(f"  Large weeds: {weed_sizes['large']} ({weed_sizes['large']/total_weeds*100:.1f}%)")
        
        if self.output_path and Path(self.output_path).exists():
            print(f"\n💾 Output saved: {self.output_path}")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Precision Weed Laser Detection System')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to precision trained model')
    parser.add_argument('--config', type=str, default='config/precision_weed_config.yaml',
                       help='Config file path')
    parser.add_argument('--source', type=str, required=True,
                       help='Video source (0 for webcam or path to video)')
    parser.add_argument('--output', type=str, 
                       help='Output video path (optional)')
    parser.add_argument('--conf', type=float, 
                       help='Confidence threshold override')
    parser.add_argument('--iou', type=float,
                       help='IoU threshold override')
    
    args = parser.parse_args()
    
    system = PrecisionLaserWeedSystem(
        model_path=args.model,
        config_path=args.config,
        source=args.source,
        output_path=args.output
    )
    
    if args.conf:
        system.conf_threshold = args.conf
        print(f"Using confidence threshold: {args.conf}")
    
    if args.iou:
        system.iou_threshold = args.iou
        print(f"Using IoU threshold: {args.iou}")
    
    system.run_precision_system()

if __name__ == "__main__":
    main()

