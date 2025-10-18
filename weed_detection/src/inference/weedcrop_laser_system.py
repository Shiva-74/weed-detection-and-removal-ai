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

class WeedCropLaserSystem:
    def __init__(self, model_path, config_path, source, output_path=None):
        self.model = YOLO(model_path)
        self.source = source
        self.output_path = output_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.conf_threshold = self.config['deployment']['confidence']
        self.iou_threshold = self.config['deployment']['iou_threshold']
        self.max_detections = self.config['deployment']['max_detections']
        
        # Class configuration
        self.weed_classes = set(self.config.get('weed_classes', [1]))
        self.crop_classes = set(self.config.get('crop_classes', [0]))
        
        # Laser system
        self.laser_queue = queue.Queue()
        self.laser_actions = []
        self.setup_laser_system()
        
        # Performance tracking
        self.fps_history = deque(maxlen=60)
        self.inference_times = deque(maxlen=60)
        
        # Colors
        self.weed_color = (0, 0, 255)      # Red for weeds
        self.crop_color = (0, 255, 0)      # Green for crops
        self.laser_color = (0, 255, 255)   # Yellow for laser
        
        print("🌿 WeedCrop Laser System Ready")
        print(f"Model: {model_path}")
        print(f"Confidence: {self.conf_threshold}")
    
    def setup_laser_system(self):
        """Setup laser control system"""
        self.laser_available = True
        self.laser_thread = threading.Thread(target=self.laser_worker, daemon=True)
        self.laser_thread.start()
        print("🔴 Laser system initialized")
    
    def laser_worker(self):
        """Laser control worker thread"""
        while True:
            try:
                laser_command = self.laser_queue.get(timeout=1)
                self.execute_laser(laser_command)
                self.laser_queue.task_done()
            except queue.Empty:
                continue
    
    def execute_laser(self, laser_command):
        """Execute laser targeting"""
        x, y, confidence, weed_class = laser_command
        
        action = {
            'timestamp': time.time(),
            'x': x, 'y': y,
            'confidence': confidence,
            'weed_class': weed_class,
            'action': 'LASER_FIRED'
        }
        self.laser_actions.append(action)
        
        print(f"🔴 LASER: Weed at ({x},{y}) conf={confidence:.2f}")
        
        # Simulate laser duration based on confidence
        duration = max(0.1, confidence * 0.3)
        time.sleep(duration)
    
    def draw_targeting_grid(self, frame):
        """Draw precision targeting grid"""
        h, w = frame.shape[:2]
        
        # 4x4 grid for targeting zones
        for i in range(1, 4):
            # Vertical lines
            x = i * w // 4
            cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 1)
            # Horizontal lines
            y = i * h // 4
            cv2.line(frame, (0, y), (w, y), (100, 100, 100), 1)
        
        # Zone labels
        zones = [f"Z{i+1:02d}" for i in range(16)]
        for i, zone in enumerate(zones):
            row, col = i // 4, i % 4
            x_pos = col * w // 4 + 5
            y_pos = row * h // 4 + 20
            cv2.putText(frame, zone, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame
    
    def process_detections(self, frame, results):
        """Process detections and trigger laser actions"""
        h, w = frame.shape[:2]
        frame = self.draw_targeting_grid(frame)
        
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
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        is_weed = cls in self.weed_classes
                        is_crop = cls in self.crop_classes
                        
                        if is_weed:
                            detection_counts['weeds'] += 1
                            color = self.weed_color
                            label = f"WEED: {conf:.2f}"
                            
                            # Queue laser action
                            laser_targets.append((center_x, center_y, conf, cls))
                            
                            # Draw laser crosshair
                            cv2.drawMarker(frame, (center_x, center_y), 
                                         self.laser_color, cv2.MARKER_CROSS, 20, 3)
                            
                        elif is_crop:
                            detection_counts['crops'] += 1
                            color = self.crop_color
                            label = f"CROP: {conf:.2f} [PROTECTED]"
                        
                        else:
                            continue
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Send laser commands
        for target in laser_targets:
            try:
                self.laser_queue.put_nowait(target)
            except queue.Full:
                print("⚠️ Laser queue full!")
        
        # Draw statistics
        stats_y = h - 80
        cv2.rectangle(frame, (10, stats_y - 10), (350, h - 10), (0, 0, 0), -1)
        
        cv2.putText(frame, f"Weeds Detected: {detection_counts['weeds']}", 
                   (15, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.weed_color, 2)
        cv2.putText(frame, f"Crops Protected: {detection_counts['crops']}", 
                   (15, stats_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.crop_color, 2)
        cv2.putText(frame, f"Laser Shots: {len(self.laser_actions)}", 
                   (15, stats_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.laser_color, 2)
        
        return frame
    
    def draw_performance_info(self, frame):
        """Draw performance metrics"""
        if self.fps_history and self.inference_times:
            avg_fps = np.mean(self.fps_history)
            avg_inf = np.mean(self.inference_times) * 1000
            
            cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
            
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Inference: {avg_inf:.1f}ms", (15, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"WeedCrop YOLO System", (15, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run_system(self):
        """Run the weed laser detection system"""
        # Open video source
        if str(self.source).isdigit():
            cap = cv2.VideoCapture(int(self.source))
        else:
            cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print(f"❌ Cannot open source: {self.source}")
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {w}x{h} @ {fps}FPS")
        if total_frames > 0:
            print(f"Duration: {total_frames/fps:.1f}s")
        
        # Video writer for output
        out = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
            print(f"💾 Recording: {self.output_path}")
        
        print("\n🚀 Starting WeedCrop laser system...")
        print("Controls: 'q' - Quit, 's' - Save frame, 'p' - Pause")
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
                
                # Run inference
                inf_start = time.time()
                results = self.model(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_detections,
                    verbose=False
                )
                inf_time = time.time() - inf_start
                self.inference_times.append(inf_time)
                
                # Process detections
                processed_frame = self.process_detections(frame, results)
                processed_frame = self.draw_performance_info(processed_frame)
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                self.fps_history.append(current_fps)
                
                # Save video
                if out:
                    out.write(processed_frame)
                
                # Progress
                if total_frames > 0 and frame_count % 120 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Weeds eliminated: {len(self.laser_actions)}")
            
            else:
                processed_frame = frame.copy()
                cv2.putText(processed_frame, "⏸️ PAUSED - Press 'p' to resume", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display
            cv2.imshow('🌿 WeedCrop Laser Detection System', processed_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"weedcrop_frame_{frame_count:06d}.jpg"
                cv2.imwrite(save_path, processed_frame)
                print(f"💾 Saved: {save_path}")
            elif key == ord('p'):
                paused = not paused
                print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Final report
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("🏁 WEEDCROP LASER SYSTEM COMPLETED")
        print("="*60)
        print(f"Frames processed: {frame_count:,}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average FPS: {frame_count/total_time:.1f}")
        print(f"Total weeds eliminated: {len(self.laser_actions):,}")
        if self.output_path:
            print(f"Output saved: {self.output_path}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='WeedCrop Laser Detection System')
    parser.add_argument('--model', type=str, required=True, help='Trained model path')
    parser.add_argument('--config', type=str, default='config/weedcrop_config.yaml')
    parser.add_argument('--source', type=str, required=True, help='Video source')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--conf', type=float, help='Confidence threshold override')
    
    args = parser.parse_args()
    
    system = WeedCropLaserSystem(
        model_path=args.model,
        config_path=args.config,
        source=args.source,
        output_path=args.output
    )
    
    if args.conf:
        system.conf_threshold = args.conf
        print(f"Using confidence threshold: {args.conf}")
    
    system.run_system()

if __name__ == "__main__":
    main()

