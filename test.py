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

class FixedWeedLaserDetector:
    def __init__(self, model_path, source, output_path=None, confidence=0.25, iou=0.5):
        # Load trained model
        self.model = YOLO(model_path)
        self.source = source
        self.output_path = output_path
        self.confidence = confidence
        self.iou = iou
        
        # Detection classes
        self.class_names = {0: 'crop', 1: 'weed'}
        self.weed_classes = [1]
        self.crop_classes = [0]
        
        # Laser system
        self.laser_queue = queue.Queue()
        self.laser_actions = []
        self.setup_laser_system()
        
        # Performance tracking
        self.fps_history = deque(maxlen=60)
        self.inference_times = deque(maxlen=60)
        
        # Colors
        self.colors = {
            'weed': (0, 0, 255),      # Red for weeds
            'crop': (0, 255, 0),      # Green for crops
            'laser': (0, 255, 255),   # Yellow for laser
            'crosshair': (255, 0, 255) # Magenta for crosshair
        }
        
        print(f"🎯 Fixed Weed Laser Detector Ready!")
        print(f"Model: {model_path}")
        print(f"Confidence: {confidence}")
    
    def setup_laser_system(self):
        """Initialize laser system"""
        self.laser_thread = threading.Thread(target=self.laser_worker, daemon=True)
        self.laser_thread.start()
        print("🔴 Laser system ready")
    
    def laser_worker(self):
        """Laser control worker"""
        while True:
            try:
                laser_command = self.laser_queue.get(timeout=1)
                self.execute_laser(laser_command)
                self.laser_queue.task_done()
            except queue.Empty:
                continue
    
    def execute_laser(self, laser_command):
        """Execute laser action"""
        x, y, confidence = laser_command
        
        action = {
            'timestamp': time.time(),
            'x': x, 'y': y,
            'confidence': confidence,
            'duration': 0.1 + confidence * 0.2
        }
        self.laser_actions.append(action)
        
        print(f"🔴 LASER: Weed at ({x},{y}) conf={confidence:.2f}")
        time.sleep(action['duration'])
    
    def process_frame(self, frame):
        """Process single frame with detections"""
        if frame is None or frame.size == 0:
            return frame
        
        h, w = frame.shape[:2]
        
        # Run inference
        inference_start = time.time()
        results = self.model(frame, conf=self.confidence, iou=self.iou, verbose=False)
        inference_time = time.time() - inference_start
        self.inference_times.append(inference_time)
        
        # Draw targeting grid
        frame = self.draw_grid(frame)
        
        # Process detections
        frame_weeds = 0
        frame_crops = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if conf > self.confidence:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        if cls in self.weed_classes:
                            frame_weeds += 1
                            color = self.colors['weed']
                            label = f"WEED: {conf:.2f}"
                            
                            # Queue laser action
                            try:
                                self.laser_queue.put_nowait((center_x, center_y, conf))
                            except queue.Full:
                                pass
                            
                            # Draw laser crosshair
                            cv2.drawMarker(frame, (center_x, center_y), 
                                         self.colors['laser'], cv2.MARKER_CROSS, 25, 3)
                            
                            # Draw targeting circle
                            cv2.circle(frame, (center_x, center_y), 15, 
                                     self.colors['crosshair'], 2)
                        
                        elif cls in self.crop_classes:
                            frame_crops += 1
                            color = self.colors['crop']
                            label = f"CROP: {conf:.2f} [PROTECTED]"
                        else:
                            continue
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw statistics
        self.draw_stats(frame, frame_weeds, frame_crops)
        
        return frame
    
    def draw_grid(self, frame):
        """Draw targeting grid"""
        h, w = frame.shape[:2]
        
        for i in range(1, 5):
            # Vertical lines
            x = i * w // 5
            cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 1)
            # Horizontal lines
            y = i * h // 5
            cv2.line(frame, (0, y), (w, y), (100, 100, 100), 1)
        
        return frame
    
    def draw_stats(self, frame, frame_weeds, frame_crops):
        """Draw statistics panel"""
        h, w = frame.shape[:2]
        
        # Background
        cv2.rectangle(frame, (10, h - 120), (350, h - 10), (0, 0, 0), -1)
        
        # Statistics
        y_pos = h - 100
        cv2.putText(frame, f"Weeds: {frame_weeds} | Crops: {frame_crops}", 
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25
        cv2.putText(frame, f"Total Laser Shots: {len(self.laser_actions)}", 
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['laser'], 2)
        y_pos += 25
        cv2.putText(frame, f"Queue: {self.laser_queue.qsize()}", 
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Performance info
        if self.inference_times:
            avg_inference = np.mean(self.inference_times) * 1000
            cv2.putText(frame, f"Inference: {avg_inference:.1f}ms", 
                       (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Conf: {self.confidence:.2f}", 
                   (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run_detection(self):
        """Main detection loop with proper video handling"""
        
        print(f"🔍 Opening video source: {self.source}")
        
        # Open video with explicit backend
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            print("❌ Failed with FFMPEG, trying default...")
            cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {self.source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video loaded successfully:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        print(f"   Duration: {total_frames/fps:.1f}s")
        
        # Setup video writer
        out = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            print(f"💾 Recording to: {self.output_path}")
        
        # Set up window
        cv2.namedWindow('Weed Laser Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Weed Laser Detection', 1280, 720)
        
        print("\n🚀 Starting detection...")
        print("Controls: 'q' - Quit, 's' - Save frame, SPACE - Pause")
        print("="*60)
        
        frame_count = 0
        start_time = time.time()
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("📹 Video completed or failed to read frame")
                    break
                
                if frame is None or frame.size == 0:
                    print(f"⚠️ Empty frame at {frame_count}")
                    continue
                
                # Process frame
                processed_frame = self.process_frame(frame.copy())
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                self.fps_history.append(current_fps)
                
                # Add FPS display
                cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", 
                           (width - 150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save to output
                if out:
                    out.write(processed_frame)
                
                # Progress update
                if frame_count % 60 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"Progress: {progress:.1f}% | Frame: {frame_count} | "
                          f"Laser shots: {len(self.laser_actions)}")
                
                # Display frame
                cv2.imshow('Weed Laser Detection', processed_frame)
            
            else:
                # Paused - show last frame with overlay
                pause_frame = processed_frame.copy()
                cv2.putText(pause_frame, "PAUSED - Press SPACE to resume", 
                           (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow('Weed Laser Detection', pause_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quitting...")
                break
            elif key == ord('s'):
                save_path = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(save_path, processed_frame)
                print(f"💾 Saved: {save_path}")
            elif key == ord(' '):  # Space bar
                paused = not paused
                print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
            elif key == ord('+'):
                self.confidence = min(0.9, self.confidence + 0.05)
                print(f"📈 Confidence: {self.confidence:.2f}")
            elif key == ord('-'):
                self.confidence = max(0.05, self.confidence - 0.05)
                print(f"📉 Confidence: {self.confidence:.2f}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Final report
        total_time = time.time() - start_time
        self.generate_report(frame_count, total_time)
    
    def generate_report(self, frame_count, total_time):
        """Generate final report"""
        print("\n" + "="*70)
        print("🏁 DETECTION COMPLETED - FINAL REPORT")
        print("="*70)
        
        print(f"📊 Performance:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average FPS: {frame_count/total_time:.1f}")
        
        print(f"\n🔴 Laser Actions:")
        print(f"   Total shots: {len(self.laser_actions)}")
        print(f"   Shots per minute: {len(self.laser_actions)/(total_time/60):.1f}")
        
        if self.laser_actions:
            avg_conf = np.mean([a['confidence'] for a in self.laser_actions])
            print(f"   Average confidence: {avg_conf:.3f}")
        
        print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Fixed Weed Detection System')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return
    
    # Check video exists
    if not str(args.source).isdigit() and not Path(args.source).exists():
        print(f"❌ Video not found: {args.source}")
        return
    
    # Run detector
    detector = FixedWeedLaserDetector(
        model_path=args.model,
        source=args.source,
        output_path=args.output,
        confidence=args.conf,
        iou=args.iou
    )
    
    detector.run_detection()

if __name__ == "__main__":
    main()

