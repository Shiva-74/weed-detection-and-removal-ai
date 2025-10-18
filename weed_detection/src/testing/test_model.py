import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
import time

class WeedModelTester:
    def __init__(self, model_path):
        """Initialize the weed model tester"""
        self.model = YOLO(model_path)
        self.model_path = model_path
        print(f"🌿 Loaded model: {model_path}")
        
        # Get class names from model
        self.class_names = self.model.names
        print(f"📋 Classes: {self.class_names}")
    
    def test_single_image(self, image_path, conf=0.25, save_results=True):
        """Test model on a single image"""
        print(f"🔍 Testing image: {image_path}")
        
        # Run inference
        results = self.model(image_path, conf=conf)
        
        # Process results
        for r in results:
            # Get image
            img = r.orig_img
            
            # Print detection info
            if r.boxes is not None:
                boxes = r.boxes
                print(f"📊 Detections found: {len(boxes)}")
                
                for i, box in enumerate(boxes):
                    # Get box info
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    print(f"  Detection {i+1}: {class_name} ({confidence:.3f}) at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
            else:
                print("📊 No detections found")
            
            # Show results
            annotated_img = r.plot()
            cv2.imshow(f'Weed Detection Results - {Path(image_path).name}', annotated_img)
            
            # Save results if requested
            if save_results:
                output_path = f"test_result_{Path(image_path).stem}.jpg"
                cv2.imwrite(output_path, annotated_img)
                print(f"💾 Results saved: {output_path}")
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results
    
    def test_webcam(self, conf=0.25, source=0):
        """Test model on live webcam feed"""
        print(f"📹 Starting webcam test (source: {source})")
        print("Press 'q' to quit, 's' to save frame")
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"❌ Cannot open webcam {source}")
            return
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = self.model(frame, conf=conf)
            
            # Process results
            for r in results:
                # Get annotated frame
                annotated_frame = r.plot()
                
                # Add info overlay
                cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Conf: {conf}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Count detections
                detection_count = len(r.boxes) if r.boxes is not None else 0
                cv2.putText(annotated_frame, f"Detections: {detection_count}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Show frame
                cv2.imshow('Live Weed Detection', annotated_frame)
            
            frame_count += 1
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"webcam_frame_{frame_count:06d}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                print(f"💾 Saved frame: {save_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("📹 Webcam test completed")
    
    def test_video(self, video_path, conf=0.25, save_video=False):
        """Test model on video file"""
        print(f"🎬 Testing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📋 Video info: {width}x{height} @ {fps}FPS, {total_frames} frames")
        
        # Video writer for saving
        out = None
        if save_video:
            output_path = f"test_output_{Path(video_path).stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Will save to: {output_path}")
        
        frame_count = 0
        detection_counts = []
        
        print("🎬 Processing video... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = self.model(frame, conf=conf)
            
            # Process results
            for r in results:
                annotated_frame = r.plot()
                
                # Count detections
                detection_count = len(r.boxes) if r.boxes is not None else 0
                detection_counts.append(detection_count)
                
                # Add progress info
                progress = (frame_count / total_frames) * 100
                cv2.putText(annotated_frame, f"Progress: {progress:.1f}%", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Detections: {detection_count}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Save frame
                if out:
                    out.write(annotated_frame)
                
                # Show frame
                cv2.imshow('Video Weed Detection', annotated_frame)
            
            frame_count += 1
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print(f"\n📊 Video Test Summary:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Total detections: {sum(detection_counts)}")
        print(f"  Average detections per frame: {np.mean(detection_counts):.2f}")
        print(f"  Max detections in frame: {max(detection_counts) if detection_counts else 0}")
    
    def test_batch_images(self, images_dir, conf=0.25, save_results=True):
        """Test model on batch of images"""
        images_dir = Path(images_dir)
        print(f"📂 Testing batch of images from: {images_dir}")
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(ext)))
            image_files.extend(list(images_dir.glob(ext.upper())))
        
        print(f"📋 Found {len(image_files)} images")
        
        if not image_files:
            print("❌ No images found!")
            return
        
        # Process batch
        all_detections = []
        
        for i, img_path in enumerate(image_files):
            print(f"🔍 Processing {i+1}/{len(image_files)}: {img_path.name}")
            
            # Run inference
            results = self.model(str(img_path), conf=conf)
            
            for r in results:
                detection_count = len(r.boxes) if r.boxes is not None else 0
                all_detections.append(detection_count)
                
                # Save annotated image
                if save_results:
                    annotated_img = r.plot()
                    output_path = f"batch_result_{img_path.stem}.jpg"
                    cv2.imwrite(output_path, annotated_img)
                    print(f"  💾 Saved: {output_path}")
                
                print(f"  📊 Detections: {detection_count}")
        
        # Print batch summary
        print(f"\n📊 Batch Test Summary:")
        print(f"  Images processed: {len(image_files)}")
        print(f"  Total detections: {sum(all_detections)}")
        print(f"  Average detections per image: {np.mean(all_detections):.2f}")
        print(f"  Images with detections: {sum(1 for x in all_detections if x > 0)}")
    
    def benchmark_speed(self, test_image_path, iterations=100):
        """Benchmark model inference speed"""
        print(f"⚡ Benchmarking speed with {iterations} iterations...")
        
        # Load test image
        img = cv2.imread(test_image_path)
        if img is None:
            print(f"❌ Cannot load test image: {test_image_path}")
            return
        
        # Warmup
        print("🔥 Warming up...")
        for _ in range(10):
            _ = self.model(img, verbose=False)
        
        # Benchmark
        print("⏱️ Running benchmark...")
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            results = self.model(img, verbose=False)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{iterations} iterations")
        
        # Calculate statistics
        avg_time = np.mean(times) * 1000  # Convert to ms
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        fps = 1.0 / np.mean(times)
        
        print(f"\n⚡ Speed Benchmark Results:")
        print(f"  Average inference time: {avg_time:.2f} ms")
        print(f"  Min inference time: {min_time:.2f} ms")
        print(f"  Max inference time: {max_time:.2f} ms")
        print(f"  Average FPS: {fps:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Test Weed Detection Model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--mode', type=str, choices=['image', 'webcam', 'video', 'batch', 'benchmark'], 
                       default='image', help='Testing mode')
    parser.add_argument('--source', type=str, help='Path to image/video/directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save results')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return
    
    # Initialize tester
    tester = WeedModelTester(args.model)
    
    # Run test based on mode
    if args.mode == 'image':
        if not args.source:
            print("❌ Please provide --source for image path")
            return
        tester.test_single_image(args.source, conf=args.conf, save_results=args.save)
    
    elif args.mode == 'webcam':
        source = int(args.source) if args.source else 0
        tester.test_webcam(conf=args.conf, source=source)
    
    elif args.mode == 'video':
        if not args.source:
            print("❌ Please provide --source for video path")
            return
        tester.test_video(args.source, conf=args.conf, save_video=args.save)
    
    elif args.mode == 'batch':
        if not args.source:
            print("❌ Please provide --source for images directory")
            return
        tester.test_batch_images(args.source, conf=args.conf, save_results=args.save)
    
    elif args.mode == 'benchmark':
        if not args.source:
            print("❌ Please provide --source for test image")
            return
        tester.benchmark_speed(args.source)

if __name__ == "__main__":
    main()

