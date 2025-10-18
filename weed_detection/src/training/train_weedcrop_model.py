import os
import yaml
import torch
import gc
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import shutil
import logging

class MemoryOptimizedWeedCropTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.setup_memory_optimization()
        self.device = self.setup_device()
        self.model_save_dir = Path("models")
        self.model_save_dir.mkdir(exist_ok=True)
        
        print("🚀 Memory-Optimized WeedCrop Trainer Ready")
    
    def setup_memory_optimization(self):
        """Aggressive memory optimization"""
        # Set environment variables
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        print("🔧 Memory optimization applied")
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('memory_optimized_weedcrop.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_device(self):
        if torch.cuda.is_available():
            device = f"cuda:{self.config['training']['device']}"
            
            # Detailed memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            free_memory = total_memory - reserved
            
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"Total: {total_memory:.1f}GB | Free: {free_memory:.1f}GB")
            self.logger.info(f"Allocated: {allocated:.1f}GB | Reserved: {reserved:.1f}GB")
            
            if free_memory < 2.0:
                self.logger.warning("⚠️  Low GPU memory available!")
        else:
            device = "cpu"
        return device
    
    def train_memory_optimized(self, data_yaml_path):
        """Ultra memory-efficient training"""
        self.logger.info("🚀 Starting ultra memory-optimized training...")
        
        # Use smallest model
        model_name = self.config['model']['architecture']
        model = YOLO(f"{model_name}.pt")
        
        # Ultra-conservative training parameters
        train_args = {
            'data': data_yaml_path,
            'epochs': self.config['training']['epochs'],
            'batch': self.config['training']['batch'],  # Very small batch (2-4)
            'imgsz': self.config['training']['image_size'],  # Reduced resolution (416)
            'lr0': self.config['training']['learning_rate'],
            'lrf': 0.01,
            'weight_decay': self.config['training']['weight_decay'],
            'momentum': self.config['training']['momentum'],
            'device': self.device,
            'workers': self.config['training']['workers'],  # Reduced workers
            'patience': self.config['training']['patience'],
            'save_period': self.config['training']['save_period'],
            'project': 'runs/detect',
            'name': f'memory_opt_weedcrop_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': self.config['model']['pretrained'],
            'optimizer': self.config['optimization']['optimizer'],
            'close_mosaic': self.config['optimization']['close_mosaic'],
            'amp': self.config['optimization']['amp'],  # Mixed precision
            
            # Memory-saving settings
            'cache': False,  # Don't cache images
            'rect': False,   # No rectangular training
            'multi_scale': False,  # Disable multi-scale
            'overlap_mask': True,
            'val': True,
            'plots': True,
            'verbose': True,
            'save': True,
            'save_txt': True,
            'save_conf': True,
            
            # Detection thresholds
            'conf': 0.25,
            'iou': 0.5,
            'max_det': 100,
        }
        
        # Conservative augmentation
        train_args.update({
            'hsv_h': 0.01, 'hsv_s': 0.3, 'hsv_v': 0.3,
            'degrees': 10, 'translate': 0.05, 'scale': 0.3,
            'shear': 0, 'perspective': 0, 'flipud': 0, 'fliplr': 0.5,
            'mosaic': 0.8, 'mixup': 0
        })
        
        self.logger.info("🎯 Ultra-Conservative Settings:")
        self.logger.info(f"  Model: {model_name} (smallest)")
        self.logger.info(f"  Batch: {self.config['training']['batch']} (very small)")
        self.logger.info(f"  Image size: {self.config['training']['image_size']} (reduced)")
        self.logger.info(f"  Workers: {self.config['training']['workers']} (reduced)")
        self.logger.info(f"  Cache: False (memory saving)")
        
        try:
            # Final memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            # Monitor memory usage
            if torch.cuda.is_available():
                before_training = torch.cuda.memory_allocated(0) / 1e9
                self.logger.info(f"Memory before training: {before_training:.1f}GB")
            
            results = model.train(**train_args)
            best_model_path = self.save_model()
            return results, best_model_path
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error("🔴 CUDA OOM Error - Try even smaller settings:")
            self.logger.error("  - Reduce batch to 1")
            self.logger.error("  - Reduce image size to 320")
            self.logger.error("  - Close all other applications")
            raise
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self):
        """Save trained model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"memory_opt_weedcrop_{timestamp}.pt"
        model_path = self.model_save_dir / model_name
        
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            latest_run = max(runs_dir.iterdir(), key=os.path.getctime)
            weights_path = latest_run / "weights" / "best.pt"
            
            if weights_path.exists():
                shutil.copy2(weights_path, model_path)
                self.logger.info(f"✅ Model saved: {model_path}")
                return str(model_path)
        return None
    
    def validate_model(self, model_path, data_yaml_path):
        """Memory-efficient validation"""
        if not model_path:
            return None
            
        self.logger.info("🔍 Validating model...")
        
        torch.cuda.empty_cache()
        model = YOLO(model_path)
        
        try:
            results = model.val(
                data=data_yaml_path,
                device=self.device,
                conf=0.25,
                iou=0.5,
                batch=2,  # Small validation batch
                imgsz=self.config['training']['image_size'],
                plots=True,
                verbose=True
            )
            
            self.logger.info(f"🏆 Validation Results:")
            self.logger.info(f"  mAP50: {results.box.map50:.3f}")
            self.logger.info(f"  mAP50-95: {results.box.map:.3f}")
            self.logger.info(f"  Precision: {results.box.mp:.3f}")
            self.logger.info(f"  Recall: {results.box.mr:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Memory-Optimized WeedCrop Training')
    parser.add_argument('--config', type=str, default='config/memory_optimized_weedcrop.yaml')
    parser.add_argument('--data', type=str, required=True)
    
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        print(f"❌ Data YAML not found: {args.data}")
        return
    
    # Ultimate memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    trainer = MemoryOptimizedWeedCropTrainer(args.config)
    results, model_path = trainer.train_memory_optimized(args.data)
    
    if model_path:
        validation_results = trainer.validate_model(model_path, args.data)
        
        print("\n" + "="*60)
        print("🏆 MEMORY-OPTIMIZED TRAINING COMPLETED!")
        print("="*60)
        print(f"✅ Model: {model_path}")
        
        if validation_results:
            print(f"📊 Performance:")
            print(f"  mAP50: {validation_results.box.map50:.3f}")
            print(f"  Precision: {validation_results.box.mp:.3f}")
            print(f"  Recall: {validation_results.box.mr:.3f}")
        
        print("="*60)
        print(f"🚀 Test: python src/inference/weedcrop_laser_system.py --model {model_path} --source 0")
    else:
        print("❌ Training failed - model not saved")

if __name__ == "__main__":
    main()

