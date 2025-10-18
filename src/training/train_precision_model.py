import os
import yaml
import torch
import gc
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import shutil
import logging

class MemoryOptimizedWeedTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.device = self.setup_device()
        self.model_save_dir = Path("models")
        self.model_save_dir.mkdir(exist_ok=True)
        
        # Memory optimization
        self.optimize_memory()
        
        print("🎯 Memory-Optimized Precision Weed Trainer Initialized")
    
    def optimize_memory(self):
        """Optimize memory usage"""
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Set memory optimization environment variables
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        print("🔧 Memory optimization applied")
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('memory_optimized_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_device(self):
        if torch.cuda.is_available():
            device = f"cuda:{self.config['training']['device']}"
            
            # Check available memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated_memory = torch.cuda.memory_allocated(0) / 1e9
            reserved_memory = torch.cuda.memory_reserved(0) / 1e9
            
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"Total GPU Memory: {total_memory:.1f} GB")
            self.logger.info(f"Allocated: {allocated_memory:.2f} GB")
            self.logger.info(f"Reserved: {reserved_memory:.2f} GB")
            self.logger.info(f"Free: {total_memory - reserved_memory:.2f} GB")
        else:
            device = "cpu"
            self.logger.info("Using CPU for training")
        return device
    
    def train_memory_optimized_model(self, dataset_yaml_path):
        """Train model with aggressive memory optimization"""
        self.logger.info("Starting memory-optimized precision weed training...")
        
        # Use lightweight model
        model_name = self.config['model']['architecture']
        model = YOLO(f"{model_name}.pt")
        
        # Memory-optimized training arguments
        train_args = {
            'data': dataset_yaml_path,
            'epochs': self.config['training']['epochs'],  # 100
            'batch': self.config['training']['batch_size'],  # 8
            'imgsz': self.config['training']['image_size'],  # 640
            'lr0': self.config['training']['learning_rate'],
            'lrf': 0.01,
            'weight_decay': self.config['training']['weight_decay'],
            'momentum': self.config['training']['momentum'],
            'device': self.device,
            'workers': self.config['training']['workers'],  # 4
            'patience': self.config['training']['patience'],  # 20
            'save_period': self.config['training']['save_period'],
            'project': 'runs/detect',
            'name': f'memory_opt_weed_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': self.config['model']['pretrained'],
            'optimizer': self.config['optimization']['optimizer'],
            'close_mosaic': self.config['optimization']['close_mosaic'],
            'amp': self.config['optimization']['amp'],  # Mixed precision
            
            # High confidence threshold (45%)
            'conf': self.config['deployment']['confidence'],  # 0.45
            'iou': self.config['deployment']['iou_threshold'],  # 0.3
            'max_det': self.config['deployment']['max_detections'],  # 100
            
            # Memory optimization settings
            'cache': False,  # Don't cache to save memory
            'rect': False,  # Disable rectangular training
            'single_cls': False,
            'multi_scale': False,  # Disable multi-scale to save memory
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'verbose': True,
            'save': True,
            'save_txt': True,
            'save_conf': True,
        }
        
        # Conservative augmentation for memory
        aug = self.config['augmentation']
        train_args.update({
            'hsv_h': aug['hsv_h'], 'hsv_s': aug['hsv_s'], 'hsv_v': aug['hsv_v'],
            'degrees': aug['degrees'], 'translate': aug['translate'], 
            'scale': aug['scale'], 'shear': aug['shear'],
            'perspective': aug['perspective'], 'flipud': aug['flipud'], 
            'fliplr': aug['fliplr'], 'mosaic': aug['mosaic'], 'mixup': aug['mixup']
        })
        
        self.logger.info(f"Memory-Optimized Training Configuration:")
        self.logger.info(f"  Model: {model_name}")
        self.logger.info(f"  Resolution: {self.config['training']['image_size']}px")
        self.logger.info(f"  Batch size: {self.config['training']['batch_size']}")
        self.logger.info(f"  Epochs: {self.config['training']['epochs']}")
        self.logger.info(f"  Confidence threshold: {self.config['deployment']['confidence']} (45%)")
        self.logger.info(f"  Workers: {self.config['training']['workers']}")
        
        try:
            # Clear memory before training
            torch.cuda.empty_cache()
            gc.collect()
            
            results = model.train(**train_args)
            best_model_path = self.save_model()
            return results, best_model_path
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"CUDA OOM Error: {e}")
            self.logger.error("Try reducing batch size further (2-4) or image size (512)")
            raise
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self):
        """Save trained model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"memory_opt_weed_detector_{timestamp}.pt"
        model_path = self.model_save_dir / model_name
        
        # Find latest run
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            latest_run = max(runs_dir.iterdir(), key=os.path.getctime)
            weights_path = latest_run / "weights" / "best.pt"
            
            if weights_path.exists():
                shutil.copy2(weights_path, model_path)
                self.logger.info(f"Model saved: {model_path}")
                return str(model_path)
        
        return None
    
    def validate_model(self, model_path, dataset_yaml_path):
        """Validate model with memory optimization"""
        if not model_path or not Path(model_path).exists():
            self.logger.error("Model path not found")
            return None
            
        self.logger.info("Validating model...")
        
        # Clear memory before validation
        torch.cuda.empty_cache()
        gc.collect()
        
        model = YOLO(model_path)
        
        validation_results = model.val(
            data=dataset_yaml_path,
            device=self.device,
            conf=self.config['deployment']['confidence'],  # 0.45
            iou=self.config['deployment']['iou_threshold'],
            batch=4,  # Smaller batch for validation
            imgsz=self.config['training']['image_size'],
            plots=True,
            verbose=True
        )
        
        # Log results
        self.logger.info(f"Validation Results (Conf ≥ 45%):")
        self.logger.info(f"  mAP50: {validation_results.box.map50:.4f}")
        self.logger.info(f"  mAP50-95: {validation_results.box.map:.4f}")
        self.logger.info(f"  Precision: {validation_results.box.mp:.4f}")
        self.logger.info(f"  Recall: {validation_results.box.mr:.4f}")
        
        return validation_results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Memory-Optimized Precision Weed Training')
    parser.add_argument('--config', type=str, default='config/memory_optimized_config.yaml')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML path')
    
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        print(f"❌ Dataset YAML not found: {args.data}")
        return
    
    # Free up GPU memory first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    trainer = MemoryOptimizedWeedTrainer(args.config)
    results, model_path = trainer.train_memory_optimized_model(args.data)
    
    if model_path:
        validation_results = trainer.validate_model(model_path, args.data)
        
        print("\n🎯 MEMORY-OPTIMIZED TRAINING COMPLETED!")
        print(f"✅ Model: {model_path}")
        
        if validation_results:
            print(f"📊 Performance (Confidence ≥ 45%):")
            print(f"  mAP50: {validation_results.box.map50:.3f}")
            print(f"  Precision: {validation_results.box.mp:.3f}")
            print(f"  Recall: {validation_results.box.mr:.3f}")
        
        print(f"\n🚀 Test your model:")
        print(f"python src/inference/precision_laser_system.py --model {model_path} --source 0 --conf 0.45")

if __name__ == "__main__":
    main()

