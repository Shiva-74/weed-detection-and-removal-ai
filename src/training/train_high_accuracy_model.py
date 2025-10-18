import os
import yaml
import torch
import gc
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import shutil
import logging

class HighAccuracyWeedTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.device = self.setup_device()
        self.model_save_dir = Path("models")
        self.model_save_dir.mkdir(exist_ok=True)
        
        print("🎯 High-Accuracy Weed Detection Trainer Initialized")
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('high_accuracy_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_device(self):
        if torch.cuda.is_available():
            device = f"cuda:{self.config['training']['device']}"
            torch.cuda.empty_cache()
            gc.collect()
            
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"Total Memory: {total_memory:.1f} GB")
        else:
            device = "cpu"
        return device
    
    def train_high_accuracy_model(self, dataset_yaml_path):
        """Train for maximum accuracy and precision"""
        self.logger.info("🚀 Starting HIGH-ACCURACY training...")
        
        model_name = self.config['model']['architecture']
        model = YOLO(f"{model_name}.pt")
        
        # High-accuracy training parameters
        train_args = {
            'data': dataset_yaml_path,
            'epochs': self.config['training']['epochs'],  # 200
            'batch': self.config['training']['batch_size'],  # 16
            'imgsz': self.config['training']['image_size'],  # 640
            'lr0': self.config['training']['learning_rate'],
            'lrf': 0.001,  # Lower final LR for fine-tuning
            'weight_decay': self.config['training']['weight_decay'],
            'momentum': self.config['training']['momentum'],
            'device': self.device,
            'workers': self.config['training']['workers'],
            'patience': self.config['training']['patience'],  # 50
            'save_period': self.config['training']['save_period'],
            'project': 'runs/detect',
            'name': f'high_accuracy_weed_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': self.config['model']['pretrained'],
            'optimizer': self.config['optimization']['optimizer'],
            'close_mosaic': self.config['optimization']['close_mosaic'],
            'amp': self.config['optimization']['amp'],
            
            # Lower confidence for high recall
            'conf': self.config['deployment']['confidence'],  # 0.15
            'iou': self.config['deployment']['iou_threshold'],  # 0.4
            'max_det': self.config['deployment']['max_detections'],  # 300
            
            # Quality settings
            'cache': False,
            'rect': False,
            'multi_scale': True,  # Multi-scale training
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
        
        # Enhanced augmentation
        aug = self.config['augmentation']
        train_args.update({
            'hsv_h': aug['hsv_h'], 'hsv_s': aug['hsv_s'], 'hsv_v': aug['hsv_v'],
            'degrees': aug['degrees'], 'translate': aug['translate'], 
            'scale': aug['scale'], 'shear': aug['shear'],
            'perspective': aug['perspective'], 'flipud': aug['flipud'], 
            'fliplr': aug['fliplr'], 'mosaic': aug['mosaic'], 'mixup': aug['mixup']
        })
        
        self.logger.info(f"🎯 High-Accuracy Configuration:")
        self.logger.info(f"  Model: {model_name}")
        self.logger.info(f"  Epochs: {self.config['training']['epochs']}")
        self.logger.info(f"  Batch: {self.config['training']['batch_size']}")
        self.logger.info(f"  Confidence: {self.config['deployment']['confidence']} (15% for high recall)")
        self.logger.info(f"  Multi-scale: True")
        
        try:
            torch.cuda.empty_cache()
            results = model.train(**train_args)
            best_model_path = self.save_model()
            return results, best_model_path
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self):
        """Save the trained model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"high_accuracy_weed_detector_{timestamp}.pt"
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
    
    def validate_model(self, model_path, dataset_yaml_path):
        """Comprehensive validation"""
        if not model_path:
            return None
            
        self.logger.info("🔍 Comprehensive validation...")
        
        model = YOLO(model_path)
        
        # Test with different confidence thresholds
        conf_thresholds = [0.1, 0.15, 0.25, 0.35, 0.45]
        best_results = None
        best_f1 = 0
        
        for conf in conf_thresholds:
            results = model.val(
                data=dataset_yaml_path,
                device=self.device,
                conf=conf,
                iou=self.config['deployment']['iou_threshold'],
                batch=4,
                plots=False,
                verbose=False
            )
            
            precision = results.box.mp
            recall = results.box.mr
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            self.logger.info(f"Conf {conf}: P={precision:.3f}, R={recall:.3f}, F1={f1_score:.3f}, mAP50={results.box.map50:.3f}")
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_results = (results, conf)
        
        if best_results:
            results, best_conf = best_results
            self.logger.info(f"🏆 BEST PERFORMANCE at confidence {best_conf}:")
            self.logger.info(f"  Precision: {results.box.mp:.3f}")
            self.logger.info(f"  Recall: {results.box.mr:.3f}")
            self.logger.info(f"  F1-Score: {best_f1:.3f}")
            self.logger.info(f"  mAP50: {results.box.map50:.3f}")
            self.logger.info(f"  mAP50-95: {results.box.map:.3f}")
        
        return best_results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train High-Accuracy Weed Detection')
    parser.add_argument('--config', type=str, default='config/high_accuracy_weed_config.yaml')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML path')
    
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        print(f"❌ Dataset not found: {args.data}")
        return
    
    trainer = HighAccuracyWeedTrainer(args.config)
    results, model_path = trainer.train_high_accuracy_model(args.data)
    
    if model_path:
        best_results = trainer.validate_model(model_path, args.data)
        
        print("\n" + "="*60)
        print("🏆 HIGH-ACCURACY TRAINING COMPLETED!")
        print("="*60)
        print(f"✅ Model: {model_path}")
        
        if best_results:
            results, best_conf = best_results
            print(f"🎯 Best Performance (Confidence = {best_conf}):")
            print(f"  Precision: {results.box.mp:.3f}")
            print(f"  Recall: {results.box.mr:.3f}")  
            print(f"  mAP50: {results.box.map50:.3f}")
            print(f"  mAP50-95: {results.box.map:.3f}")
        
        print("="*60)
        print(f"🚀 Test: python src/inference/precision_laser_system.py --model {model_path} --source 0 --conf {best_conf if best_results else 0.15}")

if __name__ == "__main__":
    main()

