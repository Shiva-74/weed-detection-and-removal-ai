import os
import shutil
import yaml
import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
import argparse

class SingleDatasetPreprocessor:
    def __init__(self, dataset_path, output_path, config_path):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.config_path = config_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"📁 Processing single dataset: {self.dataset_path}")
        print(f"📁 Output: {self.output_path}")
    
    def setup_directories(self):
        """Create directory structure"""
        dirs = [
            'train/images', 'train/labels',
            'val/images', 'val/labels',
            'test/images', 'test/labels'
        ]
        
        for dir_name in dirs:
            (self.output_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    def process_dataset(self):
        """Process single dataset with quality focus"""
        print("🔍 Processing dataset for maximum accuracy...")
        
        # Find all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(self.dataset_path.rglob(ext)))
        
        print(f"Found {len(image_files)} images")
        
        # Filter for quality
        valid_data = []
        
        for img_path in tqdm(image_files, desc="Quality filtering"):
            # Check if corresponding label exists
            possible_label_paths = [
                img_path.with_suffix('.txt'),
                img_path.parent.parent / 'labels' / (img_path.stem + '.txt'),
                img_path.parent / 'labels' / (img_path.stem + '.txt')
            ]
            
            label_path = None
            for lp in possible_label_paths:
                if lp.exists():
                    label_path = lp
                    break
            
            if not label_path:
                continue
            
            # Validate image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            if h < 200 or w < 200:  # Skip very small images
                continue
            
            # Validate labels
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                valid_boxes = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(float(parts[0]))
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Quality checks
                        if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0.005 < width <= 1 and 0.005 < height <= 1):  # Min size filter
                            
                            # Map to crop/weed classes
                            if class_id >= 1:  # Any non-background class is weed
                                class_id = 1
                            valid_boxes.append([class_id, x_center, y_center, width, height])
                
                if valid_boxes:
                    valid_data.append({
                        'image_path': str(img_path),
                        'label_path': str(label_path),
                        'boxes': valid_boxes,
                        'width': w,
                        'height': h
                    })
                    
            except Exception as e:
                continue
        
        print(f"✅ Quality filtered: {len(valid_data)} valid samples")
        
        # Split dataset - 80/10/10
        train_data, temp_data = train_test_split(valid_data, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        splits = {
            'train': train_data,
            'val': val_data, 
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            print(f"Processing {split_name}: {len(split_data)} samples")
            self.save_split_data(split_name, split_data)
    
    def save_split_data(self, split_name, split_data):
        """Save data with quality enhancement"""
        for idx, data_item in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
            try:
                img_path = Path(data_item['image_path'])
                
                # Create unique filename
                new_img_name = f"{split_name}_{idx:06d}{img_path.suffix}"
                new_label_name = f"{split_name}_{idx:06d}.txt"
                
                dst_img_path = self.output_path / split_name / 'images' / new_img_name
                dst_label_path = self.output_path / split_name / 'labels' / new_label_name
                
                # Copy and optimize image
                img = cv2.imread(data_item['image_path'])
                if img is None:
                    continue
                
                # Resize to standard size if needed
                target_size = 640
                if img.shape[1] != target_size or img.shape[0] != target_size:
                    img = cv2.resize(img, (target_size, target_size))
                
                cv2.imwrite(str(dst_img_path), img)
                
                # Save labels
                with open(dst_label_path, 'w') as f:
                    for box in data_item['boxes']:
                        class_id, x_center, y_center, width, height = box
                        f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
            except Exception as e:
                print(f"Error processing {data_item['image_path']}: {e}")
                continue
    
    def create_dataset_yaml(self):
        """Create dataset YAML"""
        dataset_config = {
            'path': str(self.output_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images', 
            'nc': self.config['dataset']['nc'],
            'names': self.config['dataset']['names']
        }
        
        yaml_path = self.output_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        return str(yaml_path)
    
    def process_complete(self):
        """Complete processing pipeline"""
        self.setup_directories()
        self.process_dataset()
        dataset_yaml = self.create_dataset_yaml()
        
        print(f"\n✅ High-quality dataset ready: {dataset_yaml}")
        return dataset_yaml

def main():
    parser = argparse.ArgumentParser(description='Process single dataset for maximum accuracy')
    parser.add_argument('--dataset', type=str, required=True, help='Path to single dataset')
    parser.add_argument('--output', type=str, default='data/processed/single_weed_dataset')
    parser.add_argument('--config', type=str, default='config/high_accuracy_weed_config.yaml')
    
    args = parser.parse_args()
    
    preprocessor = SingleDatasetPreprocessor(args.dataset, args.output, args.config)
    dataset_yaml = preprocessor.process_complete()
    
    print(f"\n🚀 Ready to train:")
    print(f"python src/training/train_high_accuracy_model.py --data {dataset_yaml}")

if __name__ == "__main__":
    main()

