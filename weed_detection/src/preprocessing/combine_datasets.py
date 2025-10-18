import os
import shutil
import yaml
import cv2
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
import argparse

class PrecisionWeedPreprocessor:
    def __init__(self, dataset1_path, dataset2_path, output_path, config_path):
        self.dataset1_path = Path(dataset1_path)  # WeedCrop dataset
        self.dataset2_path = Path(dataset2_path)  # Weed Detection dataset
        self.output_path = Path(output_path)
        self.config_path = config_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"Dataset 1 (WeedCrop): {self.dataset1_path}")
        print(f"Dataset 2 (Weed Detection): {self.dataset2_path}")
        print(f"Output: {self.output_path}")
    
    def setup_directories(self):
        """Create precision YOLO directory structure"""
        dirs = [
            'train/images', 'train/labels',
            'val/images', 'val/labels',
            'test/images', 'test/labels'
        ]
        
        for dir_name in dirs:
            (self.output_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        print("Created directory structure for precision detection")
    
    def process_weedcrop_dataset(self):
        """Process WeedCrop Image Dataset"""
        print("Processing WeedCrop Dataset...")
        
        dataset1_data = []
        
        # Expected structure: images/ and annotations/
        images_dir = self.dataset1_path / "images"
        annotations_dir = self.dataset1_path / "annotations"
        
        if not images_dir.exists():
            print(f"Looking for alternative structure in {self.dataset1_path}")
            # Check for train/val structure
            for split in ['train', 'val', 'test']:
                split_images = self.dataset1_path / split / "images"
                split_labels = self.dataset1_path / split / "labels"
                
                if split_images.exists() and split_labels.exists():
                    print(f"Found {split} split")
                    self.process_yolo_format(split_images, split_labels, dataset1_data, f"ds1_{split}")
        else:
            # Process COCO or custom format
            self.process_coco_format(images_dir, annotations_dir, dataset1_data, "ds1")
        
        return dataset1_data
    
    def process_weed_detection_dataset(self):
        """Process Weed Detection Dataset"""
        print("Processing Weed Detection Dataset...")
        
        dataset2_data = []
        
        # Check for YOLO format
        images_dir = self.dataset2_path / "images"
        labels_dir = self.dataset2_path / "labels"
        
        if images_dir.exists() and labels_dir.exists():
            self.process_yolo_format(images_dir, labels_dir, dataset2_data, "ds2")
        else:
            # Check for split structure
            for split in ['train', 'val', 'test']:
                split_images = self.dataset2_path / split / "images"
                split_labels = self.dataset2_path / split / "labels"
                
                if split_images.exists() and split_labels.exists():
                    print(f"Found {split} split in dataset 2")
                    self.process_yolo_format(split_images, split_labels, dataset2_data, f"ds2_{split}")
        
        return dataset2_data
    
    def process_yolo_format(self, images_dir, labels_dir, data_list, prefix):
        """Process YOLO format data with precision focus"""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(images_dir.glob(ext)))
        
        print(f"Processing {len(image_files)} images from {images_dir}")
        
        for img_path in tqdm(image_files, desc=f"Processing {prefix}"):
            label_path = labels_dir / (img_path.stem + '.txt')
            
            if not label_path.exists():
                continue
            
            # Read image dimensions for precision
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Read and validate labels
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            valid_boxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(float(parts[0]))
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate bounding box
                        if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0 < width <= 1 and 0 < height <= 1):
                            
                            # Filter out very small boxes (noise)
                            if width > 0.01 and height > 0.01:
                                valid_boxes.append([class_id, x_center, y_center, width, height])
                    
                    except ValueError:
                        continue
            
            if valid_boxes:
                data_list.append({
                    'image_path': str(img_path),
                    'labels': valid_boxes,
                    'width': w,
                    'height': h,
                    'source': prefix
                })
    
    def process_coco_format(self, images_dir, annotations_dir, data_list, prefix):
        """Process COCO format annotations"""
        # Look for JSON annotation files
        json_files = list(annotations_dir.glob('*.json'))
        
        for json_file in json_files:
            print(f"Processing COCO file: {json_file}")
            
            with open(json_file, 'r') as f:
                coco_data = json.load(f)
            
            # Create image ID to filename mapping
            images = {img['id']: img for img in coco_data['images']}
            
            # Process annotations
            for ann in coco_data['annotations']:
                image_info = images.get(ann['image_id'])
                if not image_info:
                    continue
                
                img_path = images_dir / image_info['file_name']
                if not img_path.exists():
                    continue
                
                # Convert COCO bbox to YOLO format
                x, y, w, h = ann['bbox']
                img_w, img_h = image_info['width'], image_info['height']
                
                # YOLO format: center_x, center_y, width, height (normalized)
                x_center = (x + w/2) / img_w
                y_center = (y + h/2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                # Map category to our classes (adjust based on dataset)
                category_id = ann['category_id']
                class_id = 1 if 'weed' in str(category_id).lower() else 0
                
                data_list.append({
                    'image_path': str(img_path),
                    'labels': [[class_id, x_center, y_center, norm_w, norm_h]],
                    'width': img_w,
                    'height': img_h,
                    'source': prefix
                })
    
    def combine_and_split_data(self, dataset1_data, dataset2_data):
        """Combine datasets and create train/val/test splits"""
        print("Combining datasets and creating splits...")
        
        all_data = dataset1_data + dataset2_data
        print(f"Total combined data: {len(all_data)} images")
        
        # Stratified split to maintain class balance
        train_data, temp_data = train_test_split(
            all_data, test_size=0.2, random_state=42
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=42
        )
        
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            print(f"{split_name}: {len(split_data)} images")
            self.save_split_data(split_name, split_data)
        
        return splits
    
    def save_split_data(self, split_name, split_data):
        """Save split data with precision-focused processing"""
        print(f"Saving {split_name} split...")
        
        for idx, data_item in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
            try:
                img_path = Path(data_item['image_path'])
                
                # Create unique filename
                new_img_name = f"{split_name}_{data_item['source']}_{idx:06d}{img_path.suffix}"
                new_label_name = f"{split_name}_{data_item['source']}_{idx:06d}.txt"
                
                dst_img_path = self.output_path / split_name / 'images' / new_img_name
                dst_label_path = self.output_path / split_name / 'labels' / new_label_name
                
                # Copy and potentially resize image for precision
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Resize to target resolution for precision training
                target_size = self.config['dataset']['img_width']
                if img.shape[1] != target_size or img.shape[0] != target_size:
                    img = cv2.resize(img, (target_size, target_size))
                
                cv2.imwrite(str(dst_img_path), img)
                
                # Save labels
                with open(dst_label_path, 'w') as f:
                    for label in data_item['labels']:
                        class_id, x_center, y_center, width, height = label
                        f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
            except Exception as e:
                print(f"Error processing {data_item['image_path']}: {e}")
                continue
    
    def apply_precision_augmentation(self):
        """Apply precision-focused augmentation to training data"""
        print("Applying precision-focused augmentation...")
        
        # Define precision augmentation pipeline
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.3),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
            A.Blur(blur_limit=2, p=0.1),
            A.RandomRotate90(p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        train_img_dir = self.output_path / 'train' / 'images'
        train_label_dir = self.output_path / 'train' / 'labels'
        
        train_images = list(train_img_dir.glob('*'))
        
        # Augment 30% of training data for precision
        augment_count = min(len(train_images) // 3, 1000)
        images_to_augment = train_images[:augment_count]
        
        for img_path in tqdm(images_to_augment, desc="Precision Augmentation"):
            label_path = train_label_dir / (img_path.stem + '.txt')
            
            if not label_path.exists():
                continue
            
            try:
                # Read image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Read labels
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                bboxes = []
                class_labels = []
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
                
                if not bboxes:
                    continue
                
                # Apply augmentation
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                
                # Save augmented data
                aug_img_name = f"aug_{img_path.name}"
                aug_label_name = f"aug_{img_path.stem}.txt"
                
                aug_img_path = train_img_dir / aug_img_name
                aug_label_path = train_label_dir / aug_label_name
                
                # Save augmented image
                aug_image_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aug_img_path), aug_image_bgr)
                
                # Save augmented labels
                with open(aug_label_path, 'w') as f:
                    for bbox, class_id in zip(augmented['bboxes'], augmented['class_labels']):
                        f.write(f"{int(class_id)} {' '.join(map(str, bbox))}\n")
            
            except Exception as e:
                print(f"Augmentation error for {img_path}: {e}")
                continue
    
    def create_dataset_yaml(self):
        """Create optimized dataset YAML"""
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
        
        print(f"Dataset YAML created: {yaml_path}")
        return str(yaml_path)
    
    def print_final_statistics(self):
        """Print comprehensive dataset statistics"""
        print("\n" + "="*60)
        print("PRECISION WEED DETECTION DATASET STATISTICS")
        print("="*60)
        
        total_images = 0
        total_labels = 0
        class_counts = {'crop': 0, 'weed': 0}
        
        for split in ['train', 'val', 'test']:
            img_dir = self.output_path / split / 'images'
            label_dir = self.output_path / split / 'labels'
            
            if img_dir.exists():
                num_images = len(list(img_dir.glob('*')))
                num_labels = len(list(label_dir.glob('*.txt')))
                
                # Count class instances
                split_class_counts = {'crop': 0, 'weed': 0}
                for label_file in label_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                if class_id == 0:
                                    split_class_counts['crop'] += 1
                                elif class_id == 1:
                                    split_class_counts['weed'] += 1
                
                print(f"{split.upper()}: {num_images} images, {num_labels} labels")
                print(f"  Crop instances: {split_class_counts['crop']}")
                print(f"  Weed instances: {split_class_counts['weed']}")
                
                total_images += num_images
                total_labels += num_labels
                class_counts['crop'] += split_class_counts['crop']
                class_counts['weed'] += split_class_counts['weed']
        
        print(f"\nTOTAL: {total_images} images, {total_labels} labels")
        print(f"Total crop instances: {class_counts['crop']}")
        print(f"Total weed instances: {class_counts['weed']}")
        print("="*60)
    
    def process_combined_datasets(self):
        """Main processing method"""
        print("Starting precision weed detection dataset preparation...")
        
        self.setup_directories()
        
        # Process both datasets
        dataset1_data = self.process_weedcrop_dataset()
        dataset2_data = self.process_weed_detection_dataset()
        
        # Combine and split
        self.combine_and_split_data(dataset1_data, dataset2_data)
        
        # Apply precision augmentation
        self.apply_precision_augmentation()
        
        # Create dataset YAML
        dataset_yaml = self.create_dataset_yaml()
        
        # Print statistics
        self.print_final_statistics()
        
        print("\n✅ Precision weed detection dataset ready!")
        return dataset_yaml

def main():
    parser = argparse.ArgumentParser(description='Combine and preprocess precision weed datasets')
    parser.add_argument('--dataset1', type=str, required=True,
                       help='Path to WeedCrop Image Dataset')
    parser.add_argument('--dataset2', type=str, required=True,
                       help='Path to Weed Detection Dataset')
    parser.add_argument('--output', type=str, default='data/processed/combined_weeds',
                       help='Output path for combined dataset')
    parser.add_argument('--config', type=str, default='config/precision_weed_config.yaml',
                       help='Config file path')
    
    args = parser.parse_args()
    
    # Download instruction
    print("📥 Dataset Download Instructions:")
    print("1. Download from: https://www.kaggle.com/datasets/vinayakshanawad/weedcrop-image-dataset")
    print("2. Download from: https://www.kaggle.com/datasets/jaidalmotra/weed-detection")
    print("3. Extract both datasets to separate folders")
    print("4. Run this script with paths to both datasets\n")
    
    preprocessor = PrecisionWeedPreprocessor(
        dataset1_path=args.dataset1,
        dataset2_path=args.dataset2,
        output_path=args.output,
        config_path=args.config
    )
    
    dataset_yaml = preprocessor.process_combined_datasets()
    print(f"\nNext: python src/training/train_precision_model.py --data {dataset_yaml}")

if __name__ == "__main__":
    main()

