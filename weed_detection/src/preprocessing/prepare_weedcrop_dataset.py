import os
import yaml
import shutil
from pathlib import Path
import argparse

class WeedCropDatasetPreparator:
    def __init__(self, dataset_path, config_path):
        self.dataset_path = Path(dataset_path)
        self.config_path = config_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"📁 WeedCrop Dataset: {self.dataset_path}")
    
    def validate_dataset_structure(self):
        """Validate the dataset has correct YOLO structure"""
        required_dirs = [
            'train/images', 'train/labels',
            'valid/images', 'valid/labels',
            'test/images', 'test/labels'
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.dataset_path / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            print(f"❌ Missing directories: {missing_dirs}")
            return False
        
        print("✅ Dataset structure validated")
        return True
    
    def update_data_yaml(self):
        """Update the data.yaml file with correct paths"""
        data_yaml_path = self.dataset_path / 'data.yaml'
        
        if not data_yaml_path.exists():
            print("❌ data.yaml not found, creating new one...")
        
        # Count classes from train labels
        train_labels_dir = self.dataset_path / 'train' / 'labels'
        classes = set()
        
        if train_labels_dir.exists():
            for label_file in train_labels_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            classes.add(int(parts[0]))
        
        num_classes = len(classes) if classes else 2
        max_class = max(classes) if classes else 1
        
        # Generate class names
        if num_classes == 2:
            class_names = ['crop', 'weed']
        else:
            class_names = [f'class_{i}' for i in range(max_class + 1)]
        
        # Create updated data.yaml content
        yaml_content = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': num_classes,
            'names': class_names
        }
        
        # Save updated data.yaml
        with open(data_yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"✅ Updated data.yaml:")
        print(f"  Classes: {num_classes}")
        print(f"  Names: {class_names}")
        
        return str(data_yaml_path)
    
    def print_dataset_stats(self):
        """Print dataset statistics"""
        print("\n" + "="*50)
        print("WEEDCROP DATASET STATISTICS")
        print("="*50)
        
        for split in ['train', 'valid', 'test']:
            img_dir = self.dataset_path / split / 'images'
            label_dir = self.dataset_path / split / 'labels'
            
            if img_dir.exists() and label_dir.exists():
                num_images = len(list(img_dir.glob('*')))
                num_labels = len(list(label_dir.glob('*.txt')))
                print(f"{split.upper()}: {num_images} images, {num_labels} labels")
        
        print("="*50)
    
    def prepare_dataset(self):
        """Complete dataset preparation"""
        print("🚀 Preparing WeedCrop dataset for training...")
        
        if not self.validate_dataset_structure():
            return None
        
        data_yaml_path = self.update_data_yaml()
        self.print_dataset_stats()
        
        print("\n✅ Dataset ready for training!")
        return data_yaml_path

def main():
    parser = argparse.ArgumentParser(description='Prepare WeedCrop Dataset')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Path to downloaded WeedCrop dataset folder')
    parser.add_argument('--config', type=str, default='config/weedcrop_config.yaml')
    
    args = parser.parse_args()
    
    preparator = WeedCropDatasetPreparator(args.dataset, args.config)
    data_yaml_path = preparator.prepare_dataset()
    
    if data_yaml_path:
        print(f"\n🚀 Ready to train:")
        print(f"python src/training/train_weedcrop_model.py --data {data_yaml_path}")

if __name__ == "__main__":
    main()

