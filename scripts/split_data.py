import os
import shutil
import random
from pathlib import Path

def split_train_val(train_dir, val_dir, split_ratio=0.2):

    train_path = Path(train_dir)
    val_path = Path(val_dir)
    
    # Create validation directory if it doesn't exist
    val_path.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in train_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    
    total_moved = 0
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Create validation class directory
        val_class_dir = val_path / class_name
        val_class_dir.mkdir(exist_ok=True)
        
        # Get all images in this class
        image_files = list(class_dir.glob("*.jpg"))
        
        # Shuffle and split
        random.shuffle(image_files)
        val_count = int(len(image_files) * split_ratio)
        val_files = image_files[:val_count]
        
        # Move validation files
        for img_file in val_files:
            dest_path = val_class_dir / img_file.name
            shutil.move(str(img_file), str(dest_path))
            
        total_moved += len(val_files)
        print(f"Class {class_name}: {len(image_files)} total, moved {len(val_files)} to validation")
    
    print(f"\nTotal images moved to validation: {total_moved}")
    
    # Count remaining files
    remaining_train = sum(len(list(d.glob("*.jpg"))) for d in train_path.iterdir() if d.is_dir())
    total_val = sum(len(list(d.glob("*.jpg"))) for d in val_path.iterdir() if d.is_dir())
    
    print(f"Final split - Train: {remaining_train}, Val: {total_val}")

if __name__ == "__main__":
    random.seed(42)
    
    # Define paths
    project_root = Path(__file__).parent.parent
    train_dir = project_root / "data" / "train"
    val_dir = project_root / "data" / "val"
    
    print("Splitting ASL Alphabet dataset into train/validation sets...")
    split_train_val(str(train_dir), str(val_dir))
    print("Split complete!")