import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(base_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset'), train_ratio=0.8):
    """
    Splits dataset/fake and dataset/real into dataset/train/... and dataset/val/...
    Handles nested directory structure (e.g., dataset/real/001/, dataset/fake/002/)
    """
    categories = ["fake", "real"]
    
    for category in categories:
        src_folder = os.path.join(base_dir, category)
        all_images = []
        
        # Walk through all subdirectories and collect image paths
        for root, _, files in os.walk(src_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                    all_images.append(rel_path)
        
        if not all_images:
            print(f"⚠️ No images found in {src_folder}")
            continue
            
        # Split into train and validation sets
        train_images, val_images = train_test_split(all_images, train_size=train_ratio, random_state=42)
        
        # Create target directories
        train_dir = os.path.join(base_dir, "train")
        val_dir = os.path.join(base_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Function to move files to their respective directories
        def move_files(file_list, target_dir):
            for img_path in file_list:
                src_path = os.path.join(base_dir, img_path)
                dst_path = os.path.join(target_dir, img_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                if os.path.exists(src_path):  # Check if source file exists before moving
                    shutil.move(src_path, dst_path)
                    print(f"Moved: {src_path} -> {dst_path}")
                else:
                    print(f"⚠️ Source file not found: {src_path}")
        
        # Move the files to their respective directories
        print(f"🚀 Moving {len(train_images)} {category} images to train...")
        move_files(train_images, train_dir)
        print(f"🚀 Moving {len(val_images)} {category} images to validation...")
        move_files(val_images, val_dir)
        
        print(f"✅ {category}: {len(train_images)} train | {len(val_images)} val")
        print("-" * 50)

if __name__ == "__main__":
    split_dataset()
