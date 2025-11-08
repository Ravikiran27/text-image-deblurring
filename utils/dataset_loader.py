"""
Dataset Loader for Text Image Deblurring

This module provides utilities to load, preprocess, and batch paired blurred/original images
for training and validation.
"""

import os
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split


class DeblurDataset:
    """
    Dataset loader for paired blurred and original images.
    
    Attributes:
        blur_dir (str): Directory containing blurred images
        orig_dir (str): Directory containing original (ground truth) images
        image_size (tuple): Target size for images (height, width)
        batch_size (int): Batch size for training
    """
    
    def __init__(self, blur_dir, orig_dir, image_size=(256, 256), batch_size=32):
        """
        Initialize the dataset loader.
        
        Args:
            blur_dir (str): Path to blurred images directory
            orig_dir (str): Path to original images directory
            image_size (tuple): Target image dimensions
            batch_size (int): Batch size for data loading
        """
        self.blur_dir = Path(blur_dir)
        self.orig_dir = Path(orig_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        
        self.blur_images = []
        self.orig_images = []
        
        print(f"Initializing dataset from:")
        print(f"  Blur: {self.blur_dir}")
        print(f"  Orig: {self.orig_dir}")
    
    def load_image_paths(self):
        """
        Load and match paired image paths from blur and orig directories.
        
        Returns:
            tuple: Lists of matched blur and original image paths
        """
        # Get all image files from both directories
        blur_files = sorted([f for f in self.blur_dir.glob('*') 
                           if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        orig_files = sorted([f for f in self.orig_dir.glob('*') 
                           if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        
        # Match files by name
        blur_dict = {f.stem: f for f in blur_files}
        orig_dict = {f.stem: f for f in orig_files}
        
        matched_blur = []
        matched_orig = []
        
        for name in blur_dict:
            if name in orig_dict:
                matched_blur.append(str(blur_dict[name]))
                matched_orig.append(str(orig_dict[name]))
        
        print(f"Found {len(matched_blur)} matched image pairs")
        
        return matched_blur, matched_orig
    
    def preprocess_image(self, image_path, normalize=True):
        """
        Load and preprocess a single image.
        
        Args:
            image_path (str): Path to image file
            normalize (bool): Whether to normalize pixel values to [0, 1]
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        if normalize:
            img = img.astype(np.float32) / 255.0
        
        return img
    
    def load_all_images(self):
        """
        Load all matched image pairs into memory.
        
        Returns:
            tuple: Arrays of blur and original images
        """
        blur_paths, orig_paths = self.load_image_paths()
        
        if len(blur_paths) == 0:
            print("WARNING: No matched images found!")
            return np.array([]), np.array([])
        
        blur_images = []
        orig_images = []
        
        print("Loading images...")
        for i, (blur_path, orig_path) in enumerate(zip(blur_paths, orig_paths)):
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{len(blur_paths)} images")
            
            try:
                blur_img = self.preprocess_image(blur_path)
                orig_img = self.preprocess_image(orig_path)
                
                blur_images.append(blur_img)
                orig_images.append(orig_img)
            except Exception as e:
                print(f"Error loading {blur_path}: {e}")
                continue
        
        print(f"Successfully loaded {len(blur_images)} image pairs")
        
        return np.array(blur_images), np.array(orig_images)
    
    def create_train_val_split(self, test_size=0.2, random_state=42):
        """
        Load images and split into train/validation sets.
        
        Args:
            test_size (float): Proportion of dataset for validation
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        X, y = self.load_all_images()
        
        if len(X) == 0:
            print("No data loaded. Returning empty arrays.")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"\nDataset split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        
        return X_train, X_val, y_train, y_val
    
    def create_tf_dataset(self, X, y, shuffle=True):
        """
        Create TensorFlow dataset from numpy arrays.
        
        Args:
            X (np.ndarray): Input images (blurred)
            y (np.ndarray): Target images (original)
            shuffle (bool): Whether to shuffle the dataset
            
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def create_dummy_dataset(blur_dir, orig_dir, num_images=100, image_size=(256, 256)):
    """
    Create dummy blurred and original images for testing purposes.
    
    This function generates synthetic image pairs by creating sharp images
    and applying Gaussian blur to create blurred versions.
    
    Args:
        blur_dir (str): Directory to save blurred images
        orig_dir (str): Directory to save original images
        num_images (int): Number of image pairs to generate
        image_size (tuple): Size of generated images
    """
    os.makedirs(blur_dir, exist_ok=True)
    os.makedirs(orig_dir, exist_ok=True)
    
    print(f"Generating {num_images} dummy image pairs...")
    
    for i in range(num_images):
        # Create a synthetic image with text-like patterns
        img = np.random.randint(200, 255, (*image_size, 3), dtype=np.uint8)
        
        # Add some text-like rectangular regions
        num_rects = np.random.randint(3, 8)
        for _ in range(num_rects):
            x1 = np.random.randint(0, image_size[1] - 50)
            y1 = np.random.randint(0, image_size[0] - 20)
            x2 = x1 + np.random.randint(30, 100)
            y2 = y1 + np.random.randint(10, 30)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        # Save original image
        orig_path = os.path.join(orig_dir, f"image_{i:04d}.png")
        cv2.imwrite(orig_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Create blurred version
        blurred = cv2.GaussianBlur(img, (15, 15), 0)
        blur_path = os.path.join(blur_dir, f"image_{i:04d}.png")
        cv2.imwrite(blur_path, cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_images} image pairs")
    
    print(f"Dummy dataset created successfully!")
    print(f"  Blurred images: {blur_dir}")
    print(f"  Original images: {orig_dir}")


if __name__ == "__main__":
    # Example usage
    blur_dir = "../data/blur"
    orig_dir = "../data/orig"
    
    # Create dummy dataset if directories are empty
    if not os.path.exists(blur_dir) or len(os.listdir(blur_dir)) == 0:
        print("Creating dummy dataset for testing...")
        create_dummy_dataset(blur_dir, orig_dir, num_images=50)
    
    # Initialize dataset loader
    dataset = DeblurDataset(blur_dir, orig_dir, image_size=(256, 256), batch_size=8)
    
    # Load and split data
    X_train, X_val, y_train, y_val = dataset.create_train_val_split(test_size=0.2)
    
    print(f"\nDataset shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")
