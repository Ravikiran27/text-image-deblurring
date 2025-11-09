"""
Train MAXIM model for text image deblurring
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.maxim_model import build_maxim_lightweight, compile_maxim_model


def load_dataset(blur_dir, orig_dir, img_size=(256, 256)):
    """Load image pairs from directories."""
    blur_files = sorted(list(Path(blur_dir).glob('**/*.png')) + 
                       list(Path(blur_dir).glob('**/*.jpg')))
    orig_files = sorted(list(Path(orig_dir).glob('**/*.png')) + 
                       list(Path(orig_dir).glob('**/*.jpg')))
    
    X, y = [], []
    
    for blur_path, orig_path in zip(blur_files, orig_files):
        # Load and preprocess blur image
        blur_img = cv2.imread(str(blur_path))
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        blur_img = cv2.resize(blur_img, img_size)
        blur_img = blur_img.astype(np.float32) / 255.0
        
        # Load and preprocess original image
        orig_img = cv2.imread(str(orig_path))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(orig_img, img_size)
        orig_img = orig_img.astype(np.float32) / 255.0
        
        X.append(blur_img)
        y.append(orig_img)
    
    return np.array(X), np.array(y)


def train_maxim(blur_dir, orig_dir, save_path, epochs=50, batch_size=8):
    """
    Train MAXIM model.
    
    Args:
        blur_dir: Directory with blurred images
        orig_dir: Directory with original images
        save_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("=" * 60)
    print("MAXIM MODEL TRAINING")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    X, y = load_dataset(blur_dir, orig_dir)
    print(f"Loaded {len(X)} image pairs")
    print(f"Shape: {X.shape}")
    
    # Split dataset
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model
    print("\nBuilding MAXIM model...")
    model = build_maxim_lightweight(input_shape=(256, 256, 3))
    model = compile_maxim_model(model, learning_rate=0.0001)
    print(f"Total parameters: {model.count_params():,}")
    
    # Callbacks
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Model saved to: {save_path}")
    
    return model, history


if __name__ == "__main__":
    # Configuration
    BLUR_DIR = r"R:\AI Image Deblurring\text_deblurring_pretrained\data\blur"
    ORIG_DIR = r"R:\AI Image Deblurring\text_deblurring_pretrained\data\orig"
    SAVE_PATH = r"R:\AI Image Deblurring\text_deblurring_pretrained\saved_models\maxim_deblur_best.h5"
    
    # Train
    model, history = train_maxim(
        BLUR_DIR, 
        ORIG_DIR, 
        SAVE_PATH,
        epochs=50,
        batch_size=8
    )
