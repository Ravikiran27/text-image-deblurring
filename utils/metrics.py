"""
Metrics for Image Quality Assessment

This module provides functions to compute image quality metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_psnr(original, reconstructed, data_range=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    PSNR measures the ratio between the maximum possible signal power
    and the power of corrupting noise. Higher values indicate better quality.
    
    Args:
        original (np.ndarray): Ground truth image
        reconstructed (np.ndarray): Reconstructed/deblurred image
        data_range (float): Dynamic range of the images (1.0 for normalized images)
        
    Returns:
        float: PSNR value in decibels (dB)
    """
    return psnr(original, reconstructed, data_range=data_range)


def calculate_ssim(original, reconstructed, data_range=1.0, multichannel=True):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    SSIM measures the perceptual similarity between two images.
    Values range from -1 to 1, where 1 indicates perfect similarity.
    
    Args:
        original (np.ndarray): Ground truth image
        reconstructed (np.ndarray): Reconstructed/deblurred image
        data_range (float): Dynamic range of the images (1.0 for normalized images)
        multichannel (bool): Whether to treat image as multichannel (RGB)
        
    Returns:
        float: SSIM value (0 to 1)
    """
    return ssim(
        original, 
        reconstructed, 
        data_range=data_range,
        channel_axis=2 if multichannel else None
    )


def calculate_batch_metrics(originals, reconstructed):
    """
    Calculate PSNR and SSIM metrics for a batch of images.
    
    Args:
        originals (np.ndarray): Batch of ground truth images (N, H, W, C)
        reconstructed (np.ndarray): Batch of reconstructed images (N, H, W, C)
        
    Returns:
        dict: Dictionary containing mean and std of PSNR and SSIM
    """
    psnr_values = []
    ssim_values = []
    
    for orig, recon in zip(originals, reconstructed):
        psnr_val = calculate_psnr(orig, recon)
        ssim_val = calculate_ssim(orig, recon)
        
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
    
    results = {
        'psnr_mean': np.mean(psnr_values),
        'psnr_std': np.std(psnr_values),
        'ssim_mean': np.mean(ssim_values),
        'ssim_std': np.std(ssim_values),
        'psnr_values': psnr_values,
        'ssim_values': ssim_values
    }
    
    return results


def display_metrics_summary(metrics):
    """
    Display a formatted summary of image quality metrics.
    
    Args:
        metrics (dict): Dictionary containing PSNR and SSIM statistics
    """
    print("\n" + "="*60)
    print("IMAGE QUALITY METRICS SUMMARY")
    print("="*60)
    print(f"PSNR (Peak Signal-to-Noise Ratio):")
    print(f"  Mean: {metrics['psnr_mean']:.2f} dB")
    print(f"  Std:  {metrics['psnr_std']:.2f} dB")
    print(f"\nSSIM (Structural Similarity Index):")
    print(f"  Mean: {metrics['ssim_mean']:.4f}")
    print(f"  Std:  {metrics['ssim_std']:.4f}")
    print("="*60)
    print(f"Total images evaluated: {len(metrics['psnr_values'])}")
    print("="*60 + "\n")


def calculate_mae(original, reconstructed):
    """
    Calculate Mean Absolute Error between two images.
    
    Args:
        original (np.ndarray): Ground truth image
        reconstructed (np.ndarray): Reconstructed image
        
    Returns:
        float: MAE value
    """
    return np.mean(np.abs(original - reconstructed))


def calculate_mse(original, reconstructed):
    """
    Calculate Mean Squared Error between two images.
    
    Args:
        original (np.ndarray): Ground truth image
        reconstructed (np.ndarray): Reconstructed image
        
    Returns:
        float: MSE value
    """
    return np.mean((original - reconstructed) ** 2)


if __name__ == "__main__":
    # Example usage with dummy images
    print("Testing metrics module...")
    
    # Create dummy images
    height, width = 256, 256
    
    # Original image (sharp)
    original = np.random.rand(height, width, 3).astype(np.float32)
    
    # Simulate reconstructed image with slight noise
    noise = np.random.normal(0, 0.05, (height, width, 3)).astype(np.float32)
    reconstructed = np.clip(original + noise, 0, 1)
    
    # Calculate metrics
    psnr_val = calculate_psnr(original, reconstructed)
    ssim_val = calculate_ssim(original, reconstructed)
    mae_val = calculate_mae(original, reconstructed)
    mse_val = calculate_mse(original, reconstructed)
    
    print(f"\nSingle Image Metrics:")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    print(f"  MAE:  {mae_val:.6f}")
    print(f"  MSE:  {mse_val:.6f}")
    
    # Test batch metrics
    batch_size = 5
    originals = np.random.rand(batch_size, height, width, 3).astype(np.float32)
    noise_batch = np.random.normal(0, 0.05, (batch_size, height, width, 3)).astype(np.float32)
    reconstructed_batch = np.clip(originals + noise_batch, 0, 1)
    
    batch_metrics = calculate_batch_metrics(originals, reconstructed_batch)
    display_metrics_summary(batch_metrics)
