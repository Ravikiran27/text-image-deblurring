"""
Test Script for Text Image Deblurring Model

This script evaluates a trained deblurring model on test images and computes
PSNR and SSIM metrics. It also generates visual comparisons.

Usage:
    python test.py --model_path saved_models/vgg16_deblur_best.h5 --test_blur data/blur --test_orig data/orig
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf

# Import utility functions
from utils.metrics import calculate_psnr, calculate_ssim, calculate_batch_metrics, display_metrics_summary
from utils.dataset_loader import DeblurDataset


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Test Text Image Deblurring Model')
    
    parser.add_argument('--model_path', type=str, default='saved_models/vgg16_deblur_best.h5',
                        help='Path to trained model file (.h5)')
    parser.add_argument('--test_blur', type=str, default='data/blur',
                        help='Directory containing blurred test images')
    parser.add_argument('--test_orig', type=str, default='data/orig',
                        help='Directory containing original test images')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save test results')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                        help='Image dimensions (height width)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--num_visual_samples', type=int, default=10,
                        help='Number of visual comparison samples to generate')
    
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from file.
    
    Args:
        model_path (str): Path to model file
        
    Returns:
        tf.keras.Model: Loaded model
    """
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully!")
    print(f"Model name: {model.name}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    return model


def test_model(model, X_test, y_test, batch_size=16):
    """
    Run inference on test data and compute metrics.
    
    Args:
        model: Trained Keras model
        X_test: Blurred test images
        y_test: Original test images
        batch_size: Batch size for inference
        
    Returns:
        tuple: (predictions, metrics_dict)
    """
    print("\n" + "="*60)
    print("RUNNING INFERENCE")
    print("="*60)
    
    # Make predictions
    predictions = model.predict(X_test, batch_size=batch_size, verbose=1)
    
    print("\nCalculating quality metrics...")
    
    # Calculate metrics for entire batch
    metrics = calculate_batch_metrics(y_test, predictions)
    
    # Display summary
    display_metrics_summary(metrics)
    
    return predictions, metrics


def save_visual_comparisons(X_test, y_test, predictions, metrics, output_dir, num_samples=10):
    """
    Save visual comparisons of blurred, deblurred, and ground truth images.
    
    Args:
        X_test: Blurred test images
        y_test: Original test images
        predictions: Model predictions
        metrics: Dictionary containing PSNR and SSIM values
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = min(num_samples, len(X_test))
    psnr_values = metrics['psnr_values']
    ssim_values = metrics['ssim_values']
    
    print(f"\nGenerating visual comparisons for {num_samples} samples...")
    
    # Create grid visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Blurred input
        axes[i, 0].imshow(X_test[i])
        axes[i, 0].set_title(f"Sample {i+1}: Blurred Input", fontsize=10)
        axes[i, 0].axis('off')
        
        # Deblurred output
        axes[i, 1].imshow(predictions[i])
        axes[i, 1].set_title(f"Deblurred Output\nPSNR: {psnr_values[i]:.2f} dB", fontsize=10)
        axes[i, 1].axis('off')
        
        # Ground truth
        axes[i, 2].imshow(y_test[i])
        axes[i, 2].set_title(f"Ground Truth\nSSIM: {ssim_values[i]:.4f}", fontsize=10)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'visual_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visual comparison saved to: {comparison_path}")
    
    # Save individual comparison images
    individual_dir = os.path.join(output_dir, 'individual_comparisons')
    os.makedirs(individual_dir, exist_ok=True)
    
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(X_test[i])
        axes[0].set_title("Blurred Input", fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(predictions[i])
        axes[1].set_title(f"Deblurred Output\nPSNR: {psnr_values[i]:.2f} dB, SSIM: {ssim_values[i]:.4f}", fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(y_test[i])
        axes[2].set_title("Ground Truth", fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        individual_path = os.path.join(individual_dir, f'comparison_{i+1}.png')
        plt.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Individual comparisons saved to: {individual_dir}")


def save_metrics_plot(metrics, output_dir):
    """
    Save plots of PSNR and SSIM distributions.
    
    Args:
        metrics: Dictionary containing metric values
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PSNR histogram
    axes[0].hist(metrics['psnr_values'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(metrics['psnr_mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {metrics['psnr_mean']:.2f} dB")
    axes[0].set_xlabel('PSNR (dB)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('PSNR Distribution', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SSIM histogram
    axes[1].hist(metrics['ssim_values'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1].axvline(metrics['ssim_mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {metrics['ssim_mean']:.4f}")
    axes[1].set_xlabel('SSIM', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('SSIM Distribution', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'metrics_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics distribution plot saved to: {plot_path}")


def save_metrics_summary(metrics, output_dir, model_name):
    """
    Save metrics summary to text file.
    
    Args:
        metrics: Dictionary containing metric values
        output_dir: Directory to save summary
        model_name: Name of the model
    """
    summary_path = os.path.join(output_dir, 'metrics_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TEXT IMAGE DEBLURRING - TEST RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Number of test images: {len(metrics['psnr_values'])}\n\n")
        f.write("METRICS SUMMARY:\n")
        f.write("-"*60 + "\n")
        f.write(f"PSNR (Peak Signal-to-Noise Ratio):\n")
        f.write(f"  Mean:   {metrics['psnr_mean']:.2f} dB\n")
        f.write(f"  Std:    {metrics['psnr_std']:.2f} dB\n")
        f.write(f"  Min:    {min(metrics['psnr_values']):.2f} dB\n")
        f.write(f"  Max:    {max(metrics['psnr_values']):.2f} dB\n\n")
        f.write(f"SSIM (Structural Similarity Index):\n")
        f.write(f"  Mean:   {metrics['ssim_mean']:.4f}\n")
        f.write(f"  Std:    {metrics['ssim_std']:.4f}\n")
        f.write(f"  Min:    {min(metrics['ssim_values']):.4f}\n")
        f.write(f"  Max:    {max(metrics['ssim_values']):.4f}\n")
        f.write("="*60 + "\n")
    
    print(f"Metrics summary saved to: {summary_path}")


def main():
    """
    Main testing function.
    """
    # Parse arguments
    args = parse_arguments()
    
    print("\n" + "="*60)
    print("TEXT IMAGE DEBLURRING - MODEL TESTING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Test blur dir: {args.test_blur}")
    print(f"  Test orig dir: {args.test_orig}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Image size: {args.image_size}")
    print(f"  Batch size: {args.batch_size}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path)
    
    # Load test data
    print("\n" + "="*60)
    print("LOADING TEST DATA")
    print("="*60)
    
    dataset = DeblurDataset(
        blur_dir=args.test_blur,
        orig_dir=args.test_orig,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size
    )
    
    X_test, y_test = dataset.load_all_images()
    
    if len(X_test) == 0:
        print("\nERROR: No test images found!")
        print("Please ensure test images are available in the specified directories.")
        return
    
    print(f"\nTest data loaded:")
    print(f"  Number of samples: {len(X_test)}")
    print(f"  Shape: {X_test.shape}")
    
    # Run testing
    predictions, metrics = test_model(model, X_test, y_test, args.batch_size)
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    model_name = Path(args.model_path).stem
    
    save_visual_comparisons(X_test, y_test, predictions, metrics, args.output_dir, args.num_visual_samples)
    save_metrics_plot(metrics, args.output_dir)
    save_metrics_summary(metrics, args.output_dir, model_name)
    
    # Save predictions as numpy arrays
    predictions_path = os.path.join(args.output_dir, 'predictions.npz')
    np.savez(predictions_path, 
             blurred=X_test, 
             predictions=predictions, 
             ground_truth=y_test,
             psnr=metrics['psnr_values'],
             ssim=metrics['ssim_values'])
    print(f"Predictions saved to: {predictions_path}")
    
    print("\n" + "="*60)
    print("TESTING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
