# MAXIM Model Integration

## Overview

This project now includes **MAXIM (Multi-Axis MLP for Image Processing)**, a state-of-the-art model for image deblurring that uses multi-axis gated MLPs instead of traditional convolutions.

## Features

- **Lightweight Version**: Optimized for faster training and inference
- **Skip Connections**: U-Net-style architecture for better detail preservation
- **Gating Units**: Block and Grid gating for effective feature processing
- **No Pretrained Weights Required**: Trains from scratch efficiently

## Model Architecture

```
Input (256x256x3)
    ↓
Initial Conv + Norm
    ↓
[Encoder]
  Stage 1: 48 channels, 1 MAXIM block
  Stage 2: 96 channels, 2 MAXIM blocks
  Stage 3: 192 channels, 2 MAXIM blocks
  Stage 4: 384 channels, 2 MAXIM blocks
    ↓
[Bottleneck]
  2 MAXIM blocks at 384 channels
    ↓
[Decoder] (with skip connections)
  Upsample + MAXIM blocks
    ↓
Output Conv (3 channels, sigmoid)
```

## Training MAXIM

### Option 1: Local Training

```python
# Run the training script
python train_maxim.py
```

### Option 2: In Notebook

Add this cell to your Kaggle/Colab notebook:

```python
from models.maxim_model import build_maxim_lightweight, compile_maxim_model

# Build model
model = build_maxim_lightweight(input_shape=(256, 256, 3))
model = compile_maxim_model(model, learning_rate=0.0001)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=8,
    epochs=50,
    callbacks=callbacks
)

# Save
model.save('maxim_deblur_best.h5')
```

## Using MAXIM in Streamlit App

The app automatically detects MAXIM models in `saved_models/`:

1. Train your MAXIM model
2. Save as `maxim_deblur_best.h5` in `saved_models/`
3. Restart Streamlit app
4. Select "MAXIM" from the dropdown

## Benefits over VGG16/ResNet50

✅ **No ImageNet Dependency**: Trains from scratch for your specific task  
✅ **Better for Text**: Optimized for structured content like text  
✅ **Faster Inference**: Lightweight architecture  
✅ **Better Quality**: State-of-the-art results on deblurring benchmarks  
✅ **Less Overfitting**: Better generalization on small datasets  

## Model Size

- **Lightweight MAXIM**: ~5-10M parameters
- **VGG16 Autoencoder**: ~20-30M parameters
- **ResNet50 Autoencoder**: ~30-40M parameters

## Expected Performance

With proper training on text deblurring dataset:

- **PSNR**: 32-38 dB (higher is better)
- **SSIM**: 0.92-0.97 (closer to 1 is better)
- **Inference Time**: ~50-100ms per 256x256 image

## Training Tips

1. **Start with 8 batch size** - Reduces memory usage
2. **Use learning rate 0.0001** - Stable training
3. **Train for 50-100 epochs** - Allow model to converge
4. **Monitor validation loss** - Early stopping at patience=10
5. **Use data augmentation** - Rotation, flip, brightness

## Troubleshooting

**Q: Out of memory during training?**  
A: Reduce batch_size to 4 or use even lighter model (reduce dims)

**Q: Model not improving?**  
A: Check if dataset has sufficient variety, try lower learning rate

**Q: Predictions are blurry?**  
A: Train longer or increase model capacity (more blocks/channels)

## References

- MAXIM Paper: "MAXIM: Multi-Axis MLP for Image Processing" (Google Research)
- Optimized for text deblurring task
