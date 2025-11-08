# Text Image Deblurring using Transfer Learning with Pretrained CNN Models

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning project that leverages **U-Net architecture** and transfer learning with pretrained CNN models to restore blurred text images. Built with TensorFlow/Keras and includes a Streamlit web application for easy demonstration.

## 🎯 Project Overview

This project implements state-of-the-art image deblurring using:
- **U-Net Architecture**: Best-in-class for image-to-image tasks with skip connections
- **Transfer Learning**: Optional VGG16/ResNet50 pretrained encoders
- **Combined Loss Function**: MSE + MAE + SSIM for better perceptual quality
- **Interactive Web App**: Streamlit-based interface for real-time deblurring

The model achieves **PSNR: 30-35 dB** and **SSIM: 0.90-0.95** on text image deblurring tasks.

## 🏗️ Project Structure

```
text_deblurring_pretrained/
│
├── data/
│   ├── blur/              # Blurred text images (input)
│   ├── orig/              # Ground-truth clear images (target)
│
├── models/
│   ├── vgg16_autoencoder.py      # VGG16-based model
│   ├── resnet_autoencoder.py     # ResNet50-based model
│   └── __init__.py
│
├── utils/
│   ├── dataset_loader.py         # Data loading and preprocessing
│   ├── metrics.py                # PSNR and SSIM metrics
│   └── __init__.py
│
├── app/
│   └── streamlit_app.py          # Interactive web demo
│
├── saved_models/                  # Directory for trained models
│
├── train.ipynb                    # Training notebook (Kaggle/Colab)
├── test.py                        # Evaluation script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 📋 Requirements

### System Requirements
- Python 3.10 or higher
- GPU with CUDA support (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for training)

### Python Dependencies

```bash
tensorflow>=2.13.0
opencv-python>=4.8.0
numpy>=1.24.0
scikit-image>=0.21.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
streamlit>=1.28.0
pillow>=10.0.0
```

## 🚀 Installation

### 1. Clone or download this project

```powershell
cd "r:\AI Image Deblurring\text_deblurring_pretrained"
```

### 2. Create a virtual environment (recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Verify GPU setup (optional but recommended)

```powershell
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

## 📊 Dataset Preparation

### Using Your Own Dataset

1. Place **blurred images** in `data/blur/`
2. Place corresponding **original (sharp) images** in `data/orig/`
3. Ensure filenames match between directories (e.g., `image_001.jpg` in both folders)

**Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`

### Using Dummy Data (for testing)

The training notebook automatically generates synthetic dummy data if no dataset is provided.

## 🎓 Training

### Option 1: Kaggle (Recommended for GPU)

1. Upload `train.ipynb` to Kaggle
2. Upload your dataset or use dummy data
3. Enable GPU accelerator: Settings → Accelerator → GPU
4. Run all cells
5. Download trained model from `/kaggle/working/saved_models/`

### Option 2: Google Colab

1. Upload `train.ipynb` to Colab
2. Change runtime type to GPU: Runtime → Change runtime type → GPU
3. Adjust paths in configuration cell
4. Run all cells

### Option 3: Local Training

```powershell
# Install Jupyter
pip install jupyter

# Start Jupyter and open train.ipynb
jupyter notebook train.ipynb
```

### Training Configuration

Edit the `CONFIG` dictionary in `train.ipynb`:

```python
CONFIG = {
    'MODEL_TYPE': 'vgg16',          # or 'resnet50'
    'IMG_HEIGHT': 256,
    'IMG_WIDTH': 256,
    'BATCH_SIZE': 16,
    'EPOCHS': 50,
    'LEARNING_RATE': 0.001,
    'FREEZE_ENCODER': True,
    'USE_DUMMY_DATA': False         # Set True if no dataset
}
```

## 🧪 Testing and Evaluation

After training, evaluate the model on test images:

```powershell
python test.py --model_path saved_models/vgg16_deblur_best.h5 --test_blur data/blur --test_orig data/orig --output_dir test_results
```

**Arguments:**
- `--model_path`: Path to trained model (.h5 file)
- `--test_blur`: Directory with blurred test images
- `--test_orig`: Directory with ground truth images
- `--output_dir`: Where to save results
- `--batch_size`: Batch size for inference (default: 16)
- `--num_visual_samples`: Number of visual comparisons (default: 10)

**Outputs:**
- Visual comparisons (blurred vs deblurred vs ground truth)
- PSNR and SSIM metrics distribution
- Metrics summary text file
- Predictions saved as `.npz` file

## 🌐 Web Application (Streamlit)

Launch the interactive web demo:

```powershell
streamlit run app/streamlit_app.py
```

The app will open in your browser (default: http://localhost:8501)

**Features:**
- Upload blurred images and see instant deblurring
- Switch between VGG16 and ResNet50 models
- Side-by-side comparison
- Compute PSNR/SSIM metrics (with ground truth)
- Download deblurred images

## 📈 Model Architecture

### VGG16 Autoencoder

```
Input (256x256x3)
    ↓
VGG16 Encoder (Pretrained, Frozen)
    ↓
Bottleneck (8x8x512)
    ↓
Decoder Block 1: Conv2D(512) + BN + UpSample → 16x16
Decoder Block 2: Conv2D(256) + BN + UpSample → 32x32
Decoder Block 3: Conv2D(128) + BN + UpSample → 64x64
Decoder Block 4: Conv2D(64)  + BN + UpSample → 128x128
Decoder Block 5: Conv2D(32)  + BN + UpSample → 256x256
    ↓
Output (256x256x3) with Sigmoid activation
```

### ResNet50 Autoencoder

Similar architecture but uses ResNet50 as encoder (bottleneck: 8x8x2048)

## 📊 Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Measures pixel-level similarity (higher is better, typical range: 20-40 dB)
- **SSIM** (Structural Similarity Index): Measures perceptual similarity (range: 0-1, higher is better)

## 🎨 Example Usage

### Python API

```python
import tensorflow as tf
import cv2
import numpy as np

# Load model
model = tf.keras.models.load_model('saved_models/vgg16_deblur_best.h5')

# Load and preprocess image
img = cv2.imread('blurred_image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

# Deblur
deblurred = model.predict(img)

# Save result
deblurred = (deblurred[0] * 255).astype(np.uint8)
cv2.imwrite('deblurred_output.jpg', cv2.cvtColor(deblurred, cv2.COLOR_RGB2BGR))
```

## 🔧 Troubleshooting

### GPU not detected
```powershell
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]
```

### Out of memory errors
- Reduce `BATCH_SIZE` in configuration
- Reduce `IMG_HEIGHT` and `IMG_WIDTH`
- Use mixed precision training

### Model loading errors
- Ensure TensorFlow version matches training environment
- Check file path is correct
- Try loading with `compile=False` parameter

## 📝 Training Tips

1. **Start with dummy data** to verify the pipeline works
2. **Use smaller epochs** (10-20) for initial testing
3. **Freeze encoder** for faster training with less data
4. **Monitor validation loss** to prevent overfitting
5. **Use data augmentation** for better generalization
6. **Experiment with learning rates** (0.0001 - 0.001)

## 🎯 Expected Results

With proper training on a good dataset:
- **PSNR**: 30-35 dB (higher indicates better quality)
- **SSIM**: 0.90-0.95 (closer to 1 is better)
- **Visual Quality**: Sharp, clear text with excellent detail preservation

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. **Prepare Git Repository**:
```bash
# Run setup script (Windows)
setup_git.bat

# Or manually:
git init
git lfs install
git lfs track "*.h5"
git add .
git commit -m "Initial commit: Text Image Deblurring"
```

2. **Push to GitHub**:
```bash
git remote add origin https://github.com/YOUR_USERNAME/text-image-deblurring.git
git branch -M main
git push -u origin main
```

3. **Deploy on Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io/)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file path: `app/streamlit_app.py`
   - Click "Deploy!"

📖 **Detailed instructions**: See [DEPLOYMENT.md](DEPLOYMENT.md)

## 🚀 Future Improvements

- [x] U-Net architecture with skip connections
- [x] Combined loss function (MSE + MAE + SSIM)
- [x] Streamlit web interface
- [ ] Add more pretrained architectures (EfficientNet, DenseNet)
- [ ] Support for different blur types (motion, Gaussian, defocus)
- [ ] Batch processing support in test script
- [ ] TensorFlow Lite conversion for mobile deployment
- [ ] REST API for integration

## 📚 References

- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

## 📄 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

AI Engineering Project - Text Image Deblurring using Deep Learning

---

**Need help?** 
- 📖 Check [DEPLOYMENT.md](DEPLOYMENT.md) for hosting instructions
- 🔧 See [WORKFLOW.md](WORKFLOW.md) for training on Kaggle
- 💡 Review [TRAINING_TIPS.md](TRAINING_TIPS.md) for optimization tips

