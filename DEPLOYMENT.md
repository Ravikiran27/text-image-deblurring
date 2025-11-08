# 🖼️ Text Image Deblurring - Deployment Guide

## 🚀 Deploy to Streamlit Cloud

### Prerequisites
1. GitHub account
2. Trained model file (`unet_deblur_best.h5`) in `saved_models/` folder
3. Model file must be < 100MB (or use Git LFS)

---

## 📦 Step 1: Prepare Repository

### Check Model Size
```bash
# If model > 100MB, you need Git LFS
ls -lh saved_models/unet_deblur_best.h5
```

### Install Git LFS (if needed)
```bash
# Windows (with Git for Windows)
git lfs install

# Track large files
git lfs track "*.h5"
git add .gitattributes
```

---

## 🔧 Step 2: Initialize Git Repository

```bash
cd "r:\AI Image Deblurring\text_deblurring_pretrained"

# Initialize git
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: Text Image Deblurring with U-Net"
```

---

## 🌐 Step 3: Push to GitHub

### Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `text-image-deblurring`
3. Description: `AI-powered text image deblurring using U-Net and Transfer Learning`
4. Make it **Public**
5. Don't initialize with README (we already have one)
6. Click **Create repository**

### Push Code
```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/text-image-deblurring.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## ☁️ Step 4: Deploy on Streamlit Cloud

### 1. Go to Streamlit Cloud
Visit: https://share.streamlit.io/

### 2. Sign in with GitHub

### 3. Deploy New App
- Click **"New app"**
- **Repository:** Select `YOUR_USERNAME/text-image-deblurring`
- **Branch:** `main`
- **Main file path:** `app/streamlit_app.py`
- **App URL:** Choose your custom URL (e.g., `text-deblurring`)

### 4. Advanced Settings (Optional)
- **Python version:** 3.10
- **Secrets:** Not needed for this app

### 5. Deploy!
Click **"Deploy!"** and wait 2-5 minutes

---

## 📝 Important Notes

### Model File Size
- GitHub free tier limit: **100MB per file**
- Your model (`unet_deblur_best.h5`) is ~101MB
- **Solution:** Use Git LFS or compress model

### Git LFS Setup
```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "saved_models/*.h5"

# Add and commit
git add .gitattributes
git add saved_models/unet_deblur_best.h5
git commit -m "Add model with Git LFS"
git push
```

### Alternative: Model Compression
```python
# In Python, optimize model
import tensorflow as tf
model = tf.keras.models.load_model('saved_models/unet_deblur_best.h5')
model.save('saved_models/unet_deblur_best.h5', 
           save_format='h5', 
           include_optimizer=False)  # Remove optimizer to reduce size
```

---

## 🔍 Verify Deployment

Once deployed, your app will be at:
```
https://YOUR_APP_NAME.streamlit.app
```

Test the app:
1. ✅ Model loads successfully
2. ✅ Upload image works
3. ✅ Deblur button functions
4. ✅ Download result works

---

## 🐛 Troubleshooting

### Error: "Model not found"
**Solution:** Make sure `saved_models/unet_deblur_best.h5` is committed to git

### Error: "File too large"
**Solution:** Use Git LFS or compress model (remove optimizer)

### Error: "Module not found"
**Solution:** Check `requirements.txt` has all dependencies

### Error: "Out of memory"
**Solution:** Streamlit Cloud has 1GB RAM limit. Model might be too large.
- Try smaller image sizes
- Use model quantization

---

## 📊 App Features Live

Your deployed app will have:
- ✅ Image upload (JPG, PNG, BMP)
- ✅ Real-time deblurring
- ✅ Side-by-side comparison
- ✅ Download deblurred images
- ✅ Adjustable model input size
- ✅ Quality metrics (optional)

---

## 🎯 Quick Deploy Checklist

- [ ] Git initialized
- [ ] Model file in `saved_models/`
- [ ] Git LFS configured (if model > 100MB)
- [ ] All files committed
- [ ] Pushed to GitHub
- [ ] Streamlit Cloud account created
- [ ] App deployed
- [ ] Tested live URL

---

## 🔗 Useful Links

- **Streamlit Docs:** https://docs.streamlit.io
- **Git LFS:** https://git-lfs.github.com
- **GitHub:** https://github.com

---

## 📧 Share Your App!

Once deployed, share your app URL:
```
https://your-app-name.streamlit.app
```

Perfect for:
- Portfolio projects
- Research demonstrations
- Client presentations
- Open source contributions

🎉 **Enjoy your deployed AI deblurring app!**
