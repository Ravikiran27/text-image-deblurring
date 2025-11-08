# 🚀 Quick Start - Git & Streamlit Deployment

## ⚡ Fast Track (5 Minutes)

### Step 1: Initialize Git (30 seconds)
```bash
cd "r:\AI Image Deblurring\text_deblurring_pretrained"
.\setup_git.bat
```

### Step 2: Create GitHub Repository (1 minute)
1. Go to https://github.com/new
2. Name: `text-image-deblurring`
3. Public repository
4. Click "Create repository"

### Step 3: Push to GitHub (1 minute)
```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/text-image-deblurring.git
git branch -M main
git push -u origin main
```

### Step 4: Deploy on Streamlit (2 minutes)
1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `YOUR_USERNAME/text-image-deblurring`
5. Main file: `app/streamlit_app.py`
6. Click "Deploy!"

### Step 5: Wait & Test (2-5 minutes)
- Deployment takes 2-5 minutes
- Your app will be at: `https://your-app-name.streamlit.app`
- Upload a blurry image and test!

---

## 🎯 That's it! Your app is live!

Share your URL:
```
https://your-app-name.streamlit.app
```

---

## 📋 Checklist

Before deploying, make sure:
- [x] `requirements.txt` exists
- [x] `.streamlit/config.toml` exists  
- [x] `.gitattributes` exists (for Git LFS)
- [x] `saved_models/unet_deblur_best.h5` exists (your trained model)
- [x] `app/streamlit_app.py` exists

---

## 🐛 Common Issues

### "Model not found" error
**Fix**: Make sure `saved_models/unet_deblur_best.h5` is committed
```bash
git add saved_models/unet_deblur_best.h5
git commit -m "Add trained model"
git push
```

### "File too large" error (model > 100MB)
**Fix**: Git LFS should handle this automatically
```bash
git lfs track "*.h5"
git add .gitattributes
git add saved_models/unet_deblur_best.h5
git commit -m "Add model with LFS"
git push
```

### "Module not found" error
**Fix**: Check requirements.txt has all dependencies
```bash
streamlit>=1.28.0
tensorflow>=2.13.0
opencv-python-headless>=4.8.0
```

---

## 🎉 Success!

Once deployed, your AI deblurring app is:
- ✅ Live and accessible worldwide
- ✅ Free hosting on Streamlit Cloud
- ✅ Automatic updates when you push to GitHub
- ✅ Perfect for portfolio/demos

---

**Need detailed help?** See [DEPLOYMENT.md](DEPLOYMENT.md)
