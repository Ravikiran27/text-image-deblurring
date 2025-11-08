# 🎯 Quick Reference - Improved Model Configuration# 🎯 Quick Reference - Kaggle Training



## What's New in Your Notebook## ⚡ Fast Setup (5 Minutes)



### ✨ Key Changes:### On Kaggle:



1. **NEW U-Net Architecture**1. **Add Dataset**

   - Best model for image deblurring   - Click "+ Add Input"

   - Skip connections preserve details   - Search: `text-deblurring-dataset-with-psf-for-ocr`

   - Expected PSNR: 30-35 dB (vs. 20-25 dB before)   - Click "Add"



2. **Better Loss Function**2. **Enable GPU**

   - Combined: MSE + MAE + SSIM   - Session Options → Accelerator → GPU T4 x2

   - Produces sharper, more realistic images   - Click "Save"



3. **Optimized Training**3. **Upload Notebook**

   - 100 epochs (was 50)   - Upload your `train.ipynb`

   - Learning rate: 0.0001 (was 0.001)   - Or create new notebook and copy cells

   - Batch size: 8 (was 16)

   - All layers trainable (encoder not frozen)4. **Run**

   - Click "Run All" ▶️

---   - Wait 2-4 hours

   - Download model from Output section

## 📊 What to Expect

---

### Training Output:

```## 📋 Configuration At A Glance

Epoch 50/100

loss: 0.0123 - mae: 0.0234 - val_loss: 0.0145### Already Configured For You:



✅ Good if loss < 0.02```python

✅ Good if decreasing steadily✅ Dataset paths: Auto-detected

```✅ Model: VGG16 autoencoder  

✅ Image size: 256×256

### Final Metrics:✅ Batch size: 16

```✅ Epochs: 50 (with early stopping)

PSNR: 32.45 dB  ← Target: > 30 dB✅ GPU: Auto-detected

SSIM: 0.9234    ← Target: > 0.90```

```

### What You Can Change:

---

```python

## 🚀 Upload to Kaggle Steps# In Configuration cell:



1. **Upload** `kagel-train (1).ipynb` to Kaggle# Try different model

2. **Add dataset**: `text-deblurring-dataset-with-psf-for-ocr`'MODEL_TYPE': 'resnet50'  # instead of 'vgg16'

3. **Enable GPU**: P100 or T4

4. **Run All** and wait ~1-2 hours# Reduce memory usage

5. **Download**: `unet_deblur_best.h5` from Output'BATCH_SIZE': 8  # instead of 16

6. **Copy** to local `saved_models/` folder'IMG_HEIGHT': 128  # instead of 256

7. **Run** Streamlit app to test!'IMG_WIDTH': 128



---# Train longer/shorter

'EPOCHS': 100  # instead of 50

## 🎨 Model Files```



After training, you'll have:---

- `unet_deblur_best.h5` - Best model during training ⭐

- `unet_deblur_final.h5` - Final model## 🎯 Expected Results

- `training_history.png` - Loss curves

- `deblur_results.png` - Sample outputs| Metric | Target | Your Result |

- `metrics_summary.json` - Performance stats|--------|--------|-------------|

| PSNR | > 25 dB | _______ dB |

---| SSIM | > 0.85 | _______ |

| Training Time | 2-4 hours | _______ |

## ✅ Success Indicators

---

Your model is PERFECT if:

- ✅ PSNR > 30 dB## 🆘 Quick Fixes

- ✅ SSIM > 0.90

- ✅ Deblurred images look sharp and clear### "Out of Memory"

- ✅ Text is readable after deblurring→ Change `'BATCH_SIZE': 8`



---### "Dataset not found"  

→ Check "+ Add Input" is clicked

**The improved notebook will give you MUCH better results! 🎉**

### "Training too slow"
→ Verify GPU is enabled

### "Poor results"
→ Train longer: `'EPOCHS': 100`

---

## 📥 After Training

1. Download: `saved_models/vgg16_deblur_best.h5`
2. Save to: Local `saved_models/` folder
3. Test: `python test.py --model_path saved_models/vgg16_deblur_best.h5`
4. Demo: `streamlit run app/streamlit_app.py`

---

## 📊 Files Generated

Kaggle will create:
```
/kaggle/working/
├── saved_models/
│   ├── vgg16_deblur_best.h5      ← DOWNLOAD THIS
│   ├── vgg16_deblur_final.h5
│   └── vgg16_savedmodel/
├── sample_data.png
├── training_history.png           ← DOWNLOAD THIS
├── deblur_results.png            ← DOWNLOAD THIS
└── metrics_summary.json
```

Download the files you need!

---

## ✅ Checklist

Before running:
- [ ] Dataset added to notebook
- [ ] GPU enabled
- [ ] Paths verified in config

While running:
- [ ] Monitor loss decreasing
- [ ] Check no errors
- [ ] Wait for completion message

After running:
- [ ] Download .h5 model file
- [ ] Download visualization plots
- [ ] Note final PSNR/SSIM values
- [ ] Test locally

---

**Ready to train! Just run all cells.** 🚀
