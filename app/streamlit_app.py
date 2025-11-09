"""
Offline-Ready Text Image Deblurring App
Using Google MAXIM (realblur-j) pretrained model
✅ Works with Keras 3
✅ Runs fully offline after first download
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from huggingface_hub import snapshot_download

# ---------------------- Streamlit Setup ----------------------
st.set_page_config(page_title="Text Image Deblurring", page_icon="🖼️", layout="wide")
st.title("🖼️ Text Image Deblurring (MAXIM - Offline Ready, Keras 3 Compatible)")

st.markdown("""
Upload a blurred **text image** to restore it using
the pretrained **MAXIM (RealBlur-J)** model from Google.

✅ First run: downloads the model  
✅ Later runs: works completely offline
""")

# ---------------------- Paths ----------------------
LOCAL_MODEL_DIR = "saved_models/maxim_offline"
os.makedirs("saved_models", exist_ok=True)

# ---------------------- Model Loading ----------------------
@st.cache_resource
def load_or_download_model():
    """Download once and load locally (Keras 3 compatible)."""
    if os.path.exists(LOCAL_MODEL_DIR):
        st.info("📁 Loading model from local folder (offline mode)...")
    else:
        st.warning("🌐 Model not found locally. Downloading from Hugging Face (first time only)...")
        try:
            snapshot_download(
                repo_id="google/maxim-s3-deblurring-realblur-j",
                local_dir=LOCAL_MODEL_DIR,
                local_dir_use_symlinks=False
            )
            st.success("✅ Model downloaded and saved for offline use!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            return None

    # Wrap TensorFlow SavedModel for Keras 3
    layer = tf.keras.layers.TFSMLayer(LOCAL_MODEL_DIR, call_endpoint="serving_default")
    inp = tf.keras.Input(shape=(256, 256, 3), dtype=tf.float32)
    out = layer(inp)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model


# ---------------------- Image Processing ----------------------
def preprocess_image(image):
    """Resize & normalize image."""
    image = image.convert("RGB")
    image = image.resize((256, 256))
    arr = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(arr, 0)

def postprocess_output(pred):
    """Extract array from dict and convert to a proper PIL image."""
    # Handle dict output from TFSMLayer
    if isinstance(pred, dict):
        pred = list(pred.values())[0]
    elif isinstance(pred, (list, tuple)):
        pred = pred[0]

    # Convert TensorFlow tensor to numpy if needed
    if hasattr(pred, "numpy"):
        pred = pred.numpy()

    # Remove extra batch or channel dimensions safely
    pred = np.squeeze(pred)  # Removes any 1-sized dims like (1,1,256,256,3) → (256,256,3)

    # Ensure shape is (H, W, 3)
    if pred.ndim == 2:
        pred = np.stack([pred] * 3, axis=-1)

    # Clip and convert to uint8
    pred = np.clip(pred, 0, 1)
    pred = (pred * 255).astype(np.uint8)

    return Image.fromarray(pred)


# ---------------------- Main UI ----------------------
model = load_or_download_model()
if model is None:
    st.stop()

uploaded_file = st.file_uploader("📤 Upload a blurred text image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Blurred Input", use_container_width=True)

    if st.button("🔄 Deblur Image", use_container_width=True):
        with st.spinner("Processing... Please wait..."):
            inp = preprocess_image(img)
            pred = model(inp)
            result = postprocess_output(pred)

        st.success("✅ Deblurring completed successfully!")
        st.image(result, caption="Deblurred Output", use_container_width=True)

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label="⬇️ Download Deblurred Image",
            data=buf,
            file_name="deblurred_result.png",
            mime="image/png"
        )
else:
    st.info("""
    👆 Upload a blurred text image to begin.  
    The model downloads once (if not saved) and then runs 100% offline.
    """)

# ---------------------- Footer ----------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & TensorFlow | MAXIM (google/maxim-s3-deblurring-realblur-j)")
