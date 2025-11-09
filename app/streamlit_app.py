"""
Offline-Ready Text Image Deblurring App
Using Google MAXIM (realblur-j) pretrained model
✅ Works with Keras 3
✅ Runs fully offline after first download
✅ State-of-the-art deblurring performance
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from huggingface_hub import snapshot_download

# ---------------------- Streamlit Setup ----------------------
st.set_page_config(
    page_title="Text Image Deblurring", 
    page_icon="🖼️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.75rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🖼️ Text Image Deblurring with MAXIM")
st.markdown("""
Upload a blurred **text image** to restore it using the pretrained **MAXIM (RealBlur-J)** model from Google Research.

✅ **State-of-the-art performance** | ✅ **First run downloads model** | ✅ **Then works offline**
""")

# ---------------------- Paths ----------------------
LOCAL_MODEL_DIR = "saved_models/maxim_offline"
os.makedirs("saved_models", exist_ok=True)

# ---------------------- Model Loading ----------------------
@st.cache_resource
def load_or_download_model():
    """Download MAXIM model once and load locally (Keras 3 compatible)."""
    if os.path.exists(LOCAL_MODEL_DIR):
        st.sidebar.success("✅ Model loaded from local storage (offline mode)")
    else:
        st.sidebar.warning("🌐 Downloading MAXIM model from Hugging Face...")
        st.sidebar.info("This is a one-time download (~100MB). Please wait...")
        try:
            with st.spinner("Downloading model... This may take a few minutes..."):
                snapshot_download(
                    repo_id="google/maxim-s3-deblurring-realblur-j",
                    local_dir=LOCAL_MODEL_DIR,
                    local_dir_use_symlinks=False
                )
            st.sidebar.success("✅ Model downloaded! Now running offline.")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.info("Please check your internet connection and try again.")
            return None

    try:
        # Wrap TensorFlow SavedModel for Keras 3
        layer = tf.keras.layers.TFSMLayer(LOCAL_MODEL_DIR, call_endpoint="serving_default")
        inp = tf.keras.Input(shape=(256, 256, 3), dtype=tf.float32)
        out = layer(inp)
        model = tf.keras.Model(inputs=inp, outputs=out)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


# ---------------------- Image Processing ----------------------
def preprocess_image(image, target_size=(256, 256)):
    """Resize & normalize image for MAXIM model."""
    # Store original size for later
    original_size = image.size
    
    # Convert to RGB and resize
    image = image.convert("RGB")
    image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize to [0, 1]
    arr = np.array(image_resized).astype(np.float32) / 255.0
    
    return np.expand_dims(arr, 0), original_size

def postprocess_output(pred, original_size):
    """Extract array from dict, resize to original size, and convert to PIL image."""
    # Handle dict output from TFSMLayer
    if isinstance(pred, dict):
        pred = list(pred.values())[0]
    elif isinstance(pred, (list, tuple)):
        pred = pred[0]

    # Convert TensorFlow tensor to numpy if needed
    if hasattr(pred, "numpy"):
        pred = pred.numpy()

    # Remove extra batch or channel dimensions
    pred = np.squeeze(pred)

    # Ensure shape is (H, W, 3)
    if pred.ndim == 2:
        pred = np.stack([pred] * 3, axis=-1)

    # Clip and convert to uint8
    pred = np.clip(pred, 0, 1)
    pred = (pred * 255).astype(np.uint8)
    
    # Convert to PIL and resize back to original size
    result = Image.fromarray(pred)
    if original_size != (256, 256):
        result = result.resize(original_size, Image.Resampling.LANCZOS)

    return result


# ---------------------- Main UI ----------------------
# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("""
**MAXIM Model**
- Google Research state-of-the-art
- Trained on RealBlur-J dataset
- Multi-axis processing
- Input: 256×256 (auto-resized)
""")

model = load_or_download_model()
if model is None:
    st.stop()

st.sidebar.markdown("---")
show_comparison = st.sidebar.checkbox("Show Side-by-Side Comparison", value=True)

# Main content
st.markdown("### 📤 Upload Image")
uploaded_file = st.file_uploader(
    "Choose a blurred text image",
    type=["png", "jpg", "jpeg", "bmp"],
    help="Upload an image to deblur"
)

if uploaded_file:
    img = Image.open(uploaded_file)
    
    if show_comparison:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Blurred Input")
            st.image(img, use_container_width=True)
    else:
        st.image(img, caption="Blurred Input", use_container_width=True)

    if st.button("🔄 Deblur Image", type="primary"):
        with st.spinner("Processing with MAXIM... Please wait..."):
            try:
                inp, original_size = preprocess_image(img)
                pred = model(inp)
                result = postprocess_output(pred, original_size)
                
                # Store in session state
                st.session_state['result'] = result
                st.session_state['processed'] = True
                
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")
                st.session_state['processed'] = False

    # Display results
    if st.session_state.get('processed', False):
        st.success("✅ Deblurring completed successfully!")
        
        if show_comparison:
            with col2:
                st.markdown("#### Deblurred Output")
                st.image(st.session_state['result'], use_container_width=True)
        else:
            st.markdown("### ✨ Result")
            st.image(st.session_state['result'], caption="Deblurred Output", use_container_width=True)
        
        # Download button
        st.markdown("### 💾 Download")
        buf = io.BytesIO()
        st.session_state['result'].save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label="⬇️ Download Deblurred Image",
            data=buf,
            file_name="deblurred_maxim.png",
            mime="image/png",
            use_container_width=True
        )
else:
    st.info("""
    👆 **Upload a blurred text image to begin**
    
    **How it works:**
    1. Upload your blurred image
    2. Click "Deblur Image"
    3. Wait for processing (~5-10 seconds)
    4. Download the result
    
    **Supported formats:** PNG, JPG, JPEG, BMP
    """)

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit & TensorFlow</p>
    <p>Powered by <strong>MAXIM</strong> (google/maxim-s3-deblurring-realblur-j)</p>
    <p><a href="https://github.com/Ravikiran27/text-image-deblurring" target="_blank">View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
