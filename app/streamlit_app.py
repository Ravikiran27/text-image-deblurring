"""
Text Image Deblurring Application
Using Transfer Learning with Pretrained CNN Models (VGG16/ResNet50)
Based on the training notebook approach
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import io

# Page configuration
st.set_page_config(
    page_title="Text Image Deblurring",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Load the trained model (same as notebook)."""
    try:
        from tensorflow.keras.losses import MeanSquaredError
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'mse': MeanSquaredError()},
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def load_and_preprocess_image(image, target_size=(256, 256)):
    """
    Load and preprocess image - identical to notebook approach.
    """
    # Convert PIL to numpy
    img = np.array(image)
    
    # Convert to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Store original size
    original_size = img.shape[:2]
    
    # Resize to model input size
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return img_normalized, original_size


def deblur_image(model, image, target_size=(256, 256)):
    """
    Deblur an image - exact same logic as notebook's deblur_image function.
    """
    # Load and preprocess
    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    original_size = img.shape[:2]
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img, axis=0)
    
    # Predict
    deblurred = model.predict(img_batch, verbose=0)
    
    # Remove batch dimension
    deblurred = deblurred[0]
    
    return img, deblurred


def calculate_metrics(img1, img2):
    """Calculate PSNR and SSIM - same as notebook."""
    try:
        # Ensure float32 and normalized
        if img1.max() > 1.0:
            img1 = img1.astype(np.float32) / 255.0
        if img2.max() > 1.0:
            img2 = img2.astype(np.float32) / 255.0
        
        # Calculate PSNR
        psnr_val = psnr(img1, img2, data_range=1.0)
        
        # Calculate SSIM
        ssim_val = ssim(img1, img2, data_range=1.0, channel_axis=2)
        
        return psnr_val, ssim_val
    except Exception as e:
        return None, None


def main():
    """Main Streamlit application."""
    
    # Title
    st.title("🖼️ Text Image Deblurring")
    st.markdown("""
    This application uses **MAXIM (Multi-Axis MLP)** and other deep learning models 
    to restore blurred text images with state-of-the-art performance.
    """)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Find available models
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    saved_models_dir = os.path.join(script_dir, "saved_models")
    
    available_models = {}
    # Prioritize MAXIM as the main model
    for model_type in ['maxim', 'vgg16', 'resnet50', 'unet']:
        model_file = os.path.join(saved_models_dir, f"{model_type}_deblur_best.h5")
        if os.path.exists(model_file):
            display_name = f"MAXIM (Recommended)" if model_type == 'maxim' else f"{model_type.upper()}"
            available_models[display_name] = model_file
    
    if not available_models:
        st.error("❌ No trained models found! Please train a model first.")
        st.info("Run the training notebook on Kaggle to create a model.")
        return
    
    # Model selection - default to first (MAXIM)
    model_choice = st.sidebar.selectbox(
        "Select Model",
        options=list(available_models.keys()),
        index=0
    )
    
    model_path = available_models[model_choice]
    
    # Load model
    model = load_model(model_path)
    if model is None:
        st.error("Failed to load model!")
        return
    
    st.sidebar.success("✅ Model Loaded Successfully")
    
    # Show model info
    if 'MAXIM' in model_choice:
        st.sidebar.info("""
        **MAXIM Model**
        - State-of-the-art architecture
        - Multi-scale processing
        - Input: 256x256 (fixed)
        """)
    else:
        st.sidebar.info("**Model Input Size:** 256x256 (fixed)")
    
    # Options
    show_comparison = st.sidebar.checkbox("Show Side-by-Side Comparison", value=True)
    compute_metrics = st.sidebar.checkbox("Compute Quality Metrics", value=False,
                                         help="Calculate PSNR and SSIM")
    
    # File uploader
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a blurred text image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Deblur button
        if st.button("🔄 Deblur Image", type="primary"):
            try:
                # Deblur - exact notebook logic
                blurred, deblurred = deblur_image(model, image, target_size=(256, 256))
                
                # Store results
                st.session_state['blurred'] = blurred
                st.session_state['deblurred'] = deblurred
                st.session_state['success'] = True
                
                st.success("✅ Deblurring completed!")
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state['success'] = False
        
        # Display results
        if st.session_state.get('success', False):
            st.markdown("### ✨ Results")
            
            if show_comparison:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Blurred Input")
                    # Convert to uint8 for display
                    blurred_display = (st.session_state['blurred'] * 255).astype(np.uint8)
                    st.image(blurred_display, use_column_width=True)
                
                with col2:
                    st.markdown("#### Deblurred Output")
                    # Convert to uint8 for display
                    deblurred_display = (st.session_state['deblurred'] * 255).astype(np.uint8)
                    st.image(deblurred_display, use_column_width=True)
            else:
                st.markdown("#### Deblurred Output")
                deblurred_display = (st.session_state['deblurred'] * 255).astype(np.uint8)
                st.image(deblurred_display, use_column_width=True)
            
            # Calculate metrics if requested
            if compute_metrics:
                st.markdown("### 📊 Quality Metrics")
                
                psnr_val, ssim_val = calculate_metrics(
                    st.session_state['blurred'], 
                    st.session_state['deblurred']
                )
                
                if psnr_val is not None:
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("PSNR (Peak Signal-to-Noise Ratio)", f"{psnr_val:.2f} dB")
                    with col_m2:
                        st.metric("SSIM (Structural Similarity)", f"{ssim_val:.4f}")
                    
                    st.info("""
                    **Interpreting metrics:**
                    - PSNR > 30 dB: Good quality
                    - SSIM > 0.90: Excellent similarity
                    """)
            
            # Download button
            st.markdown("### 💾 Download Result")
            
            # Convert to PIL for download
            deblurred_uint8 = (st.session_state['deblurred'] * 255).astype(np.uint8)
            result_pil = Image.fromarray(deblurred_uint8)
            
            buf = io.BytesIO()
            result_pil.save(buf, format='PNG')
            buf.seek(0)
            
            st.download_button(
                label="⬇️ Download Deblurred Image",
                data=buf,
                file_name="deblurred_result.png",
                mime="image/png"
            )
    
    else:
        st.info("""
        👆 **Upload a blurred text image** to get started!
        
        **How it works:**
        1. Upload your blurred image
        2. Click "Deblur Image"
        3. View and download the result
        
        **Supported formats:** PNG, JPG, JPEG, BMP, TIFF
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit | Powered by TensorFlow</p>
        <p>Using Transfer Learning with VGG16/ResNet50</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
