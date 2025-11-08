"""
Streamlit Web App for Text Image Deblurring

This interactive web application allows users to upload blurred text images
and see the deblurred results using the trained model.

Usage:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import calculate_psnr, calculate_ssim


# Page configuration
st.set_page_config(
    page_title="Text Image Deblurring",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
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
        padding: 0.5rem;
        font-size: 1.1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """
    Load the trained model (cached for efficiency).
    
    Args:
        model_path (str): Path to model file
        
    Returns:
        tf.keras.Model: Loaded model
    """
    try:
        # Load model with custom objects to handle Keras 3.x compatibility
        from tensorflow.keras.losses import MeanSquaredError
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'mse': MeanSquaredError()},
            compile=False  # Don't need to compile for inference
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def preprocess_image(image, target_size=(256, 256)):
    """
    Preprocess uploaded image for model inference.
    
    Args:
        image: PIL Image or numpy array
        target_size: Target dimensions
        
    Returns:
        tuple: (processed_image, original_size)
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Store original size
    original_size = image.shape[:2]
    
    # Convert to RGB if needed
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize to model input size
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    return image_normalized, original_size


def postprocess_image(image, original_size=None):
    """
    Postprocess model output back to displayable format.
    
    Args:
        image: Model output (normalized)
        original_size: Original image dimensions
        
    Returns:
        numpy array: Displayable image
    """
    # Clip values to [0, 1]
    image = np.clip(image, 0, 1)
    
    # Resize to original size if specified
    if original_size is not None:
        image = cv2.resize(image, (original_size[1], original_size[0]), 
                          interpolation=cv2.INTER_CUBIC)
    
    # Convert to uint8
    image = (image * 255).astype(np.uint8)
    
    return image


def deblur_image(model, image, target_size=(256, 256)):
    """
    Deblur an image using the trained model.
    
    Args:
        model: Trained Keras model
        image: Input image
        target_size: Model input size
        
    Returns:
        tuple: (deblurred_image, processed_input)
    """
    # Preprocess
    processed_img, original_size = preprocess_image(image, target_size)
    
    # Add batch dimension
    input_batch = np.expand_dims(processed_img, axis=0)
    
    # Predict
    with st.spinner("Deblurring image..."):
        output_batch = model.predict(input_batch, verbose=0)
    
    # Remove batch dimension
    deblurred = output_batch[0]
    
    # Postprocess
    deblurred_display = postprocess_image(deblurred, original_size)
    input_display = postprocess_image(processed_img, original_size)
    
    return deblurred_display, input_display


def main():
    """
    Main Streamlit application.
    """
    # Title and description
    st.title("🖼️ Text Image Deblurring")
    st.markdown("""
    This application uses **Transfer Learning with Pretrained CNN Models** (VGG16/ResNet50) 
    to restore blurred text images. Upload a blurred image to see the results!
    """)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model path (using trained model - automatically detects available models)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check for available models
    saved_models_dir = os.path.join(script_dir, "saved_models")
    available_models = {}
    
    # Check for different model types
    for model_type in ['unet', 'vgg16', 'resnet50']:
        model_file = os.path.join(saved_models_dir, f"{model_type}_deblur_best.h5")
        if os.path.exists(model_file):
            available_models[model_type.upper()] = model_file
    
    if not available_models:
        st.sidebar.error("⚠️ No trained models found!")
        st.error("""
        **No trained models found!**
        
        Please train a model using the Kaggle notebook and save it to `saved_models/` directory.
        
        Expected filenames:
        - `unet_deblur_best.h5` (recommended)
        - `vgg16_deblur_best.h5`
        - `resnet50_deblur_best.h5`
        """)
        return
    
    # Model selection (if multiple models available)
    if len(available_models) > 1:
        selected_model_name = st.sidebar.selectbox(
            "Select Model",
            options=list(available_models.keys())
        )
        model_path = available_models[selected_model_name]
        st.sidebar.info(f"Using **{selected_model_name}** Model")
    else:
        selected_model_name = list(available_models.keys())[0]
        model_path = available_models[selected_model_name]
        st.sidebar.info(f"Using **{selected_model_name}** Model")
    
    # Image size
    img_size = st.sidebar.slider(
        "Model Input Size",
        min_value=128,
        max_value=512,
        value=256,
        step=64,
        help="Size used by the model (will resize automatically)"
    )
    
    # Additional options
    st.sidebar.markdown("---")
    show_comparison = st.sidebar.checkbox("Show Side-by-Side Comparison", value=True)
    compute_metrics = st.sidebar.checkbox("Compute Quality Metrics", value=False,
                                         help="Only if ground truth is available")
    
    # Load model
    model = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model. Please check the model file.")
        return
    
    st.sidebar.success(f"✅ {selected_model_name} Model Loaded Successfully")
    
    # Show model info
    with st.sidebar.expander("ℹ️ Model Information"):
        st.write(f"**Architecture:** {selected_model_name}")
        st.write(f"**Input Size:** {img_size}x{img_size}")
        st.write(f"**Parameters:** ~{model.count_params():,}")


    
    # File uploader
    st.markdown("---")
    st.subheader("📤 Upload Blurred Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a blurred text image"
    )
    
    # Sample images option
    use_sample = st.checkbox("Or use a sample blurred image")
    
    if use_sample:
        sample_dir = os.path.join(script_dir, "data", "blur")
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if sample_files:
                selected_sample = st.selectbox("Select sample image", sample_files)
                uploaded_file = os.path.join(sample_dir, selected_sample)
    
    # Process image
    if uploaded_file is not None:
        try:
            # Load image
            if isinstance(uploaded_file, str):
                # Sample image path
                image = cv2.imread(uploaded_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Uploaded file
                image = Image.open(uploaded_file)
                image = np.array(image)
            
            # Display original
            st.markdown("---")
            st.subheader("📷 Input Image")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Blurred Input", use_container_width=True)
            
            # Deblur button
            if st.button("🚀 Deblur Image", key="deblur_btn"):
                # Deblur
                deblurred, processed_input = deblur_image(model, image, (img_size, img_size))
                
                # Display results
                st.markdown("---")
                st.subheader("✨ Results")
                
                if show_comparison:
                    # Side-by-side comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Blurred Input**")
                        st.image(processed_input, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Deblurred Output**")
                        st.image(deblurred, use_container_width=True)
                else:
                    # Only show output
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(deblurred, caption="Deblurred Output", use_container_width=True)
                
                # Compute metrics if ground truth is available
                if compute_metrics:
                    st.markdown("---")
                    st.subheader("📊 Quality Metrics")
                    
                    # File uploader for ground truth
                    gt_file = st.file_uploader(
                        "Upload ground truth (original) image for comparison",
                        type=['jpg', 'jpeg', 'png', 'bmp'],
                        key="gt_uploader"
                    )
                    
                    if gt_file is not None:
                        gt_image = Image.open(gt_file)
                        gt_image = np.array(gt_image)
                        
                        # Preprocess ground truth
                        gt_processed, _ = preprocess_image(gt_image, (img_size, img_size))
                        deblurred_normalized = deblurred.astype(np.float32) / 255.0
                        
                        # Calculate metrics
                        psnr_val = calculate_psnr(gt_processed, deblurred_normalized)
                        ssim_val = calculate_ssim(gt_processed, deblurred_normalized)
                        
                        # Display metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                label="PSNR (Peak Signal-to-Noise Ratio)",
                                value=f"{psnr_val:.2f} dB",
                                help="Higher is better (typically 20-40 dB)"
                            )
                        
                        with col2:
                            st.metric(
                                label="SSIM (Structural Similarity Index)",
                                value=f"{ssim_val:.4f}",
                                help="Range: 0-1, higher is better"
                            )
                
                # Download button
                st.markdown("---")
                st.subheader("💾 Download Result")
                
                # Convert to bytes for download
                deblurred_pil = Image.fromarray(deblurred)
                
                # Create download button
                from io import BytesIO
                buf = BytesIO()
                deblurred_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Deblurred Image",
                    data=byte_im,
                    file_name="deblurred_output.png",
                    mime="image/png"
                )
        
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.exception(e)
    
    else:
        # Instructions
        st.info("""
        👆 **Upload a blurred text image** or use a sample to get started.
        
        **Tips for best results:**
        - Use images with clear text content
        - Avoid extremely low resolution images
        - Images will be resized to the model input size
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit and TensorFlow | Transfer Learning with Pretrained CNNs</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
