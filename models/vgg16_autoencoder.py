"""
VGG16-based Autoencoder for Text Image Deblurring

This module implements an autoencoder architecture using pretrained VGG16 as the encoder
and custom upsampling layers as the decoder for image deblurring tasks.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16


def build_vgg16_autoencoder(input_shape=(256, 256, 3), freeze_encoder=True):
    """
    Build a VGG16-based autoencoder for image deblurring.
    
    Architecture:
    - Encoder: Pretrained VGG16 (blocks 1-5) with frozen weights
    - Decoder: Upsampling layers with Conv2D to reconstruct the image
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        freeze_encoder (bool): Whether to freeze VGG16 encoder weights
        
    Returns:
        tf.keras.Model: Compiled autoencoder model
    """
    
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # ============ ENCODER (Pretrained VGG16) ============
    # Load VGG16 without top classification layers, pretrained on ImageNet
    vgg16_base = VGG16(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    
    # Freeze encoder weights if specified
    if freeze_encoder:
        for layer in vgg16_base.layers:
            layer.trainable = False
    
    # Get encoder output (bottleneck features)
    encoder_output = vgg16_base.output  # Shape: (None, 8, 8, 512) for 256x256 input
    
    # ============ DECODER (Upsampling Path) ============
    # Decoder Block 1: Upsample from 8x8 to 16x16
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_conv1')(encoder_output)
    x = layers.BatchNormalization(name='decoder_bn1')(x)
    x = layers.UpSampling2D((2, 2), name='upsample1')(x)  # 16x16
    
    # Decoder Block 2: Upsample from 16x16 to 32x32
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_conv2')(x)
    x = layers.BatchNormalization(name='decoder_bn2')(x)
    x = layers.UpSampling2D((2, 2), name='upsample2')(x)  # 32x32
    
    # Decoder Block 3: Upsample from 32x32 to 64x64
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='decoder_conv3')(x)
    x = layers.BatchNormalization(name='decoder_bn3')(x)
    x = layers.UpSampling2D((2, 2), name='upsample3')(x)  # 64x64
    
    # Decoder Block 4: Upsample from 64x64 to 128x128
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='decoder_conv4')(x)
    x = layers.BatchNormalization(name='decoder_bn4')(x)
    x = layers.UpSampling2D((2, 2), name='upsample4')(x)  # 128x128
    
    # Decoder Block 5: Upsample from 128x128 to 256x256
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='decoder_conv5')(x)
    x = layers.BatchNormalization(name='decoder_bn5')(x)
    x = layers.UpSampling2D((2, 2), name='upsample5')(x)  # 256x256
    
    # Output layer: Reconstruct RGB image with sigmoid activation
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='output_image')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='VGG16_Autoencoder')
    
    return model


def compile_vgg16_model(model, learning_rate=0.001):
    """
    Compile the VGG16 autoencoder model with optimizer and loss function.
    
    Args:
        model: Keras model to compile
        learning_rate (float): Learning rate for Adam optimizer
        
    Returns:
        Compiled Keras model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',  # Mean Squared Error for pixel-wise reconstruction
        metrics=['mae']  # Mean Absolute Error as additional metric
    )
    return model


if __name__ == "__main__":
    # Test model creation
    print("Building VGG16 Autoencoder...")
    model = build_vgg16_autoencoder(input_shape=(256, 256, 3))
    model = compile_vgg16_model(model)
    
    print("\n" + "="*60)
    print("Model Summary:")
    print("="*60)
    model.summary()
    
    print("\n" + "="*60)
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    print("="*60)
