"""
MAXIM (Multi-Axis MLP) Model for Image Deblurring
Based on Google Research's MAXIM architecture
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def window_partition(x, window_size):
    """Partition image into windows."""
    B, H, W, C = x.shape
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    """Reverse window partition."""
    B = tf.shape(windows)[0] // (H * W // window_size // window_size)
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, C])
    return x


class BlockGatingUnit(layers.Layer):
    """Block Gating Unit for MAXIM."""
    
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.norm = layers.LayerNormalization()
        self.dense1 = layers.Dense(dim)
        self.dense2 = layers.Dense(dim)
        self.act = layers.Activation('gelu')
        
    def call(self, x):
        shortcut = x
        x = self.norm(x)
        x1, x2 = tf.split(x, 2, axis=-1)
        x = self.dense1(x1) * self.act(self.dense2(x2))
        return x + shortcut


class GridGatingUnit(layers.Layer):
    """Grid Gating Unit for MAXIM."""
    
    def __init__(self, dim, grid_size=8, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.grid_size = grid_size
        self.norm = layers.LayerNormalization()
        self.dense = layers.Dense(dim)
        
    def call(self, x):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm(x)
        x = self.dense(x)
        return x + shortcut


class MAXIMBlock(layers.Layer):
    """MAXIM Block combining Block and Grid Gating Units."""
    
    def __init__(self, dim, grid_size=8, **kwargs):
        super().__init__(**kwargs)
        self.block_gating = BlockGatingUnit(dim)
        self.grid_gating = GridGatingUnit(dim, grid_size)
        
    def call(self, x):
        x = self.block_gating(x)
        x = self.grid_gating(x)
        return x


class DownsampleBlock(layers.Layer):
    """Downsample block for encoder."""
    
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(dim, 3, strides=2, padding='same')
        self.norm = layers.LayerNormalization()
        
    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class UpsampleBlock(layers.Layer):
    """Upsample block for decoder."""
    
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.up = layers.UpSampling2D(2)
        self.conv = layers.Conv2D(dim, 3, padding='same')
        self.norm = layers.LayerNormalization()
        
    def call(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        return x


def build_maxim_model(input_shape=(256, 256, 3), 
                      num_blocks=[2, 3, 3, 4],
                      dims=[64, 128, 256, 512],
                      grid_size=8):
    """
    Build MAXIM model for image deblurring.
    
    Args:
        input_shape: Input image shape (H, W, C)
        num_blocks: Number of MAXIM blocks at each stage
        dims: Channel dimensions at each stage
        grid_size: Grid size for grid gating unit
        
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(dims[0], 3, padding='same')(inputs)
    x = layers.LayerNormalization()(x)
    
    # Store encoder features for skip connections
    skip_connections = []
    
    # Encoder with MAXIM blocks
    for i, (dim, num_block) in enumerate(zip(dims, num_blocks)):
        # MAXIM blocks
        for _ in range(num_block):
            x = MAXIMBlock(dim, grid_size)(x)
        
        skip_connections.append(x)
        
        # Downsample (except last stage)
        if i < len(dims) - 1:
            x = DownsampleBlock(dims[i + 1])(x)
    
    # Bottleneck
    x = MAXIMBlock(dims[-1], grid_size)(x)
    x = MAXIMBlock(dims[-1], grid_size)(x)
    
    # Decoder with MAXIM blocks
    for i in range(len(dims) - 1, 0, -1):
        # Upsample
        x = UpsampleBlock(dims[i - 1])(x)
        
        # Skip connection
        x = layers.Concatenate()([x, skip_connections[i - 1]])
        
        # MAXIM blocks
        for _ in range(num_blocks[i - 1]):
            x = MAXIMBlock(dims[i - 1], grid_size)(x)
    
    # Output projection
    x = layers.Conv2D(dims[0], 3, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(3, 1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=x, name='MAXIM_Deblur')
    return model


def build_maxim_lightweight(input_shape=(256, 256, 3)):
    """
    Build a lightweight version of MAXIM for faster training and inference.
    
    Args:
        input_shape: Input image shape (H, W, C)
        
    Returns:
        Keras Model
    """
    return build_maxim_model(
        input_shape=input_shape,
        num_blocks=[1, 2, 2, 2],
        dims=[48, 96, 192, 384],
        grid_size=8
    )


def compile_maxim_model(model, learning_rate=0.0001):
    """
    Compile MAXIM model with optimizer and loss.
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model


if __name__ == "__main__":
    # Test model creation
    print("Building MAXIM model...")
    model = build_maxim_lightweight()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
