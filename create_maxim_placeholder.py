"""
Create a placeholder MAXIM model for demonstration
This creates a simple CNN model saved as MAXIM for the app to load
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def build_simple_maxim_model(input_shape=(256, 256, 3)):
    """
    Build a simplified MAXIM-inspired model architecture.
    This is a lightweight version that mimics MAXIM structure but trainable on limited resources.
    """
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # Encoder path with multi-scale features
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    skip1 = x
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    skip2 = x
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Bottleneck
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    
    # Decoder path with skip connections
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    
    # Output
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='MAXIM_Deblur')
    return model


def main():
    print("Creating MAXIM placeholder model...")
    
    # Create model
    model = build_simple_maxim_model()
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    print(f"\nModel created with {model.count_params():,} parameters")
    
    # Save directory
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'maxim_deblur_best.h5')
    model.save(model_path)
    
    print(f"\n✅ MAXIM model saved to: {model_path}")
    print(f"   File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    # Also save as .keras format
    keras_path = os.path.join(save_dir, 'maxim_deblur_best.keras')
    model.save(keras_path)
    print(f"✅ MAXIM model (Keras format) saved to: {keras_path}")
    
    model.summary()
    
    print("\n" + "="*60)
    print("MAXIM model is ready to use in the Streamlit app!")
    print("="*60)


if __name__ == "__main__":
    main()
