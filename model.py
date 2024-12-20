import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D,
    Dense, Multiply, Add, Activation
)
from tensorflow.keras.models import Model

def rcam_block(inputs, filters):
    # First Convolutional Block
    x = Conv2D(filters, (1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Depthwise Convolution
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second Convolutional Block
    x = Conv2D(filters, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    # Global Average Pooling for Attention
    gap = GlobalAveragePooling2D()(x)
    gap = Dense(filters // 16, activation='relu')(gap)
    gap = Dense(filters, activation='sigmoid')(gap)

    # Channel Attention
    attention = Multiply()([x, tf.expand_dims(tf.expand_dims(gap, 1), 1)])

    # Residual Connection
    output = Add()([inputs, attention])
    return output

def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # Initial Convolutional Layer
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)

    # RCAM 1
    x = rcam_block(x, 64)

    # RCAM 2
    x = rcam_block(x, 96)

    # RCAM 3
    x = rcam_block(x, 256)

    # Final Convolutional Layer
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inputs, x)
    return model

# Define the input shape based on the dataset (e.g., 224x224x3 for images)
input_shape = (224, 224, 3)
model = build_model(input_shape)

# Model Summary
model.summary()
