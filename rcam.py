import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, Multiply, Softmax, Add, Reshape
)
from tensorflow.keras.models import Model

def rscam_block(inputs, filters):
    # First Convolution
    x1 = Conv2D(filters, (1, 1), padding='same')(inputs)
    x1 = BatchNormalization()(x1)

    # Second Convolution
    x2 = Conv2D(filters // 2, (1, 1), padding='same')(inputs)
    x2 = BatchNormalization()(x2)

    # Reshaping x1 and x2 for Attention Mechanism
    shape = tf.shape(x1)
    x1_reshaped = Reshape((shape[1] * shape[2], shape[3]))(x1)
    x2_reshaped = Reshape((shape[1] * shape[2], shape[3]))(x2)

    # Compute Attention
    attention = tf.matmul(x1_reshaped, x2_reshaped, transpose_b=True)
    attention = Softmax()(attention)

    # Apply Attention
    x2_attention = tf.matmul(attention, x2_reshaped)
    x2_attention = Reshape((shape[1], shape[2], shape[3]))(x2_attention)

    # Residual Connection and Multiplication
    x = Multiply()([x1, x2_attention])

    # Final Residual Addition
    output = Add()([inputs, x])
    return output

def build_rscam_model(input_shape):
    inputs = Input(shape=input_shape)

    # Initial Convolutional Layer
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)

    # Apply RSCAM Block
    x = rscam_block(x, 64)

    # Final Output Convolution
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

# Define the input shape based on the dataset (e.g., 224x224x3 for images)
input_shape = (224, 224, 3)
model = build_rscam_model(input_shape)

# Model Summary
model.summary()
