from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf

def preprocess(images):
    # images: float32 in [0,255] (image_dataset_from_directory returns uint8 -> cast)
    images[0] = tf.cast(images[0], tf.float32)
    images = preprocess_input(images)  # handles normalization appropriate for EfficientNet
    return images