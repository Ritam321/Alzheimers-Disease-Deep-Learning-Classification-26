import cv2
import numpy as np

IMG_SIZE = 128

def preprocess(batch_images):
    """
    Preprocess a batch of MRI images.

    Steps:
    1. Convert RGB images to grayscale.
    2. Resize each image to 128 × 128.
    3. Normalize pixel values to [0, 1].
    4. Add a channel dimension.

    Args:
        batch_images: NumPy array of shape (batch_size, H, W, 3)

    Returns:
        NumPy array of shape (batch_size, 128, 128, 1)
    """
    processed = []

    for img in batch_images:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Resize
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

        # Normalize
        gray = gray.astype(np.float32) / 255.0

        processed.append(gray)

    processed = np.array(processed)
    processed = np.expand_dims(processed, axis=-1)

    return processed