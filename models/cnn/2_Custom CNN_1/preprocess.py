from tensorflow.keras import layers


rescale_layer = layers.Rescaling(1./255)
resize_layer = layers.Resizing(224, 224)

def preprocess(image):
    """Performs static adjustments. Safe to cache."""
    image = resize_layer(image)
    image = rescale_layer(image)
    return image