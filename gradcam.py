import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def get_last_spatial_layer_name(model):
    """Finds the last convolutional layer in the model architecture."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Could not identify a suitable Conv2D layer for Grad-CAM.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Generates the Grad-CAM heatmap and extracts confidence scores."""
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)

        if isinstance(preds, list):
            preds = preds[0]

        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    # Calculate confidence score via Softmax
    # Check if outputs are already probabilities (sum equals ~1.0). If not, apply softmax.
    if tf.abs(tf.reduce_sum(preds[0]) - 1.0) < 1e-3:
        confidence = preds[0][pred_index]
    else:
        confidence = tf.nn.softmax(preds[0])[pred_index]
    
    return heatmap.numpy(), pred_index.numpy(), confidence.numpy()

def generate_gradcam_figure(orig_img_pil, heatmap, pred_idx, confidence, class_names, alpha=0.5):
    """
    Creates a complete Matplotlib figure matching the target visual layout:
    superimposed heatmap, title with prediction & confidence, and colorbar.
    """
    # Convert PIL Image to RGB NumPy array
    img = np.array(orig_img_pil)
    
    # Resize heatmap to match image spatial dimensions
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
    colored_heatmap = cv2.resize(colored_heatmap, (img.shape[1], img.shape[0]))
    
    # OpenCV uses BGR; convert back to RGB for Matplotlib rendering
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = cv2.addWeighted(img, 1 - alpha, colored_heatmap, alpha, 0)
    
    # Create Matplotlib Figure
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(superimposed_img)
    
    label = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
    ax.set_title(f"Prediction: {label}\nConf: {confidence:.2%}", fontsize=12, pad=10)
    ax.axis('off')
    
    # Create the vertical colorbar
    norm = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap='hot', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Activation Importance', rotation=270, labelpad=15)
    
    plt.tight_layout()
    return fig