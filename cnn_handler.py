import os
import glob
import importlib.util
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

# Monkey-patch to prevent quantization_config errors from saved Keras 3 layers
original_dense_init = Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
Dense.__init__ = patched_dense_init

def load_cnn_model(models_dir, model_name):
    """Loads a traditional standalone CNN architecture."""
    model_folder = os.path.join(models_dir, "cnn", model_name)
    model_files = glob.glob(os.path.join(model_folder, "*.keras")) + \
                  glob.glob(os.path.join(model_folder, "*.h5")) + \
                  glob.glob(os.path.join(model_folder, "*.hdf5"))
                  
    if not model_files:
        raise FileNotFoundError(f"No CNN weight file found in {model_folder}")
        
    model = load_model(model_files[0])
    
    preprocess_path = os.path.join(model_folder, "preprocess.py")
    if not os.path.exists(preprocess_path):
        raise FileNotFoundError(f"Missing preprocess.py in {model_folder}")
        
    spec = importlib.util.spec_from_file_location("model_preprocess", preprocess_path)
    preprocess_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocess_module)
    
    return model, preprocess_module.preprocess

def classify_cnn(image, model, preprocess_func, alzheimer_classes):
    """Executes prediction utilizing native Keras neural layers."""
    target_height, target_width = model.input_shape[1], model.input_shape[2]
    resized_image = image.resize((target_width, target_height))
    
    img_array = tf.keras.preprocessing.image.img_to_array(resized_image)
    batched_image = np.expand_dims(img_array, axis=0)
    preprocessed_image = preprocess_func(batched_image)
    
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction[0])
    predicted_label = alzheimer_classes[predicted_class_index]

    prob_df = pd.DataFrame({
        "Class": alzheimer_classes,
        "Probability": prediction[0]
    }).sort_values(by="Probability", ascending=False)

    return predicted_label, prob_df, preprocessed_image