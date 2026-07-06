import os
import glob
import importlib.util
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

def load_hybrid_model(models_dir, model_name):
    """Loads a dual-stage CNN Feature Extractor + Classical ML Classifier configuration."""
    model_folder = os.path.join(models_dir, "hybrid", model_name)
    
    # 1. Fetch CNN Backbone (Used exclusively for numeric embeddings output)
    cnn_files = glob.glob(os.path.join(model_folder, "*.keras")) + \
                glob.glob(os.path.join(model_folder, "*.h5")) + \
                glob.glob(os.path.join(model_folder, "*.hdf5"))
    if not cnn_files:
        raise FileNotFoundError(f"No Hybrid CNN backbone file found in {model_folder}")
    model = load_model(cnn_files[0])
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)  # Layer before final softmax )
    
    # 2. Fetch Classical ML Head (SVM, Random Forest, etc.)
    ml_files = glob.glob(os.path.join(model_folder, "*.pkl")) + \
               glob.glob(os.path.join(model_folder, "*.joblib"))
    if not ml_files:
        raise FileNotFoundError(f"No serialized ML classifier head (.pkl/.joblib) found in {model_folder}")
    classifier = joblib.load(ml_files[0])
    
    # 3. Fetch Image Preprocessor script
    preprocess_path = os.path.join(model_folder, "preprocess.py")
    if not os.path.exists(preprocess_path):
        raise FileNotFoundError(f"Missing preprocess.py in {model_folder}")
        
    spec = importlib.util.spec_from_file_location("model_preprocess", preprocess_path)
    preprocess_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocess_module)
    
    return feature_extractor, classifier, preprocess_module.preprocess

def classify_hybrid(image, feature_extractor, classifier, preprocess_func, alzheimer_classes):
    """Extracts features via CNN and infers target category via traditional ML Classifier."""
    target_height, target_width = feature_extractor.input_shape[1], feature_extractor.input_shape[2]
    resized_image = image.resize((target_width, target_height))
    
    img_array = tf.keras.preprocessing.image.img_to_array(resized_image)
    batched_image = np.expand_dims(img_array, axis=0)
    preprocessed_image = preprocess_func(batched_image)
    
    # Stage 1: Feedforward pass to extract numerical vector matrices
    features = feature_extractor.predict(preprocessed_image)
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1) # Flatten multi-dimensional spatial arrays
        
    # Stage 2: Inference execution via traditional Classifier
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features)[0]
        predicted_class_index = np.argmax(probabilities)
    else:
        # Fallback if the classical classifier (e.g., basic LinearSVC) doesn't calculate probabilities
        prediction = classifier.predict(features)[0]
        if isinstance(prediction, (str, np.str_)):
            predicted_class_index = alzheimer_classes.index(prediction)
        else:
            predicted_class_index = int(prediction)
            
        probabilities = np.zeros(len(alzheimer_classes))
        probabilities[predicted_class_index] = 1.0 # Categorical binary mask assignment
        
    predicted_label = alzheimer_classes[predicted_class_index]

    prob_df = pd.DataFrame({
        "Class": alzheimer_classes,
        "Probability": probabilities
    }).sort_values(by="Probability", ascending=False)

    return predicted_label, prob_df, preprocessed_image