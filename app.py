import os
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Import custom handlers and visual helper models
from gradcam import get_last_spatial_layer_name, make_gradcam_heatmap, generate_gradcam_figure
from cnn_handler import load_cnn_model, classify_cnn
from hybrid_handler import load_hybrid_model, classify_hybrid

# ----------- This Code is sending signal to the Arduino Chip ----------------
import serial
import time

def send_signal(index):
    predicted_class = index
    try:
        ser = serial.Serial('COM3', 9600, timeout=1) 
        time.sleep(2)  # Give time for the connection to establish
        ser.write(str(predicted_class).encode())
        ser.close()
        print(f"Sent command for class: {predicted_class}")
    except Exception as e:
        print(f"Serial communication failed: {e}")
# ----------------------------------------------------------------------------------

alzheimer_classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
MODELS_DIR = "models"

# --- State Management Callback ---
def reset_classification_state():
    """Wipes all previous classification data from session state when model options change."""
    keys_to_clear = ["classified", "result", "prob_df", "preprocessed_batch"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# --- Educational Markdown Content ---
markdown_content = """
---
### Alzheimer's Disease Classification: Breakdown and Comparison

Medical image classification of Alzheimer's disease is often performed by analyzing brain MRI scans and categorizing them into four distinct stages. This comparative analysis highlights the key differences between these classes.

---

#### 1. Non-Demented (ND)

* **Characteristics:** This is the baseline, representing a healthy brain with no cognitive impairment.
* **Cognitive State:** Individuals are cognitively normal and show no signs of memory loss or confusion.
* **MRI Findings:** Brain structure and volume appear normal for the individual's age. There is no significant atrophy in key regions like the hippocampus or cerebral cortex.

#### 2. Very Mild Demented (VMD)

* **Characteristics:** This is the earliest stage of cognitive decline, sometimes referred to as Subjective Cognitive Decline (SCD) or Mild Cognitive Impairment (MCI).
* **Cognitive State:** Individuals may experience subtle, occasional memory lapses or difficulty finding words, but these do not significantly impact daily life.
* **MRI Findings:** Subtle or no visible signs of atrophy. Neuroimaging may show very early, subtle changes in brain volume that are often detectable only through advanced computational analysis.

#### 3. Mild Demented (MD)

* **Characteristics:** Memory loss and cognitive issues become more noticeable and begin to affect daily living.
* **Cognitive State:** Individuals may get lost in familiar places, have trouble with complex tasks (like managing finances), and exhibit personality changes.
* **MRI Findings:** Detectable brain atrophy, particularly in the hippocampus and medial temporal lobes. This shrinkage is a key indicator of the disease's progression.

#### 4. Moderate Demented (MOD)

* **Characteristics:** This is a more severe stage where cognitive decline is significant, and individuals require substantial assistance.
* **Cognitive State:** Individuals experience significant confusion, poor judgment, and may forget personal history or names of close family members. They cannot live independently.
* **MRI Findings:** Widespread, noticeable atrophy across multiple brain regions, including the cerebral cortex. The ventricles (fluid-filled cavities) of the brain often appear enlarged as the surrounding brain tissue shrinks.

---

#### Comparative Analysis

| Feature | Non-Demented (ND) | Very Mild Demented (VMD) | Mild Demented (MD) | Moderate Demented (MOD) |
| :--- | :--- | :--- | :--- | :--- |
| **Cognitive Decline** | None | Minimal, subjective | Noticeable, affecting daily life | Significant, requiring assistance |
| **Independence** | Fully independent | Fully independent | Independent with some difficulty | Requires significant assistance |
| **Brain Atrophy on MRI** | Not present | Not visible with the naked eye | Mild to moderate, localized to hippocampus | Widespread and severe |
| **Symptoms** | No symptoms | Occasional memory lapses | Trouble with complex tasks, spatial disorientation | Severe confusion, behavioral changes |
"""

def get_available_models(architecture_type):
    """Scans the designated subfolder (cnn or hybrid) for valid models."""
    target_path = os.path.join(MODELS_DIR, architecture_type.lower())
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        return []
    return sorted([d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d))])

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Alzheimer's Disease Medical Image Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Alzheimer's Disease Classifier On 2D MRI")

interactive_section = st.container()
footer_section = st.container()

with footer_section:
    st.markdown(markdown_content)

with interactive_section:
    # 1. Pipeline architecture selector split with clear placeholder option
    architecture_type = st.selectbox(
        "Select ML Architecture Pipeline:", 
        ["Select Architecture...", "CNN", "Hybrid"],
        on_change=reset_classification_state
    )
    
    # Halt execution if user hasn't selected an architecture type yet
    if architecture_type == "Select Architecture...":
        st.info("💡 Please select an ML Architecture Pipeline above to begin.")
        st.stop()
    
    available_models = get_available_models(architecture_type)
    if not available_models:
        st.error(f"No subfolders detected under `./{MODELS_DIR}/{architecture_type.lower()}/`. Please add valid dependencies.")
        st.stop()

    # Prepend placeholder selection string to the list of models
    model_options = ["Select Model Variant..."] + available_models

    # 2. Model target directory dropdown selector
    selected_model_name = st.selectbox(
        f"Select Precise {architecture_type} Variant:", 
        model_options,
        on_change=reset_classification_state
    )

    # Halt execution if user hasn't selected a specific model variant yet
    if selected_model_name == "Select Model Variant...":
        st.info(f"💡 Please select a specific {architecture_type} model variant to proceed.")
        st.stop()

    # Contextual environment loading assignment - Only triggers when valid selections exist
    try:
        if architecture_type == "CNN":
            model, preprocess_func = load_cnn_model(MODELS_DIR, selected_model_name)
        else:
            feature_extractor, classifier, preprocess_func = load_hybrid_model(MODELS_DIR, selected_model_name)
    except Exception as e:
        st.error(f"Error initializing architectural elements: {e}")
        st.stop()

    uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)
        if raw_image.mode != "RGB":
            image = raw_image.convert("RGB")
        else:
            image = raw_image
        
        st.image(image, caption="Uploaded Image Baseline Source", width=300)
        
        if st.button("Classify Image"):
            with st.spinner(f"Running inferential metrics across {selected_model_name}..."):
                if architecture_type == "CNN":
                    result, prob_df, preprocessed_batch = classify_cnn(image, model, preprocess_func, alzheimer_classes)
                else:
                    result, prob_df, preprocessed_batch = classify_hybrid(image, feature_extractor, classifier, preprocess_func, alzheimer_classes)
                
                st.session_state["classified"] = True
                st.session_state["result"] = result
                st.session_state["prob_df"] = prob_df
                st.session_state["preprocessed_batch"] = preprocessed_batch
                
                # Send the physical trigger signature to Arduino chip
                class_index = alzheimer_classes.index(result)
                # send_signal(class_index)

        # Output results are strictly guarded by this session check.
        # If the dropdown changes, this whole block is skipped because 'classified' is removed!
        if st.session_state.get("classified", False):
            result = st.session_state["result"]
            prob_df = st.session_state["prob_df"]
            
            if result == "NonDemented":
                st.success(f"Classification Label: **{result}**")
                st.balloons()
            elif result == "VeryMildDemented":
                st.warning(f"Classification Label: **{result}**")
            else:
                st.error(f"Classification Label: **{result}**")

            st.subheader("Prediction Weights Map")
            st.table(prob_df)

            # Grad-CAM is restricted to pure CNN backprop pipelines
            if architecture_type == "CNN":
                st.subheader("Explainability Analysis (Grad-CAM)")
                gradcam_placeholder = st.empty()
                
                if st.button("Generate Grad-CAM Overlay"):
                    with st.spinner("Calculating gradients..."):
                        preprocessed_batch = st.session_state["preprocessed_batch"]
                        layer_name = get_last_spatial_layer_name(model)
                        heatmap, pred_idx, confidence = make_gradcam_heatmap(preprocessed_batch, model, layer_name)
                        
                        fig = generate_gradcam_figure(image, heatmap, pred_idx, confidence, alzheimer_classes)
                        
                        with gradcam_placeholder.container():
                            col1, col2 = st.columns([1, 1.3])
                            with col1:
                                st.image(image, caption="Original Input", width='stretch')
                            with col2:
                                st.pyplot(fig, width='stretch')
                                plt.close(fig)
            else:
                st.info("💡 Grad-CAM visualization is unavailable for Hybrid models because downstream classical machine learning heads (like SVM or Random Forest) do not support backpropagation gradients.")