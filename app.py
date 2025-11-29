import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Radiologist",
    page_icon="🩻",
    layout="wide"
)

# --- LOAD MODEL (Cached for speed) ---
@st.cache_resource
def load_model():
    # Attempt to load the model
    try:
        model = tf.keras.models.load_model('pneumonia_detection_model_vgg16.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- GRAD-CAM FUNCTIONS ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3", pred_index=None):
    # Split the model: VGG16 (Base) and Custom Head
    base_model = model.layers[0] 
    
    # Model that outputs the Conv Layer AND VGG's final output
    base_grad_model = tf.keras.models.Model(
        inputs=[base_model.input], 
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = base_grad_model(img_array)
        # Pass output through custom head layers
        preds = base_outputs
        for layer in model.layers[1:]:
            preds = layer(preds)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate Gradients
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Generate Heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    # 1. Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # 2. Colorize
    jet = matplotlib.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # 3. Create RGB Image from heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # 4. Superimpose
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

# --- UI LAYOUT ---
st.title("🩻 AI Radiologist: Pneumonia Detection")
st.markdown("Upload a Chest X-Ray to get an instant analysis with **Explainable AI (Grad-CAM)**.")

# File Uploader
uploaded_file = st.file_uploader("Choose an X-Ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Preprocess Image
    image = Image.open(uploaded_file).convert('RGB')
    st.write("Processing image...")
    
    # Resize to 224x224 (Model Expectation)
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.array(image_resized) / 255.0 # Normalize
    img_tensor = np.expand_dims(img_array, axis=0)

    # 2. Prediction
    if st.button("Analyze X-Ray"):
        with st.spinner("AI is examining the lungs..."):
            prediction = model.predict(img_tensor)[0][0] # Raw probability
            
            # 3. Visualization
            heatmap = make_gradcam_heatmap(img_tensor, model)
            result_img = overlay_heatmap(img_array * 255, heatmap) # Denormalize for display

            # 4. Display Results
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original X-Ray", use_column_width=True)
            
            with col2:
                st.image(result_img, caption="AI Heatmap (Red = Affected Area)", use_column_width=True)

            # 5. Diagnosis Box
            st.divider()
            if prediction > 0.5:
                confidence = prediction * 100
                st.error(f"### ⚠️ Diagnosis: PNEUMONIA DETECTED")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.write("The AI has highlighted areas of opacity (fluid/inflammation) in red.")
            else:
                confidence = (1 - prediction) * 100
                st.success(f"### ✅ Diagnosis: NORMAL")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.write("Lungs appear clear.")

# --- SIDEBAR INFO ---
with st.sidebar:
    st.info("ℹ️ **About this Model**")
    st.write("This AI uses a **VGG16 Convolutional Neural Network** trained on pediatric chest X-rays.")
    st.write("It uses **Grad-CAM** technology to visualize exactly *where* it is looking, reducing the 'Black Box' problem.")
    st.warning("**Disclaimer:** This tool is for educational purposes only. Always consult a doctor.")