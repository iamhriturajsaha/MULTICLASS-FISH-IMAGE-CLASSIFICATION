import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
# CONFIGURATION
st.set_page_config(page_title="🐟 Fish Classifier", layout="centered", page_icon="🐟")
# MODEL LOADING
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Best Fish Classifier VGG16.h5")
    return model
model = load_model()
# CLASS LABELS
CLASS_NAMES = [
    'Black Sea Sprat',
    'Gilt Head Bream',
    'Red Mullet',
    'Red Sea Bream',
    'Shrimp',
    'Trout'
]
# HEADER SECTION
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🐟 Fish Species Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a clear fish image and the model will predict its species with confidence!</p>", unsafe_allow_html=True)
st.markdown("---")
# IMAGE UPLOAD
uploaded_file = st.file_uploader("📷 Upload a fish image (JPG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)
    # Preprocess
    st.markdown("🔄 Processing image...")
    image_resized = image.resize((160, 160))  # ✅ Match model input shape
    img_array = img_to_array(image_resized)
    img_array = tf.expand_dims(img_array, 0) / 255.0  # Normalize
    # Predict
    predictions = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    # Results
    st.markdown("## 📢 Prediction Result")
    st.success(f"**Predicted Class:** `{predicted_class}`")
    st.info(f"**Confidence:** `{confidence:.2f}%`")
    # Show All Class Probabilities
    st.markdown("## 📊 Class Probabilities")
    prob_dict = {cls: f"{score*100:.2f}%" for cls, score in zip(CLASS_NAMES, predictions)}
    st.json(prob_dict)
    # Bar Chart
    st.bar_chart(data=dict(zip(CLASS_NAMES, predictions)))
else:
    st.info("⬆️ Please upload a fish image to begin classification.")
