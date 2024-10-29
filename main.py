import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf
import json
from butterfly_data import butterfly_facts

# Page configuration
st.set_page_config(page_title="Butterfly Classifier", page_icon="🦋", layout="centered")

# Title and prompt for users
st.title('Butterfly Classification Application')
st.markdown("### 📸 Please upload a butterfly image for classification.")
st.divider()

# Sidebar with information
st.sidebar.title("About This App")
st.sidebar.write("This app classifies butterfly images with 80% accuracy and provides interesting facts about them.")
st.sidebar.markdown("### Instructions")
st.sidebar.write("1. Upload an image of a butterfly.\n2. View the predicted species and learn a fun fact!")

# Load the model
model = load_model('butterfly_model.keras')

# Load class names
with open('class_order.json', 'r') as f:
    class_names_ordered = json.load(f)

# Preprocess image function
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# File uploader
uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image")
    
    # Preprocess and predict
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)
    
    with st.spinner("Classifying..."):
        predictions = model.predict(processed_image)
    
    st.success("Classification complete!")

    # Get predicted class and corresponding fact
    predicted_class = np.argmax(predictions, axis=1)
    butterfly_name = class_names_ordered[predicted_class[0]]
    butterfly_fact = butterfly_facts.get(butterfly_name, "No fact available for this butterfly.")

    # Display results in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### 🦋 Predicted butterfly: **{butterfly_name}**")
    with col2:
        st.markdown(f"### 🌟 Fun Fact: {butterfly_fact}")