import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Set the page configuration
st.set_page_config(layout="wide")

# Title of the application
st.title('Butterfly Classification Application')

st.divider()

# Load the pre-trained model
model = load_model('butterfly_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize image to match model input shape
    image = np.array(image) / 255.0    # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# File uploader
uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image")
    
    # Preprocess the uploaded image
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)

    # Predict the class
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)

    # Assuming you have a mapping from class index to butterfly names
    class_names = ['Class1', 'Class2', ..., 'Class76']  # Replace with actual class names
    butterfly_name = class_names[predicted_class[0]]

    # Display the prediction
    st.write(f"The predicted butterfly is: **{butterfly_name}**")

# Camera input
picture = st.camera_input("Take a picture")

if picture is not None:
    # Display the image taken from camera
    st.image(picture, caption="Camera Image")

    # Preprocess the camera image
    image = Image.open(picture)
    processed_image = preprocess_image(image)

    # Predict the class
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)

    # Display the prediction for the camera image
    butterfly_name = class_names[predicted_class[0]]
    st.write(f"The predicted butterfly is: **{butterfly_name}**")