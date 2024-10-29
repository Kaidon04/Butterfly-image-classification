import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import tensorflow as tf
import json
from butterfly_data import butterfly_facts

st.set_page_config(layout="wide")
st.title('Butterfly Classification Application')
st.divider()

model = tf.keras.models.load_model('butterfly_model.keras')


#load in class names with json
with open('class_order.json', 'r') as f:
    class_names_ordered = json.load(f)



def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image")
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)

    butterfly_name = class_names_ordered[predicted_class[0]]
    butterfly_fact = butterfly_facts.get(butterfly_name, "No fact available for this butterfly.")

    st.write(f"The predicted butterfly is: **{butterfly_name}**")
    st.write(f"Fun Fact: {butterfly_fact}")
