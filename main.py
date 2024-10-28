import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import tensorflow as tf

# Set the page configuration
st.set_page_config(layout="wide")

# Title of the application
st.title('Butterfly Classification Application')

st.divider()

# Load the pre-trained model
model = tf.keras.models.load_model('butterfly_model.keras')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize image to match model input shape
    image = np.array(image) / 255.0    # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define all the butterfly class names
class_names = [
    'AFRICAN GIANT SWALLOWTAIL', 'ADONIS', 'AN 88', 'AMERICAN SNOOT', 'APOLLO', 'ATALA', 
    'BANDED ORANGE HELICONIAN', 'BANDED PEACOCK', 'BANDED WHITE', 'BECKERS WHITE', 
    'BLACK HAIRSTREAK', 'BLUE SPOT HAIRSTREAK', 'BLUE MORPHO', 'BROWN SIPROETA', 
    'CABBAGE WHITE', 'CAIRNS BIRDWING', 'CHEQUERED SKIPPER', 'CLEOPATRA', 
    'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 'COMMON BANDED PEACOCK', 'COMMON WOOD-NYMPH', 
    'COPPER TAIL', 'CRAMER\'S ORANGE TIP', 'CRIMSON PATCH', 'DAINTY SULPHUR', 
    'DANNAUS GILIPPUS', 'EASTERN COMMA', 'EASTERN DAPPLE WHITE', 'EASTERN PIED PIERROT', 
    'ELBOWED PIERROT', 'GAEA', 'GOLD BANDED', 'GRAY HAIRSTREAK', 'GREAT EGGFLY', 
    'GREAT JAY', 'GREEN CELLED CATTLEHEART', 'GREY HAIRSTREAK', 'INDRA SWALLOW', 
    'IPHICLUS SISTER', 'ISIS', 'JASON', 'LARGE MARBLE', 'MANGROVE SKIPPER', 
    'METALMARK', 'MILBERTS TORTOISESHELL', 'MINO WING', 'MONARCH', 'MOURNING CLOAK', 
    'ORANGE OAKLEAF', 'ORANGE TIP', 'ORCHARD SWALLOW', 'PAINTED LADY', 'PAPER KITE', 
    'PIPEVINE SWALLOW', 'PURPLE HAIRSTREAK', 'PURPLE LEAFWING', 'QUEEN', 'QUESTION MARK', 
    'RED ADMIRAL', 'RED COSTUM', 'RED POSTMAN', 'SCARCE SWALLOW', 'SILVER SPOT HAIRSTREAK', 
    'SILVER-SPOT SKIPPER', 'SMALL COPPER', 'SLEEPY ORANGE', 'SOUTHERN DOGFACE', 
    'SPICEBUSH SWALLOW', 'STRAIGHT OAKEDGE', 'STRIPED QUEEN', 'TROPICAL LEAFWING', 
    'TWO BARRED FLINDER', 'ULYSSES', 'VICEROY', 'WOOD SATYR', 'YELLOW SWALLOW TAIL', 
    'ZEBRA LONG WING'
]

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

    # Display the prediction
    butterfly_name = class_names[predicted_class[0]]
    st.write(f"The predicted butterfly is: **{butterfly_name}**")

