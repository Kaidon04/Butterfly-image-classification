import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import tensorflow as tf


st.set_page_config(layout="wide")
st.title('Butterfly Classification Application')
st.divider()

model = tf.keras.models.load_model('butterfly_model.keras')

# Define the butterfly class names in the order of model indices
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

# Assuming model uses 0 to 74 as indices; modify if needed
unique_classes = np.arange(75)  # Adjust if indices are different

# Define the ordered class names for the model
class_names_ordered = [class_names[i] for i in unique_classes]

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
    st.write(f"The predicted butterfly is: **{butterfly_name}**")
