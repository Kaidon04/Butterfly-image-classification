import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import tensorflow as tf
import json


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


with open('class_order.json', 'r') as f:
    class_names_ordered = json.load(f)

# define interesting facts for each butterfly
butterfly_facts = {
    'AFRICAN GIANT SWALLOWTAIL': "This butterfly is the largest in Africa, with a wingspan reaching up to 9.8 inches. Known for its toxicity, it likely gets its toxins from its caterpillar diet of certain rainforest plants, which helps it ward off predators.",
    'ADONIS': "The Adonis Blue butterfly is known for its bright blue males and its interesting relationship with ants. During its caterpillar stage, it secretes honeydew to attract ants, which protect it from predators in return. These butterflies also prefer warm, chalky grasslands in Europe.",
    'AN 88': "The '88' butterfly, or Diaethria anna, is named for the striking '88' pattern on the underside of its wings, making it visually unique among butterflies.",
    'AMERICAN SNOOT': "The American Snoot is named for its elongated mouthparts, giving it a 'snooty' look. It has bold black and white markings that help it blend into its surroundings.",
    'APOLLO': "Apollo butterflies are adapted to cold mountainous areas in Europe and Asia and have large, translucent wings that help with camouflage against rocky backgrounds.",
    'ATALA': "The Atala butterfly was once thought to be extinct due to habitat loss but has rebounded. Its caterpillars feed on toxic plants, which makes the butterfly itself distasteful to predators.",
    'BANDED ORANGE HELICONIAN': "This butterfly is known for its bright orange and black banded wings, which act as a warning to predators that it may be poisonous.",
    'BANDED PEACOCK': "The Banded Peacock butterfly sports a stunning iridescent green band on its wings and is found primarily in South Asia, where it is highly territorial.",
    'BANDED WHITE': "The Banded White is notable for its distinctive white bands, which help it camouflage among flowers and foliage in its South American habitats.",
    'BECKERS WHITE': "Native to North America’s arid regions, Becker's White butterfly has striking green-bordered wings and prefers desert plants from the mustard family as host plants.",
    'BLACK HAIRSTREAK': "This butterfly is quite rare in the UK and can be found only in certain blackthorn-rich habitats. It relies on ants to protect its larvae.",
    'BLUE SPOT HAIRSTREAK': "Known for the blue spots on its wings, this Mediterranean butterfly is adapted to hot, dry environments and is mostly active in summer.",
    'BLUE MORPHO': "The Blue Morpho's vibrant blue color is not due to pigment but to the microscopic structure of its scales, which reflect light to create an iridescent effect.",
    'BROWN SIPROETA': "Also called the Malachite butterfly, its brown and green wing pattern provides camouflage in tropical forests, where it feeds on rotting fruit.",
    'CABBAGE WHITE': "This common butterfly is considered a pest to gardeners as its larvae feed on cabbage and other plants in the mustard family.",
    'CAIRNS BIRDWING': "One of Australia’s largest butterflies, the Cairns Birdwing is known for its vivid green and black wings and relies on specific rainforest vines as host plants.",
    'CHEQUERED SKIPPER': "Found in Scotland and parts of Europe, this butterfly prefers grassy, open woodlands and has a quick, darting flight pattern.",
    'CLEOPATRA': "The Cleopatra butterfly resembles a large Brimstone and can be distinguished by the male’s deep orange wing coloration, which helps attract females."
    # Continue with other butterfly species up to 'ZEBRA LONG WING'
}

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
