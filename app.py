
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(page_title="Fruit & Veg Classifier", page_icon="🍎")

@st.cache_resource
def load_my_model():
    return load_model("fruit_veg_classifier.h5")

model = load_my_model()

class_names = ['Apple Braeburn 1', 'Banana 1', 'Cabbage white 1', 'Carrot 1',
               'Cucumber 1', 'Grape Blue 1', 'Mango 1', 'Onion White 1',
               'Orange 1', 'Pear 1', 'Pineapple 1', 'Potato Red 1',
               'Strawberry 1', 'Tomato 1', 'Watermelon 1']

st.title("🍎 Fruit & Vegetable Classifier")
st.write("Upload an image and the model will predict which fruit or vegetable it is.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = img.resize((100, 100))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    top_indices = predictions.argsort()[-3:][::-1]

    st.subheader("Top 3 Predictions:")
    for idx in top_indices:
        st.write(f"**{class_names[idx]}**: {predictions[idx]*100:.2f}%")
        st.progress(float(predictions[idx]))
