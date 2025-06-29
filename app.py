import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import os


base_path = "C:/Users/HP/Documents/Image-Classifier"
model = load_model(os.path.join(base_path, 'catvsdogmodel.h5'))

st.title("🐾 Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog, and the model will predict what it is.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((100, 100))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = '🐱 Cat' if prediction > 0.5 else '🐶 Dog'
    st.markdown(f"### Prediction: {result}")
