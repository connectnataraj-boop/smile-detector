import os
import streamlit as st
import joblib
import pickle
import numpy as np
from PIL import Image


@st.cache_resource
def load_model():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model  = joblib.load(os.path.join(BASE_DIR, "models", "regression.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

    return model, scaler


model, scaler = load_model()


def perprocessing_image(image):
    image = image.resize((64,64))
    image = image.convert('L')
    image = np.array(image).flatten()
    image = scaler.transform(image.reshape(1, -1))
    return image


st.title("Smile Detector")

upload_image = st.file_uploader("Upload_image: ", type=["jpeg", "jpg", "png"])

if upload_image is not None:
    img = Image.open(upload_image)
    st.image(img, caption='image uploaded')

    perprocessing_ima = perprocessing_image(img)
    image_prob = model.predict_proba(perprocessing_ima)[0][1]
    model_predict = model.predict(perprocessing_ima)

    smile_score = int(image_prob*100)
    st.slider("smile score", 0, 100, smile_score, disabled=True)

    if model_predict[0] == 1:
        st.success(f"person is smiling, with smile_score {smile_score}%")
    else:
        st.warning(f"person is not smiling, with smile_score {smile_score}%")

