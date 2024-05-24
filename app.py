# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
from keras.utils import custom_object_scope

# Define the custom layer
class CustomScaleLayer(Layer):
    def __init__(self, scale_factor, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs):
        return inputs * self.scale_factor

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({'scale_factor': self.scale_factor})
        return config

# Attempt to load the model
try:
    with custom_object_scope({'CustomScaleLayer': CustomScaleLayer}):
        model = load_model('dog_breed.h5')
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Name of Classes
CLASS_NAMES = ['Scottish Deerhound', 'Maltese Dog', 'Bernese Mountain Dog']

# Setting Title of App
st.title("Dog Breed Prediction")
st.markdown("Upload an image of the dog")

# Uploading the dog image
dog_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
submit = st.button('Predict')

# On predict button click
if submit:
    if dog_image is not None:
        try:
            # Convert the file to an OpenCV image
            file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Displaying the image
            st.image(opencv_image, channels="BGR")

            # Resizing the image
            opencv_image = cv2.resize(opencv_image, (224, 224))

            # Convert image to 4 Dimensions
            opencv_image = np.expand_dims(opencv_image, axis=0)

            # Normalize the image
            opencv_image = opencv_image.astype('float32') / 255.0
            
            # Print debug statements
            st.write(f"Image shape: {opencv_image.shape}")
            st.write(f"Image dtype: {opencv_image.dtype}")

            # Make Prediction
            Y_pred = model.predict(opencv_image)
            st.title(f"The Dog Breed is {CLASS_NAMES[np.argmax(Y_pred)]}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.warning("Please upload an image before submitting.")
