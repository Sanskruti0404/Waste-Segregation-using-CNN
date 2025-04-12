import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("waste_segregation_model.h5")

# Define class labels (ensure these match your dataset folder names)
class_labels = ['Plastic', 'Glass', 'Metal', 'Paper', 'Light Bulbs',' Organic','Batteries','Clothes','E-waste']

# Function to predict the category of waste
def predict_waste(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

# Streamlit UI
st.title("Waste Segregation")
st.write("Upload an image to predict the waste category.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save the image temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict the waste category
    predicted_class, confidence = predict_waste("temp_image.jpg")

    # Display the results
    st.write(f"**Predicted Category:** {predicted_class}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
