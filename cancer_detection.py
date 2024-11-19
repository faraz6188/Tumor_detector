import streamlit as st
import joblib
import tensorflow as tf
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Load the scaler and the Keras model
scaler = joblib.load('scaler.h5')  # Ensure this is the correct path to your scaler file
model = tf.keras.models.load_model('cancer_prediction_model.h5')  # Ensure this is the correct path to your model file

# Main Title
st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #ff6347;
        text-align: center;
    }
    .sub-title {
        font-size: 18px;
        color: #4682b4;
        text-align: center;
    }
    </style>
    <div>
        <h1 class="main-title">Tumor Cancer Detector 🩺</h1>
        <p class="sub-title">Predict tumor status with Faraz's advanced AI model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar with description
with st.sidebar:
    st.title("About This App")
    st.write(
        "This AI tool, developed by Faraz, helps detect whether a tumor is cancerous. It's designed to assist doctors in early diagnosis."
    )
    st.header("How It Works:")
    st.write(
        """
        1. Enter tumor features using sliders.
        2. Click **Predict** to analyze the data.
        3. Get instant results with visual feedback!
        """
    )
    faqs = {
        "Problem Statement": "Detecting cancer early is important, but it can take time and sometimes lead to mistakes. We need a faster, more accurate way to check if a tumor is cancerous.",
        "Solution Overview": "This tool uses a deep learning model to study tumor features like size and texture. It helps doctors quickly and reliably find out if a tumor is cancerous.",
    }
    for question, answer in faqs.items():
        with st.expander(question):
            st.write(answer)


# Feature Input Section
st.subheader("Enter Tumor Features 📝")

st.write("Note: Enter valid details otherwise will not work")
col1, col2, col3 = st.columns(3)

# Feature inputs
# Input fields organized into columns
# Feature inputs using select sliders
mean_radius = col1.select_slider("Mean Radius", options=[round(x, 1) for x in list(range(60, 301, 1))], value=140) / 10
mean_texture = col2.select_slider("Mean Texture", options=[round(x, 1) for x in list(range(90, 401, 1))], value=200) / 10
mean_perimeter = col3.select_slider("Mean Perimeter", options=[round(x, 1) for x in list(range(400, 2001, 1))], value=900) / 10
mean_area = col1.select_slider("Mean Area", options=[x for x in range(100, 3001, 50)], value=600)
mean_smoothness = col2.select_slider("Mean Smoothness", options=[round(x, 3) for x in list(range(50, 201))], value=100) / 1000
mean_compactness = col3.select_slider("Mean Compactness", options=[round(x, 2) for x in list(range(2, 36))], value=10) / 100
mean_concavity = col1.select_slider("Mean Concavity", options=[round(x, 2) for x in list(range(0, 51))], value=8) / 100
mean_concave_points = col2.select_slider("Mean Concave Points", options=[round(x, 2) for x in list(range(0, 21))], value=5) / 100
mean_symmetry = col3.select_slider("Mean Symmetry", options=[round(x, 2) for x in list(range(10, 41))], value=18) / 100
mean_fractal_dimension = col1.select_slider("Mean Fractal Dimension", options=[round(x, 2) for x in list(range(50, 101))], value=60) / 1000
radius_error = col2.select_slider("Radius Error", options=[round(x, 1) for x in list(range(10, 301, 1))], value=30) / 100
texture_error = col3.select_slider("Texture Error", options=[round(x, 1) for x in list(range(50, 501, 1))], value=150) / 100
perimeter_error = col1.select_slider("Perimeter Error", options=[round(x, 1) for x in list(range(5, 101, 1))], value=20) / 10
area_error = col2.select_slider("Area Error", options=[x for x in range(5, 101, 5)], value=15)
smoothness_error = col3.select_slider("Smoothness Error", options=[round(x, 3) for x in list(range(1, 11))], value=5) / 1000
compactness_error = col1.select_slider("Compactness Error", options=[round(x, 3) for x in list(range(5, 51))], value=20) / 1000
concavity_error = col2.select_slider("Concavity Error", options=[round(x, 3) for x in list(range(5, 51))], value=20) / 1000
concave_points_error = col3.select_slider("Concave Points Error", options=[round(x, 3) for x in list(range(1, 21))], value=10) / 1000
symmetry_error = col1.select_slider("Symmetry Error", options=[round(x, 3) for x in list(range(10, 51))], value=30) / 1000
fractal_dimension_error = col2.select_slider("Fractal Dimension Error", options=[round(x, 3) for x in list(range(1, 6))], value=2) / 1000
worst_radius = col3.select_slider("Worst Radius", options=[round(x, 1) for x in list(range(70, 401, 1))], value=160) / 10
worst_texture = col1.select_slider("Worst Texture", options=[round(x, 1) for x in list(range(120, 501, 1))], value=250) / 10
worst_perimeter = col2.select_slider("Worst Perimeter", options=[round(x, 1) for x in list(range(500, 2501, 10))], value=1000) / 10
worst_area = col3.select_slider("Worst Area", options=[x for x in range(200, 4001, 50)], value=700)
worst_smoothness = col1.select_slider("Worst Smoothness", options=[round(x, 2) for x in list(range(5, 31))], value=15) / 100
worst_compactness = col2.select_slider("Worst Compactness", options=[round(x, 2) for x in list(range(2, 51))], value=25) / 100
worst_concavity = col3.select_slider("Worst Concavity", options=[round(x, 2) for x in list(range(0, 101))], value=20) / 100
worst_concave_points = col1.select_slider("Worst Concave Points", options=[round(x, 2) for x in list(range(0, 31))], value=10) / 100
worst_symmetry = col2.select_slider("Worst Symmetry", options=[round(x, 2) for x in list(range(10, 61))], value=25) / 100
worst_fractal_dimension = col3.select_slider("Worst Fractal Dimension", options=[round(x, 2) for x in list(range(50, 151))], value=80) / 1000

# Input list for the model
user_inputs = [
    mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity,
    mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error,
    area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error,
    fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
    worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension
]

# Prediction function
def predict_cancer(inputs):
    # Convert input list to DataFrame
    input_df = pd.DataFrame([inputs], columns=[
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
        'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ])

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)
    return "Cancerous" if prediction[0][0] > 0.5 else "Not Cancerous"

# Predict button
import time

import time
import streamlit as st

if st.button("Predict"):
    # Create a placeholder for the "Processing..." message
    processing_placeholder = st.empty()
    
    # Display the "Processing..." message
    processing_placeholder.markdown(
        "<div style='text-align:center; font-size: 20px;'>Processing...</div>", 
        unsafe_allow_html=True
    )
    
    # Wait for 2 seconds
    time.sleep(2)
    
    # Clear the "Processing..." message
    processing_placeholder.empty()
    
    # Perform prediction after the delay
    result = predict_cancer(user_inputs)
    if result == "Cancerous":
        st.error("🚨 The tumor is CANCEROUS. Consult a doctor immediately!")
        st.snow()
    else:
        st.success("🎉 The tumor is NOT Cancerous. Stay healthy!")
        st.balloons()

        
