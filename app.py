import streamlit as st
import joblib
import numpy as np

model = joblib.load("best_iris_model.pkl")

st.title("🌸 Iris Flower Classification")

st.write("Enter the flower measurements:")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Species: {prediction[0]}")
