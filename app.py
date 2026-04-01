import streamlit as st
import pickle
import os

st.title("Fake Job Detection")

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(_file_))

try:
    model_path = os.path.join(BASE_DIR, "model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

user_input = st.text_area("Enter Job Description")

if st.button("Predict"):
    if user_input.strip():
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Fake Job")
        else:
            st.success("✅ Real Job")
    else:
        st.warning("Please enter job description")
