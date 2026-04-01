import streamlit as st
import pickle

st.title("Fake Job Detection")

try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
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
