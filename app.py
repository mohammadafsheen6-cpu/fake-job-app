import streamlit as st
import pickle

st.title("Fake Job Detection")

try:
    model = pickle.load(open(r"fake-job-app\model.pkl", "rb"))
    vectorizer = pickle.load(open(r"fake-job-app\vectorizer.pkl", "rb"))
except:
    st.error("Model files not loaded properly")
    st.stop()

user_input = st.text_area("Enter Job Description")

if st.button("Predict"):
    if user_input:
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("Fake Job")
        else:
            st.success("Real Job")
    else:
        st.warning("Enter text first")
