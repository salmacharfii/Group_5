import joblib
import streamlit as st

st.title("Customer Review Sentiment")
st.write("Group 5 · Ironhack AI Engineering · TF-IDF + Logistic Regression trained on 27,700 Amazon reviews")


@st.cache_resource
def load_model():
    return joblib.load("App/sentiment_model.joblib")


model = load_model()

review = st.text_area("Paste a product review", "Stopped charging after two weeks.")

if st.button("Predict"):
    probabilities = model.predict_proba([review])[0]
    prediction = model.classes_[probabilities.argmax()]
    st.subheader(f"Sentiment: {prediction}")
    for label, p in zip(model.classes_, probabilities):
        st.write(f"{label}: {p:.0%}")
        st.progress(float(p))
