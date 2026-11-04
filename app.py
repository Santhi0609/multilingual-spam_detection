
import streamlit as st
import joblib
from pathlib import Path

# Locate the saved model
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "multilingual_spam_model.joblib"

# Configure the web page
st.set_page_config(
    page_title="Multilingual Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Application title and description
st.title("🛡️ Multilingual Spam Detection")
st.write(
    "Check whether a message is spam or ham "
    "using a machine learning model."
)

st.info("Supported dataset languages: English, Hindi, and French.")

# Message input
message = st.text_area(
    "Enter your message:",
    placeholder="Type or paste a message here...",
    height=150
)

# Prediction button
if st.button("Check Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message.strip()])[0]

        if prediction == "spam":
            st.error("🚨 Prediction: SPAM")
            st.write("This message may be unwanted or suspicious.")
        else:
            st.success("✅ Prediction: HAM")
            st.write("This message was classified as legitimate.")

st.caption(
    "Machine learning predictions can be incorrect. "
    "Check suspicious messages carefully."
)
