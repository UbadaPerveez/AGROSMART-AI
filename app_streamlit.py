import streamlit as st
from google import genai
from dotenv import load_dotenv
from PIL import Image
import os
from pathlib import Path

# Load API key
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

# UI Setup
st.set_page_config(page_title="AgroSmart AI 🌿", layout="centered")

st.title("🌿 AgroSmart AI")
st.subheader("Plant Disease Detection using AgroSmart AI")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    prompt = """
You are an expert agricultural AI.

Analyze this plant leaf image and return:
1. Plant name
2. Disease name
3. Cause
4. Treatment
5. Prevention


Keep it simple for farmers
"""

    with st.spinner("Analyzing image... 🌱"):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image]
        )

    st.success("Analysis Complete ✅")
    st.markdown("### 🌾 Result")
    st.write(response.text)