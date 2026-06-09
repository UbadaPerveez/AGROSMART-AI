import streamlit as st
from google import genai
from google.genai import errors
from dotenv import load_dotenv
from PIL import Image
import os
from pathlib import Path

# 1. UI Setup (MUST BE FIRST Streamlit command)
st.set_page_config(page_title="AgroSmart AI 🌿", layout="centered")

# Load local .env file (only used on your local computer)
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# Automatically pulls from local .env OR from the Streamlit Cloud dashboard
API_KEY = os.environ.get("GOOGLE_API_KEY")

# Connect to Google Gemini safely
client = None
if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    st.error("Error: GOOGLE_API_KEY environment variable is missing!")

# Page Visual Header
st.title("🌿 AgroSmart AI")
st.subheader("Plant Disease Detection using AgroSmart AI")

# File Uploader Widget
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    # Prompt engineering designed for agricultural analysis
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

    # Only run the analysis if the client connected successfully
    if client:
        with st.spinner("Analyzing image... 🌱"):
            try:
                # Safe API call to Gemini
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[prompt, image]
                )
                
                # Render results if successful
                st.success("Analysis Complete ✅")
                st.markdown("### 🌾 Result")
                st.write(response.text)
                
            except errors.ServerError as e:
                # Specifically captures the 503 high demand spike error
                st.error("⚠️ Google's Gemini servers are experiencing extremely high demand right now. Please wait 10–15 seconds and try clicking upload again.")
            except Exception as e:
                # Captures any other random network hiccups safely
                st.error(f"⚠️ An unexpected error occurred: {e}")
    else:
        st.warning("Cannot analyze image because the Google API client could not be initialized.")
