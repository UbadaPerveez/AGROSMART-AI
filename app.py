from flask import Flask, request, render_template_string
from google import genai
from dotenv import load_dotenv
from PIL import Image
import os
from pathlib import Path

# -------------------------
# Load .env correctly
# -------------------------
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

client = genai.Client(api_key=API_KEY)

app = Flask(__name__)

# -------------------------
# Simple UI
# -------------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AgroSmart AI</title>
</head>
<body>
    <h2>🌿 AgroSmart AI - Plant Disease Detector</h2>

    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Analyze</button>
    </form>

    <pre>{{ result }}</pre>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML, result="Upload a leaf image")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]

        image = Image.open(file)

        prompt = """
You are an expert agricultural AI.
Analyze this plant leaf image.

Return:
1. Disease name (if any)
2. Cause
3. Cure
4. Prevention tips

Keep it simple for farmers.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image]
        )

        return render_template_string(HTML, result=response.text)

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)