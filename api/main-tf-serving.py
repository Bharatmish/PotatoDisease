from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requestsfrom dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

endpoint = "http://localhost:8501/v1/models/potato-models:predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# OpenAI API Configuration
API_KEY = os.getenv("API_KEY")  # Retrieve API key from .env file
ENDPOINT = os.getenv("API_ENDPOINT") 

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def generate_summary(disease_name):
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    prompt = (
        f"The detected disease in the potato plant leaf is {disease_name}. "
        "Generate a detailed summary for the farmer. Include: "
        "1. Disease causes and symptoms. "
        "2. Recommended insecticides and pesticides for treatment. "
        "3. Precautions and preventive measures. "
        "4. Any additional advice for the farmer."
    )
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are an agricultural expert helping farmers manage plant diseases."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        summary = result["choices"][0]["message"]["content"]
        return summary
    except requests.RequestException as e:
        return f"Error generating summary: {e}"

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {"instances": img_batch.tolist()}
    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    # Generate the summary for the predicted class
    summary = generate_summary(predicted_class)

    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'summary': summary
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
