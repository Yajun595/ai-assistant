import os
import base64
import requests
from openai import OpenAI
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from google.genai import types
from google import genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RAPID_KEY = os.getenv("RAPID_KEY")

# Initialize OpenAI client
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def get_current_weather(location: str) -> dict:
    """
    Get weather information for a specific location using RapidAPI
    
    Args:
        location (str): Name of the city
        
    Returns:
        dict: Weather data for the specified location
    """
    url = f"https://open-weather13.p.rapidapi.com/city/{location}/EN"
    headers = {
        "x-rapidapi-key": RAPID_KEY,
        "x-rapidapi-host": "open-weather13.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers)
    return response.json()

def generate_image(prompt: str = "a portrait of a sheepadoodle wearing cape"):
    """
    Generate and display an image using AI model
    
    Args:
        prompt (str): Description of the image to generate
    """
    response = client.images.generate(
        model="imagen-3.0-generate-002",
        prompt=prompt,
        response_format='b64_json',
        n=1,
    )
    for image_data in response.data:
        image = Image.open(BytesIO(base64.b64decode(image_data.b64_json)))
        image.show()

def function_call():
    """
    Demonstrate function calling with Gemini model for weather queries
    """
    config = types.GenerateContentConfig(tools=[get_current_weather])
    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        config=config,
        contents="What's the weather like in Changsha today?"
    )
    print(response.text)

def generate_text(content: str):
    """
    Generate text response using AI model
    
    Args:
        content (str): Input text prompt
    """
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        n=1,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
    )
    print(response.choices[0].message.content)

def analyze_audio(file_path: str):
    """
    Transcribe and analyze audio file from local path or URL
    
    Args:
        file_path (str): Local path or URL to the audio file
    """
    if file_path.startswith(('http://', 'https://')):
        # Download file from URL
        response = requests.get(file_path)
        if response.status_code != 200:
            raise Exception(f"Failed to download audio file: {response.status_code}")
        audio_content = response.content
    else:
        # Read local file
        with open(file_path, "rb") as audio_file:
            audio_content = audio_file.read()

    # Encode audio content
    base64_audio = base64.b64encode(audio_content).decode('utf-8')

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio"},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64_audio,
                        "format": "wav"
                    }
                }
            ]
        }]
    )
    print(response.choices[0].message.content)

def main():
    """Main function to demonstrate different API functionalities"""
    # generate_image()
    # function_call()
    # generate_text("Tell me about Python programming")
    analyze_audio("https://audio-samples.github.io/samples/mp3/blizzard_unconditional/sample-0.mp3")

if __name__ == "__main__":
    main()
