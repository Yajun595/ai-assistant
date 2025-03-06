import os
import base64
import logging
import asyncio
import requests
from io import BytesIO
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.client.session.aiohttp import AiohttpSession
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RAPID_KEY = os.getenv("RAPID_KEY")

# Initialize OpenAI client
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Initialize Telegram bot with proxy
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Utility Functions
def get_current_weather(location: str) -> str:
    """Get weather information for a specific location"""
    url = f"https://open-weather13.p.rapidapi.com/city/{location}/EN"
    headers = {
        "x-rapidapi-key": RAPID_KEY,
        "x-rapidapi-host": "open-weather13.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers)
    return response.json()

def determine_message(content: str) -> str:
    """
    Analyze user intent and return appropriate action type
    Returns format: 'action:content'
    """
    system_prompt = """You are a helpful assistant, need to understand user's intent and choice the appropriate functionality. 
    If user wants to inquire about the weather, return 'weather:city'. 
    If user wants to generate image, return 'image:image description'. 
    If it's a casual conversation, return 'chat:user input'"""
    
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        n=1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
    )
    return response.choices[0].message.content

def generate_text(prompt: str) -> str:
    """Generate text response using AI model"""
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        n=1,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Created by Yazhen. Your name is Gemi"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def generate_image(prompt: str) -> BytesIO:
    """Generate image based on text prompt"""
    response = client.images.generate(
        model="imagen-3.0-generate-002",
        prompt=prompt,
        response_format="b64_json",
        size="1024x1024",
        quality="hd",
        n=1,
    )
    img_data = base64.b64decode(response.data[0].b64_json)
    return BytesIO(img_data)

def analyze_audio(audio_file) -> str:
    """Transcribe and analyze audio content"""
    audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio"},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_data,
                        "format": "wav"
                    }
                }
            ],
        }]
    )
    return response.choices[0].message.content

# Message Processing
async def process_message(message, content):
    """Main message processing function"""
    # Determine user intent
    intent = determine_message(content)

    if intent.startswith('weather:'):
        # Handle weather queries
        location = intent.split(':', 1)[1].strip()
        wait_msg = await message.answer("🔍 Querying for weather information...")
        try:
            weather_data = get_current_weather(location)
            weather_response = (
                f"📍 The weather in {location}:\n"
                f"Temperature: {weather_data['main']['temp']}°F\n"
                f"Humidity: {weather_data['main']['humidity']}%\n"
                f"Weather condition: {weather_data['weather'][0]['description']}"
            )
            await message.answer(weather_response)
        except Exception as e:
            logging.error(str(e))
            await message.answer("Sorry, an error occurred while querying weather info. Please try again later")
        await wait_msg.delete()

    elif intent.startswith('image:'):
        # Handle image generation
        prompt = intent.split(':', 1)[1].strip()
        wait_msg = await message.answer("🖌️ Generating image. Please wait...")
        try:
            image_io = generate_image(prompt)
            image_io.seek(0)
            await message.answer_photo(
                types.BufferedInputFile(
                    image_io.read(),
                    filename="generated_image.png"
                ),
                caption=f"Prompt: {prompt}"
            )
        except Exception as e:
            logging.error(str(e))
            await message.answer("Sorry, an error occurred while generating image. Please try again later")
        await wait_msg.delete()

    else:
        # Handle general conversation
        chat_content = intent.split(':', 1)[1].strip()
        response = generate_text(chat_content)
        await message.answer(response)

# Message Handlers
@dp.message(F.voice | F.audio)
async def handle_audio(message: types.Message):
    """Handle voice and audio messages"""
    wait_msg = await message.answer("🎵 Analyzing audio...")
    try:
        file_id = message.voice.file_id if message.voice else message.audio.file_id
        file = await bot.get_file(file_id)
        file_content = await bot.download_file(file.file_path)
        
        analysis_result = analyze_audio(file_content)
        await message.answer(f"📝 Audio transcription:\n{analysis_result}")
        await process_message(message, analysis_result)
    except Exception as e:
        logging.error(str(e))
        await message.answer("Sorry, an error occurred while analyzing the audio. Please try again later")
    finally:
        await wait_msg.delete()

@dp.message(F.text == "/start")
async def cmd_start(message: types.Message):
    """Handle /start command with a welcoming message"""
    welcome_text = """
👋 *Welcome to AI Assistant Bot!*

I'm your AI assistant powered by Gemini, created by Yazhen. I can help you with:
• 💬 General conversation and questions
• 🌤 Weather information for any city
• 🎨 Generating images from descriptions
• 🎵 Transcribing voice messages

Just send me a message to get started!
Use /help to see all available commands.
    """
    await message.answer(welcome_text, parse_mode="Markdown")

@dp.message(F.text == "/help")
async def cmd_help(message: types.Message):
    """Handle /help command with detailed command instructions"""
    help_text = """
📚 *Available Commands*

Basic Commands:
• /start - Start a new conversation
• /help - Show this help message

Features:
1️⃣ *Chat*: Just type any message to chat with me
2️⃣ *Weather*: Ask about weather in any city
   Example: "What's the weather like in Tokyo?"
3️⃣ *Image Generation*: Ask me to create images
   Example: "Generate an image of a sunset"
4️⃣ *Voice Transcription*: Send any voice message

Created with ❤️ by Yazhen
    """
    await message.answer(help_text, parse_mode="Markdown")

@dp.message(F.text)
async def handle_message(message: types.Message):
    """Handle text messages"""
    content = message.text
    await process_message(message, content)

# Main function
async def main():
    """Start the bot"""
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
