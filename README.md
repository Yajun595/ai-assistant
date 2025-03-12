# AI Assistant Application

A comprehensive AI application that provides various AI services through web interface and Telegram bot.

## Features

- **Chat Bot**: Intelligent conversation powered by Gemini model
- **Image Generation**: Creates images from text descriptions using Imagen model
- **Weather Information**: Real-time weather data via RapidAPI
- **Voice Transcription**: Converts voice messages to text
- **Vector Database**: Document search and Q&A using ChromaDB
- **Multiple Interfaces**:
  - Web API (FastAPI)
  - Telegram Bot

## Prerequisites

- Python 3.8+
- Gemini API Key
- OpenAI API Key
- Telegram Bot Token
- RapidAPI Key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Yajun595/ai-assistant.git
cd ai-assistant
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file with:
```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
RAPID_KEY=your_rapid_api_key
```

## Running the Applications

### Telegram Bot
```bash
python openai/telegram_bot.py
```

### Gemini Demo
```bash
python gemini/gemini_demo.py
```

### Web Server
```bash
uvicorn app:app --reload
```
Access at: `http://127.0.0.1:8000/`

## Components

### Telegram Bot Features
- Start conversation with `/start`
- Get help with `/help`
- Chat with AI assistant
- Generate images from descriptions
- Get weather information
- Voice message transcription

### Gemini Demo Features
- Text generation using Gemini model
- Image generation using Imagen
- Weather information retrieval
- Audio transcription and analysis

### Web API Endpoints
- **POST /summarize/**
  - Summarizes input text
  - Request: `{"prompt": "text to summarize"}`
  - Response: `{"summary": "summarized text"}`

- **POST /generate/image**
  - Generates images from text
  - Request: `{"prompt": "image description"}`
  - Response: `{"prompt": "original prompt", "image_base64": "base64 encoded image"}`

- **POST /generate/speech**
  - Converts text to speech
  - Request: `{"prompt": "text to speak"}`
  - Response: Audio file (WAV format)

## Project Structure
```bash
ai-assistant/
├── app.py                # FastAPI web application
├── gemini/
│   ├── gemini_demo.py    # Gemini demo script
│   └── telegram_bot.py   # Telegram bot implementation
├── vector_db/
│   └── chromadb_demo.py  # Vector database demo
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Technologies Used

- FastAPI - Web framework
- aiogram - Telegram Bot API framework
- OpenAI/Gemini - AI models
- Python-dotenv - Environment management
- Pillow - Image processing
- requests - HTTP client
