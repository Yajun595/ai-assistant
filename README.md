# AI Assistant Web Application

A comprehensive web application that provides AI-powered text summarization, image generation, and speech synthesis services.

## Features

- **Text Summarization**: Summarizes long text into concise versions using a fine-tuned text summarization model based on Falconsai/text_summarization
- **Image Generation**: Creates images from text descriptions using Stable Diffusion Turbo model
- **Speech Synthesis**: Converts text to natural-sounding speech using Bark-small model

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU
- At least 8GB GPU memory

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

3. Download the pre-trained models:

The models will be downloaded automatically when first running the application

## Running the Application

1. Start the FastAPI server:

```bash
uvicorn app:app --reload
```

2. Open your browser and navigate to:

```bash
http://127.0.0.1:8000/
```


## API Endpoints

### POST /summarize/
Summarizes input text
- Request body: `{"prompt": "text to summarize"}`
- Returns: `{"summary": "summarized text"}`

### POST /generate/image
Generates an image from text prompt
- Request body: `{"prompt": "image description"}`
- Returns: `{"prompt": "original prompt", "image_base64": "base64 encoded image"}`

### POST /generate/speech  
Converts text to speech
- Request body: `{"prompt": "text to speak"}`
- Returns: Audio file (WAV format)

## Model Training

### Text Summarization Training
The text summarization model is trained on the Billsum dataset:

```bash
cd train
python train_summarization.py
```

### Image Generation
Uses the pre-trained Stable Diffusion Turbo model:

```bash
python text_to_image.py
```

### Speech Synthesis
Uses the pre-trained Bark-small model:

```bash
python text_to_speech.py
```

## Project Structure

```bash
generative-ai/
├── app.py # FastAPI application
├── requirements.txt # Dependencies
├── train/ # Model training scripts
├── models/ # Pre-trained model files
└── README.md # Project documentation
```

## Technologies Used

- FastAPI - Web framework
- Transformers - For text summarization and speech synthesis
- Diffusers - For image generation
- PyTorch - Deep learning framework
- TorchAudio - Audio processing
