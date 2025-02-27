import base64
from io import BytesIO

import torch
import torchaudio
from diffusers import AutoPipelineForText2Image
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# if you train your own model, you can use the path to the model 
# the model will be saved in the ./model folder, use the latest checkpoint
# TEXT_SUMMARIZATION_MODEL = "./model/checkpoint-2368"

TEXT_SUMMARIZATION_MODEL = "Falconsai/text_summarization"
tokenizer = AutoTokenizer.from_pretrained(TEXT_SUMMARIZATION_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(TEXT_SUMMARIZATION_MODEL)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

GENERATE_IMAGE_MODEL = "stabilityai/sd-turbo"
image_pipe = AutoPipelineForText2Image.from_pretrained(GENERATE_IMAGE_MODEL, torch_dtype=torch.float16,
                                                       variant="fp16").to("cuda")

GENERATE_SPEECH_MODEL = "suno/bark-small"
speech_pipe = pipeline("text-to-speech", model=GENERATE_SPEECH_MODEL, device="cuda")


class PromptRequest(BaseModel):
    prompt: str


@app.get("/")
async def read_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        raise HTTPException(status_code=404, detail="index.html not found")



@app.post("/summarize/")
async def summarize_text(request: PromptRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Text input is required")

    summary = summarizer(request.prompt, max_length=130, min_length=30, do_sample=False)
    return {"summary": summary[0]['summary_text']}


@app.post("/generate/image")
async def generate_image(request: PromptRequest):
    prompt = request.prompt
    image = image_pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"prompt": prompt, "image_base64": img_base64}


@app.post("/generate/speech")
async def generate_speech(request: PromptRequest):
    prompt = request.prompt

    speech = speech_pipe(prompt)

    audio_tensor = torch.from_numpy(speech["audio"])

    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    audio_bytes = BytesIO()
    torchaudio.save(audio_bytes, audio_tensor, sample_rate=speech["sampling_rate"], format="wav")
    audio_bytes.seek(0)

    return Response(
        media_type="audio/wav",
        content=audio_bytes.getvalue(),
        headers={"Content-Disposition": "attachment; filename=output.wav"})
