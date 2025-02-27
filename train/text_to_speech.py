import torch
import torchaudio
from transformers import pipeline

if __name__ == '__main__':
    text = "Python is a high-level, general-purpose programming language."

    pipe = pipeline("text-to-speech", model="suno/bark-small", device="cuda")

    output = pipe(text)

    audio_tensor = torch.from_numpy(output["audio"])

    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # 转换为 [1, samples]

    output_file = "speech.wav"
    torchaudio.save(output_file, audio_tensor, sample_rate=output["sampling_rate"])
