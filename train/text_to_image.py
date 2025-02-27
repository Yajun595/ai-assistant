from diffusers import StableDiffusionPipeline
import torch

if __name__ == '__main__':
    model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
    model_id2 = "stabilityai/sd-turbo"

    pipe = StableDiffusionPipeline.from_pretrained(model_id2, torch_dtype=torch.float16, use_safetensors=True,
                                                   safety_checker=None)
    pipe = pipe.to("cuda")
    prompt = """A steampunk airship flying over a Victorian-era city, gears and pipes visible on the ship, mechanical wings, steam clouds in the sky, warm golden sunset lighting, cinematic perspective, ultra-detailed, 4K
"""

    image = pipe(prompt).images[0]

    print("[PROMPT]: ", prompt)

    image.show()
