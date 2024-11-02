import inspect
from typing import List, Optional, Union

import numpy as np
import torch

import PIL
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageOps, ImageFilter


def sd_outpaint():

    device = "cuda"
    model_path = "runwayml/stable-diffusion-inpainting"
    

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        variant="fp16",
        torch_dtype=torch.float16,
    ).to(device)

    image_path = "/home/cap6411.student1/CVsystem/assignment/hw8/cityscapes/val/img/val100.png"
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image = image.resize((512, 512))

    mask_path = "./mymask/mask_val100.png"
    mask_image = Image.open(mask_path).convert("L")
    mask_image = mask_image.resize((512, 512))
    mask_image = ImageOps.invert(mask_image)

    #mask_image = mask_image.filter(ImageFilter.MinFilter(size=25))

    mask_image.save("./myoutput/expand_mask.png")

    prompt="consistent snow background"
    #n_prompt="car, truck, vehicle"
    guidance_scale=7.5
    num_samples = 1
    generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results

    images = pipe(
        prompt=prompt,
        #negative_prompt=n_prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        #generator=generator,
        num_images_per_prompt=num_samples,
    ).images
    print(type(images[0]))
    output = images[0].resize((width, height))
    output.save("./myoutput/outpaint_test_val100.png")

if __name__ == '__main__':

    sd_outpaint()




