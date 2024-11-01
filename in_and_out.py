import inspect
from typing import List, Optional, Union

import numpy as np
import torch

import PIL
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
import os

def sd_outpaint(image, mask_image):

    prompt="snow background"
    guidance_scale=7.5
    num_samples = 1
    generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results

    images = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        #generator=generator,
        num_images_per_prompt=num_samples,
    ).images
    
    output = images[0].resize((width, height))
    return output 

def sd_inpaint(image, mask_image):

    prompt="consistent cityscape background"
    n_prompt="car, truck, vehicle"
    guidance_scale=7.5
    num_samples = 1
    generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results

    images = pipe(
        prompt=prompt,
        negative_prompt=n_prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        #generator=generator,
        num_images_per_prompt=num_samples,
    ).images
    
    output = images[0].resize((width, height))
    return output                              
    
if __name__ == '__main__':       

    device = "cuda"
    model_path = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        variant="fp16",
        torch_dtype=torch.float16,
    ).to(device)

    folder = "/home/cap6411.student1/CVsystem/assignment/hw8/cityscapes/val/img/"
    mask_folder = "./mymask/"
    for filename in tqdm(os.listdir(folder)):
        image = Image.open(os.path.join(folder,filename)).convert("RGB")
        width, height = image.size
        image = image.resize((512, 512))
    
        mask_image = Image.open(f"{mask_folder}mask_{filename}").convert("L")
        mask_image = mask_image.resize((512, 512))
        #mask_image = mask_image.filter(ImageFilter.MinFilter(size=25))
        invert_mask = ImageOps.invert(mask_image)
        invert_mask = invert_mask.filter(ImageFilter.MinFilter(size=25))
        mask_image = mask_image.filter(ImageFilter.MaxFilter(size=25))

        in_result = sd_inpaint(image, mask_image)
        out_result = sd_outpaint(image, invert_mask)

        in_result.save(f"./myoutput/inpaint_result_expand/inpaint_{filename}")
        out_result.save(f"./myoutput/outpaint_result_expand/outpaint_{filename}")
         





