import os
import torch
import numpy as np
from PIL import Image
from rembg import remove
from diffusers import StableDiffusionPipeline

# Load Stable Diffusion Text-to-Image Model
MODEL_ID = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID).to(device)

def remove_background(image_path):
    """Removes the background from an image."""
    image = Image.open(image_path).convert("RGBA")
    return remove(image)

def generate_lifestyle_background(prompt, width=512, height=512):
    """Generates a realistic AI-based lifestyle background."""
    generated_image = pipe(prompt, width=width, height=height).images[0]
    return generated_image

def blend_product_into_background(product_img, background_img, position=(50, 50), size=(200, 200)):
    """Resizes and pastes the product image onto the generated AI background."""
    product_resized = product_img.resize(size, Image.LANCZOS)
    background_img.paste(product_resized, position, product_resized)
    return background_img

def generate_final_image(product_path, output_path, prompt="Modern interior with natural lighting"):
    """Processes the product image and blends it into an AI-generated background."""
    product_no_bg = remove_background(product_path)
    background_img = generate_lifestyle_background(prompt)
    blended_image = blend_product_into_background(product_no_bg, background_img)
    blended_image.save(output_path)
    print(f"Saved generated image to {output_path}")

def main():
    """Handles batch processing of multiple images."""
    num_images = int(input("Enter the number of product images to process: "))
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_images):
        product_image_path = input(f"Enter path for product image {i+1}: ")
        output_image_path = os.path.join(output_dir, f"output_{i+1}.png")
        generate_final_image(product_image_path, output_image_path)
    
if __name__ == "__main__":
    main()
