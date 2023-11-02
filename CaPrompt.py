# Import necessary libraries
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Import the correct modules from transformers and diffusers
import torch
from transformers import pipeline
from diffusers import DiffusionPipeline

# Define the image-to-text pipeline
image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# Initialize the pipeline for StableDiffusion
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, variant="fp16")  # Use full precision
pipe.to("cpu")  # Use CPU

# Define the sidebar options
st.sidebar.title("Navigation Menu")
st.sidebar.header("Image Input")
source = st.radio("Choose an image source", ["Upload File", "URL", "Selfie"])
st.sidebar.header("Prompt Input")
prompt_text = st.text_input("Enter a prompt for image generation")

# Initialize img to None
img = None

# Get the image from the user's chosen source
if source == "Upload File":
    uploaded_file = st.file_uploader("Choose an image file")
    if uploaded_file is not None:
        img_data = uploaded_file.read()
        img = Image.open(BytesIO(img_data))
elif source == "URL":
    url = st.text_input('Enter the URL of the image')
    if url != '':
        response = requests.get(url)
        img_data = response.content
        img = Image.open(BytesIO(img_data))
elif source == "Selfie":
    selfie = st.camera_input(label="Take a selfie")
    if selfie is not None:
        img_selfie = Image.open(selfie).convert('RGB')

        # Generate a caption for the selfie
        caption_selfie = image_to_text(img_selfie)

        # Print the caption
        st.write(f"Generated Caption: {caption_selfie}")

# Generate a caption for the image using the loaded model
if img is not None:
    caption = image_to_text(img)
    st.write(f"Generated Caption: {caption}")

# Generate an image using StableDiffusion based on the user's prompt text
if prompt_text != '':
    generated_image_tensor = pipe(prompt=prompt_text).images[0]

    # Convert tensor to PIL Image and display it
    generated_image = Image.fromarray(generated_image_tensor.numpy())
    st.image(generated_image)
