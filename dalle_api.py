# OpenAI Image Generation API: https://platform.openai.com/docs/guides/images/usage
# A good reading: https://www.makeuseof.com/generate-images-using-openai-api-dalle-python/
# A dall-e prompt book: https://dallery.gallery/the-dalle-2-prompt-book/

# Import modules
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from PIL import Image


# Load environment variables
load_dotenv()


# Create an OpenAI client
client = OpenAI()


# Define a class for image generation
class ImageGenerator:
    # Constructor
    def __init__(self):
        self.openai_api_key = os.environ["OPENAI_API_KEY"]

    # Generate images
    def generate_images(self, img_prompt, img_size="1024x1024", img_count=1, img_model='dall-e-2'):
        if self.openai_api_key:
            # Generate images
            response = client.images.generate(
                model=img_model,
                prompt = img_prompt,
                size = img_size,
                n = img_count
            )
            self.images = response.data
            
            # Get the URLs of the generated images
            self.image_urls = [image.url for image in self.images]
            print(self.image_urls)
            return self.image_urls
        else:
            print("OpenAI API key not found")

    # Download the images
    def download_images(self, folder, img_names):
        for img_url, img_name in zip(self.image_urls, img_names):
            img = requests.get(img_url)
            with open("{}/{}.png".format(folder, img_name), "wb") as f:
                f.write(img.content)
    
    # Edit the image
    # Note: Use the following tool to prepare the input image and the mask: https://labs.openai.com/editor
    # Use Upload button to upload the edited image, use Erase button to create a mask, and use Download button to download the edited image
    def edit_images(self, folder, img_name, mask_name, img_prompt, img_count=1, img_size="512x512", img_model='dall-e-2'):
        if self.openai_api_key:
            response = client.images.edit(
                image = open("{}/{}.png".format(folder, img_name), "rb"),
                mask = open("{}/{}.png".format(folder, mask_name), "rb"),
                prompt = img_prompt,
                n = img_count,
                size = img_size,
            )
            self.images = response.data
            self.image_urls = [image.url for image in self.images]
            print(self.image_urls)
            return self.image_urls
        else:
            print("OpenAI API key not found")
    
    # Generate a variation of an existing image
    def vary_images(self, folder, img_name, img_count=1, img_size="1024x1024", img_model='dall-e-2'):
        if self.openai_api_key:
            response = client.images.create_variation(
                image = open("{}/{}.png".format(folder, img_name), "rb"),
                n = img_count,
                size = img_size,
                model = img_model
                )
            self.images = response.data
            self.image_urls = [image.url for image in self.images]
            print(self.image_urls)
            return self.image_urls
        else:
            print("OpenAI API key not found")


# Instantiate the class 
imageGen = ImageGenerator() 


# Generate images
# imageGen.generate_images(
#     img_prompt = "Film still, extreme wide shot of an elephant alone on the savannah, extreme long shot",
#     img_count = 2,
#     img_size = '1024x1024'
# )


# Download the images
# imageGen.download_images(
#     folder="images", 
#     img_names=["elephant1", "elephant2"])


# Inpainting
# Note: Use the following tool to prepare the input image and the mask: https://labs.openai.com/editor
# imageGen.edit_images(
#     folder = "images_edits",
#     img_name = "tower",
#     mask_name = "tower_masked",
#     img_prompt = "A university tower with a blue sky and a hot air balloon",
#     img_count = 2,
#     img_size = '1024x1024'
# )


# # Create a variation of an existing image
imageGen.vary_images(
    folder = "images",
    img_name = "elephant1",
    img_count = 2,
    img_size = '1024x1024'
)