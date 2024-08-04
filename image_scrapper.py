import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO
import re

# Function to sanitize folder names (remove invalid characters)
def sanitize_folder_name(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

# Function to download images from a web page
def download_images(search_query, num_images=50):
    # Sanitize search query for use in directory name
    folder_name = sanitize_folder_name(search_query)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Create search URL
    search_url = f'https://www.google.com/search?hl=en&tbm=isch&q={search_query}'
    
    # Send request to the URL
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find image elements
    img_tags = soup.find_all('img')
    img_urls = [img['src'] for img in img_tags if 'src' in img.attrs]
    
    # Download images
    for i, img_url in enumerate(img_urls[:num_images]):
        try:
            img_response = requests.get(img_url)
            img = Image.open(BytesIO(img_response.content))
            img_format = img.format.lower()
            img_path = os.path.join(folder_name, f'accident_{i + 1}.{img_format}')
            img.save(img_path)
            print(f'Downloaded {img_path}')
        except Exception as e:
            print(f'Could not download image {img_url}. Error: {e}')

# Example usage
download_images('car accident', num_images=50)
