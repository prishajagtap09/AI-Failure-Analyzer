from PIL import Image
import os

def load_images_from_folder(folder_path):
    images = []
    image_names = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path,filename)
            image = Image.open(img_path).convert("RGB")
            images.append(image)
            image_names.append(filename)

    return images, image_names
