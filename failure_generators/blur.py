import cv2
import numpy as np
from PIL import Image

def apply_gaussian_blur(image, kernel_size=11):
    """
    image: PIL Image
    kernel_size: odd number (higher = more blur)
    returns: blurred PIL Image
    """
    img_np = np.array(image)
    blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
    return Image.fromarray(blurred)
