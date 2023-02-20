import cv2
import os
from tqdm import tqdm
import numpy as np
def read_images(folder_path):
    images = []
    for filename in tqdm(os.listdir(folder_path)):
        # read only png images
        if not filename.endswith('.png'):
            continue
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        # Invert the grayscale image
        gray = cv2.bitwise_not(img)
        # Convert the image to binary using a threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img = thresh / 255.0
        # Resize the image to 28x28 pixels
        img = cv2.resize(img, (28, 28))
        #dilate image using cv2.dilate
        kernel = np.ones((2,2),np.uint8)
        img = cv2.dilate(img,kernel,iterations = 1)
        if img is not None:
            images.append(img)
    return images