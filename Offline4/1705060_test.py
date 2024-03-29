import importlib
import os
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

module = importlib.import_module("1705060_train")
LeNet = getattr(module, "LeNet")
Convolution2D = getattr(module, "Convolution2D")
MaxPooling2D = getattr(module, "MaxPooling2D")
Flatten = getattr(module, "Flatten")
FullyConnected = getattr(module, "FullyConnected")
Softmax = getattr(module, "Softmax")
RELU= getattr(module, "RELU")
def read_images(folder_path):
    images = []
    names = []
    for filename in tqdm(os.listdir(folder_path)):
        # read only png images
        if not filename.endswith('.png'):
            continue
        image_name = filename
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        # Invert the grayscale image
        gray = cv2.bitwise_not(img)
        # Convert the image to binary using a threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # Normalize the pixel values between 0 and 1
        img = thresh / 255.0
        # Resize the image to 28x28 pixels
        img = cv2.resize(img, (28, 28))
        #dilate image using cv2.dilate
        kernel = np.ones((2,2),np.uint8)
        img = cv2.dilate(img,kernel,iterations = 1)
        if img is not None:
            images.append(img)
            names.append(image_name)
    return images, names
if __name__=="__main__":
    testing_path = sys.argv[1]
    batch_size = 1
    print ("Reading images...")
    images,names = read_images(testing_path)

   

    if os.path.exists("1705060_prediction.csv"):
        os.remove("1705060_prediction.csv")
    output_file = open("1705060_prediction.csv", "w")
    #write column name
    output_file.write("FileName,Digit")


    with open("1705060_model.pickle", "rb") as file:
        model = pickle.load(file)
    #predict output images by model with batch size 1
    y_pred = []
    print("Predicting ...")

    for i in tqdm(range(len(images))):
        y_pred_encoded,y_pred_prob=model.predict(np.array([images[i]]))
        y_pred.append(y_pred_encoded)
        
        predicted_digit = np.argmax(y_pred_encoded)
        output_file.write("\n"+names[i]+","+str(predicted_digit))

    output_file.close()




