import os
import pickle

import cv2
import numpy as np
import pandas as pd
# Load the model object from disk
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import importlib
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
    for filename in tqdm(os.listdir(folder_path)):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        # Invert the grayscale image
        gray = cv2.bitwise_not(img)
        # Convert the image to binary using a threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # Normalize the pixel values between 0 and 1
        img = thresh / 255
        # Resize the image to 28x28 pixels
        img = cv2.resize(img, (28, 28))
        kernel = np.ones((2,2),np.uint8)
        img = cv2.dilate(img,kernel,iterations = 1)
        if img is not None:
            images.append(img)
    return images
if __name__=="__main__":
    #load data
    training_path = "E:\\4-2\\training-d"
    batch_size = 1
    print ("Reading images...")
    images = read_images(training_path)
    #read output csv file and get y_true value
    df = pd.read_csv("E:\\4-2\\training-d.csv")
    y_true = df['digit'].values
    
    #convert y_true to one hot encoding
    y_true = np.eye(10)[y_true]
    with open("model_e28.pkl", "rb") as file:
        model = pickle.load(file)
    #predict output images by model with batch size 1
    y_pred = []
    print("Predicting ...")
    for i in tqdm(range(len(images))):
        y_pred_encoded,y_pred_prob=model.predict(np.array([images[i]]))
        y_pred.append(y_pred_encoded)
       
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[2])

    #calculate accuracy and f1 score
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print("Accuracy: ", accuracy)
    print("F1 score: ", f1)