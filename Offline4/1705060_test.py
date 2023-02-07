import pickle
import numpy as np
import pandas as pd
import cv2
import os
# Load the model object from disk

def read_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        # Invert the grayscale image
        gray = cv2.bitwise_not(img)
        # Convert the image to binary using a threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # Normalize the pixel values between 0 and 1
        img = thresh / 255
        # Resize the image to 28x28 pixels
        img = cv2.resize(img, (128, 128))
        if img is not None:
            images.append(img)
    return images
if __name__=="__main__":
    #load data
    training_path = "E:\\4-2\\training-b"
    batch_size = 1
    images = read_images(training_path)
    #read output csv file and get y_true value
    df = pd.read_csv("E:\\4-2\\training-b.csv")
    y_true = df['digit'].values
    
    #convert y_true to one hot encoding
    y_true = np.eye(10)[y_true]
    with open("1705060_model.pkl", "rb") as file:
        model = pickle.load(file)
    #predict output images by model with batch size 1
    y_pred = []
    for i in range(len(images)):
        y_pred.append(model.predict(np.array([images[i]])))
       
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[2])

      
    print(y_pred.shape, y_true.shape)
    #calculate accuracy
    accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)) / y_true.shape[0]
    print("Accuracy: ", accuracy)

