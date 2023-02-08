import cv2
import os
import numpy as np
import pickle
import pandas as pd
from CNN import LeNet
from tqdm import tqdm
from sklearn.metrics import log_loss,accuracy_score,f1_score
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
        # Normalize the pixel values between 0 and 1
        img = thresh / 255.0
        # Resize the image to 28x28 pixels
        img = cv2.resize(img, (28, 28))
        #dilate image using cv2.dilate
        kernel = np.ones((2,2),np.uint8)
        img = cv2.dilate(img,kernel,iterations = 1)
        if img is not None:
            images.append(img)
    return images
if __name__=="__main__":
    validation_ratio = 0.2
    #load data
    training_path1 = "E:\\4-2\\training-a"
    training_path2 = "E:\\4-2\\training-b"
    training_path3 = "E:\\4-2\\training-c"
    batch_size = 32
    print("Reading images...")

    images = read_images(training_path1)
    # split data into training and validation
    split_index = int(len(images) * (1 - validation_ratio))
    training_images = images[:split_index]
    validation_images = images[split_index:]
    
    df = pd.read_csv('E:\\4-2\\training-a.csv')
    y_true = df['digit'].values
    #convert y_true to one hot encoding
    training_y_true = y_true[:split_index]
    validation_y_true = y_true[split_index:]

    images = read_images(training_path2)
    # split data into training and validation
    split_index = int(len(images) * (1 - validation_ratio))
    training_images = training_images + images[:split_index]
    validation_images = validation_images + images[split_index:]
    df = pd.read_csv('E:\\4-2\\training-b.csv')
    y_true = df['digit'].values
    y_true=np.array(y_true)
    #split the numpy array into 2 parts with split_index and append to training_y_true and validation_y_true
    training_y_true=np.append(training_y_true,y_true[:split_index])
    validation_y_true = np.append(validation_y_true,y_true[split_index:])


    images = read_images(training_path3)
    # split data into training and validation
    split_index = int(len(images) * (1 - validation_ratio))
    training_images = training_images + images[:split_index]
    validation_images = validation_images + images[split_index:]
    df = pd.read_csv('E:\\4-2\\training-c.csv')
    y_true = df['digit'].values
    y_true=np.array(y_true)
    #split the numpy array into 2 parts with split_index and append to training_y_true and validation_y_true
    training_y_true=np.append(training_y_true,y_true[:split_index])
    validation_y_true = np.append(validation_y_true,y_true[split_index:])
    # print("Images",len(training_images))
    # print("y_true",len(training_y_true))
    #read output csv file and get y_true values of training data and validation data

    #convert training_y_true to one hot encoding
    training_y_true = np.eye(10)[training_y_true.astype(int)]
    validation_y_true = np.eye(10)[validation_y_true.astype(int)]
    #print("validation y true",validation_y_true.shape)
    
    num_batches = len(training_images) // batch_size
    image_batches = [training_images[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]
    y_true_batches = [training_y_true[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]
    y_true_batches = np.array(y_true_batches)
    
    model=LeNet({ 'learning_rate': 0.001})
    print("Training ...")
    y_true_all=np.array([])
    y_pred_all=np.array([])
    for i in range(20):
        print("Epoch: ", i)
        yp,yt=model.train(image_batches, y_true_batches)
        y_true_all=np.append(y_true_all,yt)
        y_pred_all=np.append(y_pred_all,yp)
    print("Training Loss: ", log_loss(y_true_all, y_pred_all))

    print("Validating ...")
    y_pred_val_all=[]
   
    for i in range(len(validation_images)):

        y_pred_encoded,y_pred_prob=model.predict(np.array([validation_images[i]]))
        # add the predicted value to y_pred_val_all list with reduce dimension
        y_pred_val_all.append(y_pred_prob[0])

    #print(np.array(y_pred_val_all))
    print("Validation Loss: ", log_loss(validation_y_true, y_pred_val_all))
    print("Validation Accuracy: ", accuracy_score(validation_y_true.argmax(axis=1), np.array(y_pred_val_all).argmax(axis=1)))
    print("Validation F1 Score: ", f1_score(validation_y_true.argmax(axis=1), np.array(y_pred_val_all).argmax(axis=1), average='macro'))
    # #save model
    with open('1705060_model.pkl', 'wb') as f:
        pickle.dump(model, f)

