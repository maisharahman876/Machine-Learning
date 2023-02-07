import cv2
import os
import numpy as np
import pickle
import pandas as pd
from CNN import LeNet
from tqdm import tqdm
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
        img = cv2.resize(img, (128, 128))
        if img is not None:
            images.append(img)
    return images
if __name__=="__main__":
    #load data
    training_path1 = "E:\\4-2\\training-a"
    training_path2 = "E:\\4-2\\training-b"
    training_path3 = "E:\\4-2\\training-c"
    batch_size = 32
    print("Reading images...")
    images = read_images(training_path1)
    #append image list
    images.extend(read_images(training_path2))
    images.extend(read_images(training_path3))
    #read output csv file and get y_true value
    
    df = pd.concat(map(pd.read_csv, ['E:\\4-2\\training-a.csv', 'E:\\4-2\\training-b.csv','E:\\4-2\\training-c.csv']))
    #print no of rows in df
    print(df.shape[0])
    #print no of images
    print(len(images))
   
    y_true = df['digit'].values
    
    #convert y_true to one hot encoding
    y_true = np.eye(10)[y_true]
   
    #print(y_true)
    #print all image sizes
    
    num_batches = len(images) // batch_size
    image_batches = [images[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]
    y_true_batches = [y_true[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]
    model=LeNet({ 'learning_rate': 0.01})
    print("Training ...")
    model.train(image_batches, y_true_batches)
    # #save model
    with open('1705060_model.pkl', 'wb') as f:
        pickle.dump(model, f)

