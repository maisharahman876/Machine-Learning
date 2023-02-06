import cv2
import os
import numpy as np
import pickle
from CNN import Convolution2D, MaxPooling2D, Flatten,RELU,Softmax,FullyConnected
import pandas as pd
class LeNet:
    def __init__(self, config):
        self.learning_rate = config['learning_rate']
        self.convolution_layer1 = Convolution2D(6, 5)
        self.max_pooling_layer1 = MaxPooling2D(2, 2)
        self.convolution_layer2 = Convolution2D(16, 5)
        self.max_pooling_layer2 = MaxPooling2D(2, 2)
        self.relu1 = RELU()
        self.relu2 = RELU()
        self.flatten_layer = Flatten()
        # Fully connected layer with 10 neurons
        self.fc1 = FullyConnected(120)
        self.fc2 = FullyConnected(84)
        self.fc3 = FullyConnected(10)
        self.softmax = Softmax()
    def forward(self, input):
        #first convolution layer
        output = self.convolution_layer1.forward(input)
        output = self.relu1.forward(output)
        output = self.max_pooling_layer1.forward(output)
        #second convolution layer
        output = self.convolution_layer2.forward(output)
        output = self.relu2.forward(output)
        output = self.max_pooling_layer2.forward(output)
        #flatten layer
        output = self.flatten_layer.forward(output)
        #first fully connected layer
        output = self.fc1.forward(output)
        #second fully connected layer
        output = self.fc2.forward(output)
        #third fully connected layer
        output = self.fc3.forward(output)
        #softmax layer
        output = self.softmax.forward(output)
        return output
    def backward(self, delta):
        delta = self.fc3.backward(delta, self.learning_rate)
        delta = self.fc2.backward(delta, self.learning_rate)
        delta = self.fc1.backward(delta, self.learning_rate)
        
        delta = self.flatten_layer.backward(delta)
        delta = self.max_pooling_layer2.backward(delta)
        delta = self.relu2.backward(delta)
        delta = self.convolution_layer2.backward(delta, self.learning_rate)

        delta = self.max_pooling_layer1.backward(delta)
        delta = self.relu1.backward(delta)
        delta = self.convolution_layer1.backward(delta, self.learning_rate)
        return delta
    def train(self, image_batches, y_true_batches):
        for i,j in zip(image_batches,y_true_batches):
            output = self.forward(i)
            delta=output-j
            self.backward(delta)
        #save learnt weights
        self.convolution_layer1.save_weights()
        self.convolution_layer2.save_weights()
        self.fc1.save_weights()
        self.fc2.save_weights()
        self.fc3.save_weights()
    def predict(self, images):
        images=np.array(images)
        output = self.forward(images)
        return output
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
    batch_size = 32
    images = read_images(training_path)
    #read output csv file and get y_true value
    df = pd.read_csv("E:\\4-2\\training-b.csv")
    y_true = df['digit'].values
    #convert y_true to one hot encoding
    y_true = np.eye(10)[y_true]
    #print(y_true)
    #print all image sizes
    
    num_batches = len(images) // batch_size
    image_batches = [images[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]
    y_true_batches = [y_true[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]
    model=LeNet({ 'learning_rate': 0.01})
    model.train(image_batches, y_true_batches)
    #save model
    with open('1705060_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    
   
   


        
       



    
    

