import numpy as np
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm
from sklearn.metrics import log_loss
def getWindows(output_size,input_data, kernel_size, stride=1, padding=0, dilation=0):
    input_=input_data
    batch_size, channels, height, width = input_data.shape
    _, _, output_height, output_width = output_size
    if dilation != 0:
        input_ = np.insert(input_, range(1, input_data.shape[2]), 0, axis=2)
        input_ = np.insert(input_, range(1, input_data.shape[3]), 0, axis=3)
    # add padding to the input data
    padded_input = np.pad(input_, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
    batch_stride, channel_stride, height_stride, width_stride = padded_input.strides
    return np.lib.stride_tricks.as_strided(padded_input, shape=(batch_size, channels, output_height, output_width, kernel_size, kernel_size), strides=(batch_stride, channel_stride, stride * height_stride, stride * width_stride, height_stride, width_stride))


class Convolution2D:
    def __init__(self, no_of_filters,kernel_size, stride=1, padding=0):
        self.no_of_filters = no_of_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        #calculate the filters using Xavier initialization
        self.filters = None
        self.biases = None
    
    def forward(self, input_data):
        self.input_data = input_data
        batch_size,channels, height, width = input_data.shape
        self.batch_size=batch_size

        if self.filters is None:
            #initialize by Xavier initialization
            self.filters = np.random.randn(self.no_of_filters,channels, self.kernel_size, self.kernel_size) * np.sqrt(2 / (self.kernel_size * self.kernel_size * channels))
            #self.filters=np.ones((self.no_of_filters,channels, self.kernel_size, self.kernel_size))
            self.biases = np.zeros(self.no_of_filters)

        output_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
        output=np.zeros((batch_size, self.no_of_filters, output_height, output_width))
        #input_data = np.pad(input_data, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
        windows=getWindows(output.shape,input_data, self.kernel_size, self.stride, self.padding)
        
        output = np.einsum('bihwkl,oikl->bohw', windows, self.filters)

        # add bias to kernels
        output += self.biases[None, :, None, None]
        #print("output shape",output.shape)
        
        
        self.windows=windows
        
        return output
    def backward(self, dL_dout,learning_rate):
       ##
        padding = self.kernel_size - 1 if self.padding == 0 else self.padding

        dout_windows = getWindows(self.input_data.shape,dL_dout, self.kernel_size, padding=padding, stride=1, dilation=self.stride - 1)
        rot_kern = np.rot90(self.filters, 2, axes=(2, 3))

        db = np.sum(dL_dout, axis=(0, 2, 3))/self.batch_size
        dw = np.einsum('bihwkl,bohw->oikl', self.windows, dL_dout)/self.batch_size
        dx = np.einsum('bohwkl,oikl->bihw', dout_windows, rot_kern)
        # update weights and biases
        self.filters -= learning_rate * dw
        self.biases -= learning_rate * db
        return dx
    def save_weights(self):
       self.filters_matrix = self.filters
       self.biases_matrix = self.biases
      
class RELU:
    def forward(self, input_data):
        self.input_data = input_data
        output_data = np.maximum(0, input_data)
        return output_data

    def backward(self, d_out):
        d_input = d_out.copy()
        d_input[self.input_data <= 0] = 0
        self.d_input = d_input
        return d_input
class MaxPooling2D:

    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.input_map = None
    def forward(self,x):
        self.input=x
        batch_size,channels, height, width = x.shape
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        output=np.zeros((batch_size, channels, output_height, output_width))
        windows=getWindows(output.shape,x, self.pool_size, self.stride)
        #take the max value of each window
        output = np.amax(windows, axis=(4, 5))
        
        self.output_list=output

        return output
    # def backward(self,grad_output):

    #     grad_input = np.zeros(self.input_shape)
    #     batch_size,channels, height, width = self.input_shape
    #     output_height = (height - self.pool_size) // self.stride + 1
    #     output_width = (width - self.pool_size) // self.stride + 1
    
    #     for i in range(batch_size):
    #         for j in range(channels):
    #             for k in range(output_height):
    #                 for l in range(output_width):
    #                     #find the index of the max value in the window
    #                     (a, b) = np.unravel_index(self.windows[i, j, k, l].argmax(), self.windows[i, j, k, l].shape)
    #                     grad_input[i, j, k * self.stride + a, l * self.stride + b] = grad_output[i, j, k, l]
    #     return grad_input
    def backward(self, d_z):

        d_x= np.zeros(self.input.shape)
      
        for i in range(0,self.pool_size):
            for j in range(0,self.pool_size):
                sliced=self.input[:, :, i:i+d_z.shape[2]*self.stride:self.stride, j:j+d_z.shape[3]*self.stride:self.stride]
                # print('sliced shape: ',sliced.shape)
                mask=(sliced == self.output_list)
                d_x[:, :, i:i+d_z.shape[2]*self.stride:self.stride, j:j+d_z.shape[3]*self.stride:self.stride]+=mask*d_z
        return d_x
                
                

       
class Flatten:
    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = input_data.reshape(input_data.shape[0], -1)
        return self.output_data

    def backward(self, d_out):
        d_input = d_out.reshape(self.input_data.shape)
        self.d_input = d_input
        return d_input
class FullyConnected:
    def __init__(self, output_size):
        self.weights = None
        self.biases = None
        self.output_size = output_size
        
    def forward(self, input_data):
        if self.weights is None:
            self.weights = np.random.randn(input_data.shape[1], self.output_size) * np.sqrt(1 / input_data.shape[1])
            self.biases = np.zeros(self.output_size)
        self.input_data = input_data
        self.output_data = np.dot(input_data, self.weights) + self.biases
        return self.output_data
    def backward(self, d_out, learning_rate):
        batch_size,_=d_out.shape
        self.d_weights = np.dot(self.input_data.T, d_out)/batch_size
        self.d_biases = np.sum(d_out, axis=0)/batch_size
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases
        d_input = np.dot(d_out, self.weights.T)
        return d_input
    def save_weights(self):
        self.weight_matrix=np.copy(self.weights)
        self.bias_matrix=np.copy(self.biases)
class Softmax:
    def forward(self, input_data):
        self.input_data = input_data
        exp_values = np.exp(input_data-np.max(input_data, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output_data = probabilities
        return probabilities
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
        y_preds = np.array([]).reshape(0, 10)
        #y true without batch dimension for calculating accuracy
        y_true = np.array(y_true_batches).reshape(-1, 10)
        #print (y_true.shape)
        for i,j in tqdm(zip(image_batches,y_true_batches)):
            # print(i.shape)
            i=np.array(i)
            j=np.array(j)
            i = np.expand_dims(i, axis=1)
            output = self.forward(i)
            y_preds = np.vstack((y_preds, output))
            #calculate log loss
            delta=output-j
            self.backward(delta)

        return y_preds,y_true
            
            
            
            
        
    def predict(self, image):

        image = np.expand_dims(image, axis=1)
        #print (image.shape)
        output = self.forward(image)
        probabilities = output.copy()
        #convert output to one-hot encoding with the highest probability of 10 neurons
        output = np.eye(10)[np.argmax(output, axis=1)]
        return output, probabilities