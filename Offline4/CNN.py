import numpy as np
from numpy.lib.stride_tricks import as_strided

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
        batch_size, height, width,channels = input_data.shape

        if self.filters is None:
            self.filters = np.random.randn(self.kernel_size, self.kernel_size, channels, self.no_of_filters) * np.sqrt(1 / (self.kernel_size * self.kernel_size * channels))
            self.biases = np.zeros(self.no_of_filters)

        output_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
        output = np.zeros((batch_size, output_height, output_width,self.no_of_filters))
        input_data = np.pad(input_data, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')


        for k in range(batch_size):
            for l in range(self.no_of_filters):
                for i in range(output_height):
                    for j in range(output_width):
                        output[k, i, j, l] = np.sum(self.input_data[k, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size, :] * self.filters[:, :, :, l]) + self.biases[l]
        #print(output)
        return output
    def backward(self, dL_dout,learning_rate):
       ##
        print("baaaaaal ta pore korum") 
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
    def forward(self,x):
        batch_size, height, width, channels = x.shape
        self.input_shape=x.shape
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        out=np.zeros((batch_size, output_height, output_width, channels))
        self.out_map=np.zeros((batch_size, output_height, output_width, channels)).astype(np.int32)
        for k in range(batch_size):
            for l in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        out[k, i, j, l] = np.max(x[k, i * self.stride: i * self.stride + self.pool_size, j * self.stride: j * self.stride + self.pool_size, l])
                        self.out_map[k, i, j, l] = np.argmax(x[k, i * self.stride: i * self.stride + self.pool_size, j * self.stride: j * self.stride + self.pool_size, l])
        return out
    def backward(self,d_out):
        batch_size, height, width, channels = d_out.shape
        d_x=np.zeros(self.input_shape)

        for k in range(batch_size):
            for l in range(channels):
                for i in range(height):
                    for j in range(width):
                        d_x[k, i * self.stride: i * self.stride + self.pool_size, j * self.stride: j * self.stride + self.pool_size, l][np.unravel_index(self.out_map[k, i, j, l], (self.pool_size, self.pool_size))]+=d_out[k, i, j, l]
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
    def backward(self, d_out, learning_rate=0.01):
        self.d_weights = np.dot(self.input_data.T, d_out)
        self.d_biases = np.sum(d_out, axis=0)
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
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output_data = probabilities
        return probabilities
    def backward(self, d_out):
        d_input = d_out.copy()
        return d_input