#test convolution
import numpy as np
from CNN import Convolution2D, MaxPooling2D, Flatten,RELU,Softmax,FullyConnected

# input an array of (batch_size,  channels,height, width)=(1,1,3,3)
x = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])

conv=Convolution2D(1, 2,1,1)

output = conv.forward(x)
print(output)
relu=RELU()
output=relu.forward(output)
maxpooling=MaxPooling2D(2,2)
output=maxpooling.forward(output)
print(output)
flatten=Flatten()
output=flatten.forward(output)
fc=FullyConnected(2)
output=fc.forward(output)
Softmax=Softmax()
output=Softmax.forward(output)
output=fc.backward(output)
output=flatten.backward(output)
print(output)
output=maxpooling.backward(output)
output=relu.backward(output)
output=conv.backward(output,0.01)
print(output)

