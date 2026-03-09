# Here we make the neuron.
# And the neural network in general
# we will only impliment relu as an activation as thats the only one in scope for now
# other functions perhaps will be implimented in the numpy part of the project
import random as rand
import math


class neuron:
    def __init__(self, num_inputs):
        
        # biases are initialised at 0 by convention cause weights are already randomised
        self.bias=0
        self.weights=[]
        for i in range(num_inputs):
            # Why are we using gaussian? because for ReLU, random distribution of weights is optimally normal distribution around zero
            # with a range of the square root of 2 over the number of inputs. other activations have different optimal weight distribution functions
            self.weights.append(rand.gauss(0, math.sqrt(2/num_inputs)))
    def forward(self, inputs):
        self.z=0 
        for x, w in zip(inputs, self.weights):
            self.z+=x*w 
        self.z+=self.bias


class layer:
    def __init__(self, num_inputs, num):
        self.neurons=[]
        for i in range(num):
            self.neurons.append(neuron(num_inputs))
        self.weights=[]
        self.biases=[]
        for i in self.neurons:
            self.weights.append(i.weights)
            self.biases.append(i.bias)
    def forward(self, inputs):
        self.outputs=[]
        for i in self.neurons:
            i.forward(inputs)
            self.outputs.append(i.z)




class ReLU:
    def forward(self, inputs):
        self.outputs=[]
        for i in inputs:
            if i > 0:
                self.outputs.append(i)
            else:
                self.outputs.append(0)
    def derive(self):
        self.derivatives=[]
        for i in self.outputs:
            if i > 0:
                self.derivatives.append(1)
            else:
                self.derivatives.append(0)


class MSE_loss:
    def forward(self, predicted, expected):
        self.loss=[]
        for ex,pre in zip(expected,predicted):
            self.loss.append((ex-pre)**2)



# testing individual layers
'''
inputs=[1,2,3]
layer1=layer(len(inputs), 4)
layer1.forward(inputs)
print(layer1.outputs)
print(layer1.weights)
print(layer1.biases)
'''
class net:
    def __init__(self, num_inputs, layer_dims):
        self.layers=[]
        for i in range(len(layer_dims)):

            if i==0:
                self.layers.append(layer(num_inputs, layer_dims[i]))
            else:
                self.layers.append(layer(layer_dims[i-1], layer_dims[i]))
        self.weights=[]
        self.biases=[]

        for i in self.layers:
            self.weights.append(i.weights)
            self.biases.append(i.biases)
    def forward(self, inputs):

        self.buffer=inputs
        for i in self.layers:

            i.forward(self.buffer)
            if i == self.layers[-1]:
                self.buffer=i.outputs
            else:
                activation=ReLU()
                activation.forward(i.outputs)
                self.buffer=activation.outputs

        self.outputs=self.buffer 
    def train(self, inputs, expected):

        self.forward(inputs)
        self.predictions=self.outputs
        self.expected=expected
        self.loss=MSE_loss()
        self.loss.forward(self.predictions, self.expected)
        self.lossresult=self.loss.loss
        
# test if feedforward of neuron is working
'''
net_inputs=[1]
layerdims=[3,5,1]
net1=net(3, layerdims)
net1.forward(net_inputs)
print(net1.outputs)
print(net1.weights)
print(net1.biases)
'''





# create a dataset
size=100
dataset=[]

# y=2x

for i in range(size):
    x=[rand.randrange(-100,100)]
    y=[2*x[0]]
    dataset.append([x,y])
#print(dataset)

layerdims=[3,5,2,1]
net1=net(1, layerdims)

net1.train(*dataset[1])

print(net1.predictions)
print(net1.expected)
print(net1.lossresult)
