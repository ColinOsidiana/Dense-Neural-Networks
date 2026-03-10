# Here we make the neuron.
# And the neural network in general
# we will only impliment relu as an activation as thats the only one in scope for now
# other functions perhaps will be implimented in the numpy part of the project
# NOTE: I am really bad at best practices, so most of this will either be really inefficient or downright wrong in some contexts, like the loss function probably
# yeah
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
    def update(self):
        for n,w,b in zip(self.neurons, self.weights, self.biases):
            print(n.weights, w,"\n")
            print(n.bias, b,"\n")
            n.weights=w 
            n.bias=b
            print(n.weights, w,"\n")
            print(n.bias, b,"\n")

            




class ReLU:
    def forward(self, inputs):
        self.outputs=[]
        for i in inputs:
            if i > 0:
                self.outputs.append(i)
            else:
                self.outputs.append(0)
    def derive(self, input):
        self.derivatives=[]
        for i in input:
            if i > 0:
                self.derivatives.append(1)
            else:
                self.derivatives.append(0)


class MSE_loss:
    def forward(self, predicted, expected):
        self.loss=0
        for ex,pre in zip(expected,predicted):
            self.loss+=((ex-pre)**2)
    def derive(self, predicted, expected):
        self.gradient=0
        for ex,pre in zip(expected,predicted):
            # remember, the derivative is always y-hat - y. it determines the direction in which gradients move
            self.gradient+=(2*(pre-ex))
        self.gradient=self.gradient/len(predicted)
       



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
    def __init__(self, num_inputs, layer_dims, lr):
        self.dimensions=layer_dims
        self.layers=[]
        for i in range(len(layer_dims)):

            if i==0:
                self.layers.append(layer(num_inputs, layer_dims[i]))
            else:
                self.layers.append(layer(layer_dims[i-1], layer_dims[i]))
        self.weights=[]
        self.biases=[]
        self.lr=lr 


        for i in self.layers:
            self.weights.append(i.weights)
            self.biases.append(i.biases)
    def forward(self, inputs):
        self.results=[]
        self.buffer=inputs
        for i in self.layers:
            i.forward(self.buffer)
            if i == self.layers[-1]:
                for x, w in zip(inputs, self.weights):
                    self.buffer=i.outputs
                    self.results.append(self.buffer)

                
            else:
                activation=ReLU()
                activation.forward(i.outputs)
                
                self.buffer=activation.outputs
                self.results.append(self.buffer)

        self.outputs=self.buffer
    def forward2(self, inputs):
        self.buff=inputs
        self.results2=[]
        for weights, biases in zip(self.weights, self.biases):
            out=[]
            
            for w,b in zip(weights,biases):
                z=0
                
                for m, x in zip(w, self.buff):
                    z+=m*x 
                z+=b 
                out.append(z)
            self.buff=out
            self.results2.append(out)
        self.outputs2=self.buff
        return self.outputs2 


'''
    def train(self, inputs, expected):
        self.predictions = self.forward2(inputs)
        self.loss = MSE_loss()
        print(self.predictions, expected)
        self.loss.forward(self.predictions, expected)
        self.lossresult = self.loss.loss

        # upstream_deltas starts as [1] for output layer
        upstream_deltas = [1] * len(self.layers[-1].weights)

        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            weights = layer.weights
            biases = layer.biases

        # get inputs that fed into this layer
            if i == 0:
                layer_inputs = inputs
            else:
                layer_inputs = self.results2[i - 1]

        # get activation derivatives for this layer
            act = ReLU()
            act.derive(self.results2[i])
            actderiv = act.derivatives

        # output layer has no activation — override
            if i == len(self.layers) - 1:
                actderiv = [1] * len(actderiv)
                self.loss.derive(self.predictions, expected)
                loss_grad = self.loss.gradient
            else:
                loss_grad = 1

            # compute delta for each neuron in this layer
        # delta_k = loss_grad * actderiv[k] * upstream_deltas[k]
            deltas = []
            for k in range(len(weights)):
                delta = loss_grad * actderiv[k] * upstream_deltas[k]
                deltas.append(delta)
    
        # compute next upstream_deltas for the layer below
        # each neuron j in layer i-1 receives gradients from all neurons k in layer i
        # upstream_delta[j] = sum over k of (delta[k] * weights[k][j])
            if i > 0:
                next_upstream = []
                for j in range(len(layer_inputs)):
                    total = 0
                    for k in range(len(weights)):
                        total += deltas[k] * weights[k][j]
                    next_upstream.append(total)
                upstream_deltas = next_upstream

        # update weights and biases
            for k in range(len(weights)):
                for l in range(len(weights[k])):
                    wderivative = deltas[k] * layer_inputs[l]
                    weights[k][l] -= self.lr * wderivative
                bderivative = deltas[k] * 1
                biases[k] -= self.lr * bderivative

            layer.weights = weights
            layer.biases = biases
        print(self.forward2(inputs))
        print(expected)
    def repeatedtrain(self, inputs, expected, epochs):
        for i in range(epochs):
            print("epoch: ",i)
            self.train(inputs,expected)


'''








                    

                




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
lr=0.01
net1=net(1, layerdims, lr)

net1.forward2(dataset[1][0])

net1.repeatedtrain(*dataset[1], 100)


