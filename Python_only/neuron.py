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
        return self.outputs
    def derive(self, input):
        self.derivatives=[]
        for i in input:
            if i > 0:
                self.derivatives.append(1)
            else:
                self.derivatives.append(0)
        return self.derivatives


class MSE_loss:
    def forward(self, predicted, expected):
        self.loss=0
        for ex,pre in zip(expected,predicted):
            self.loss+=((ex-pre)**2)
        return self.loss
    def derive(self, predicted, expected):
        self.gradient=0
        for ex,pre in zip(expected,predicted):
            # remember, the derivative is always y-hat - y. it determines the direction in which gradients move
            self.gradient+=(2*(pre-ex))
        self.gradient=self.gradient/len(predicted)
        return self.gradient
       



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
    def __init__(self, num_inputs, layer_dims, actfn, lossfn, lr):
        self.lossfn=lossfn
        self.actfn=actfn
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
        self.buff=inputs
        self.results=[inputs]
        
        for weights, biases in zip(self.weights, self.biases):
            out=[]
            
            
            for w,b in zip(weights,biases):
                z=0
                
                for m, x in zip(w, self.buff):
                    z+=m*x 
                z+=b 
                out.append(z)
            
            
            if weights!=self.weights[-1]:
                out=self.actfn.forward(out)
                self.buff=out
                self.results.append(out)
            
            
        
        self.outputs2=self.buff
        return self.outputs2 


    def train(self, inputs, expected):
        predicted=self.forward(inputs)
        # initialises loss
        loss=self.lossfn.forward(predicted, expected)
        print(loss)
        # initialises derivative buffers as the loss derivative 
        weightderivatives=self.lossfn.derive(predicted, expected)
        # this is just as the previous comment said, but technically not the right way to do it

        prevadv=[1]*self.dimensions[-1]
        for i in range(len(self.layers) -1, -1, -1):
             
            cweights=self.weights[i]
            cbiases=self.biases[i]
            coutputs=self.results[i]
            
            aderivative=[]
            wderv=[]
            bderv=[]
            wdv=0
            # checks if  current layer is last layer 
            
            if i == (len(self.layers)-1):
                #if last layyer, all activation derivatives are set to 1, basically not affecting anything
                # else the derivative is the respective derivative of the activation function
                aderivative=[1]*len(coutputs)
            else:
                aderivative=self.actfn.derive(coutputs)
            # generated derivatives for weights
            for t in range(len(cweights)):
                neuron_wderv=[]
                for u in range(len(cweights[t])):
                    x=coutputs[u]
                    dv=weightderivatives*aderivative[u]*x 

                    neuron_wderv.append(dv)
                    wdv+=dv
                wderv.append(neuron_wderv)

            for n in range(len(cbiases)):
                bderv.append(weightderivatives*prevadv[n])

            for j in range(len(cweights)):
                for k in range(len(cweights[j])):
                    cweights[j][k]-=self.lr*wderv[j][k]
            for l in range(len(cbiases)):
                cbiases[l]-=self.lr*bderv[l]
            weightderivatives=wdv

            self.weights[i]=cweights
            self.biases[i]=cbiases
            prevadv=aderivative
        newprediction=self.forward(inputs)
        newloss=self.lossfn.forward(newprediction, expected)
        print(newloss)
    def repeatedtrain(self, inputs, expected, epochs):
        for i in range(epochs):
            print("epoch:", i+1)
            self.train(inputs,expected)




            

                
            








                    

                




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
lr=0.001
loss=MSE_loss()
activation=ReLU()
net1=net(1, layerdims,activation,loss , lr)

net1.forward(dataset[1][0])


net1.train(*dataset[1])
net1.train(*dataset[1])
net1.train(*dataset[1])
net1.train(*dataset[1])
net1.train(*dataset[1])
net1.train(*dataset[1])
