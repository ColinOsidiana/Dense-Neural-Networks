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
        self.bias=1
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
            self.gradient+=(2*(ex-pre))
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



    def train(self, inputs, expected):
        
       

        self.predictions=self.forward2(inputs)
        print(expected)
        print(self.predictions)
        self.expected=expected
        self.loss=MSE_loss()
        self.loss.forward(self.predictions, self.expected)
        self.lossresult=self.loss.loss
        print(self.lossresult)

        self.wprevderiv=[]
        self.bprevderiv=[]
        self.working=[]
        # range(lastindex, stop_before, step)
        for i in range(len(self.layers)-1, -1, -1):
            print("layer: ",i+1," ,start:", self.dimensions[i], "neuron(s)")
            layer=self.layers[i]
            weights=layer.weights
            biases=layer.biases
            act=ReLU()
            wbufferderiv=[]
            bbufferderiv=[]
            lossderiv=1
            actderiv=[]
            self.working=self.results2[i]
            for j in self.working:
                act.derive(self.working)
                actderiv=act.derivatives


            # ensures lossderiv does not affect anything unless on final layer
            if i==(len(self.layers)-1):
                self.loss.derive(self.predictions, self.expected)

                # ensures last layer is not treated as if it has activation
                for u in actderiv:
                    u=1
                lossderiv=self.loss.gradient

                # ensures previous derivative does not affect last layer 
                for m in range(len(weights)):
                    yup=[]
                    for n in range(len(weights[m])):
                        yup.append(1)
                    self.bprevderiv.append(1)
                    self.wprevderiv.append(yup)
            for k in range(len(weights)):
                yeah=[]
                for l in range(len(weights[k])):

                    
                    wderivative=lossderiv * actderiv[k] * weights[k][l] * self.wprevderiv[0][k]

                    yeah.append(wderivative)
                    weights[k][l]-=self.lr*wderivative
                bderivative=lossderiv*actderiv[k]*1*self.wprevderiv[0][k]

                biases[k]-=self.lr*bderivative


                wbufferderiv.append(yeah)
                bbufferderiv.append(bderivative)

            # update the derivatives buffers

            self.bprevderiv=bbufferderiv
            self.wprevderiv=wbufferderiv

            # update weights, biases
            layer.weights=weights
            layer.biases=biases

            


            print("layer: ",i+1," ,done:", self.dimensions[i], "neuron(s)")

        self.predictions2=self.forward2(inputs)
        self.loss.forward(self.predictions2, self.expected)
        self.lossresult2=self.loss.loss
        print(expected)
        print(self.predictions)
        print(self.predictions2)
        print(self.lossresult)
        print(self.lossresult2)
    def repeatedtrain(self, inputs, expected, epochs):
        for i in range(epochs):
            print("epoch: ",0)
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
lr=0.01
net1=net(1, layerdims, lr)

net1.forward2(dataset[1][0])

net1.repeatedtrain(*dataset[1], 100)


