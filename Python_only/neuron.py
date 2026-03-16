# Here we make the neuron.
# And the neural network in general
# we will only impliment relu as an activation as thats the only one in scope for now
# other functions perhaps will be implimented in the numpy part of the project
# NOTE: I am really bad at best practices, so most of this will either be really inefficient or downright wrong in some contexts, like the loss function probably
# yeah
import random as rand
import math

#activations 
class ReLU:
    def activate(self,input):
        output=0
        
        if input > 0:
            self.output=input
        else: 
            self.output=0
        return self.output 
    def derive(self,outputs):
        derivative=0
        
        if output > 0:
            derivative=1
        else:
            derivative=0
        return derivative 

class Linear:
    def activate(self,input):
        return input 
    def derive(self,outputs):
        return 1

#losses
class MSE:
    def getloss(self,predicted, expected):
        loss=(expected-predicted)**2
        return loss
    def derive(self,predicted, expected):
        derivative=2*(predicted-expected)
        return derivative 




class neuron:
    def __init__(self, num_inputs, actfn, lossfn, lr):
        
        # biases are initialised at 0 by convention cause weights are already randomised
        self.actfn=actfn
        self.lr=lr
        self.lossfn=lossfn 
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
        self.z=self.actfn.activate(self.z)
        return self.z 
    def train(self, data):
        # data is a list of the inputs, and expected outputs at index 0 and 1 respectively 
        inputs=data[0]
        prediction=self.forward(inputs)
        expectation=data[1]
        loss=self.lossfn.getloss(prediction, expectation)
        # get derivatives
        lossdv=self.lossfn.derive(prediction, expectation)
        actdv=self.actfn.derive(prediction)

        bdv=lossdv*actdv 
        wdv=[]

        for x in inputs:
            dv=bdv*x 
            wdv.append(dv)

        # now, learning occurs
        self.bias-=bdv*self.lr
        for i in range(len(self.weights)):
            self.weights[i]-=wdv[i]*self.lr 
        
        # now to test
        newprediction=self.forward(inputs)
        newloss=self.lossfn.getloss(newprediction, expectation)

    #for interoperability with layers
    def compoundtrain(self, inputs, backprop_derivative):
        # data is a list of the inputs, and expected outputs at index 0 and 1 respectively 
        inputs=inputs
        backpropdv=backprop_derivative
        actdv=self.actfn.derive(prediction)

        bdv=backpropdv*actdv 
        wdv=[]

        for x in inputs:
            dv=bdv*x 
            wdv.append(dv)

        # now, learning occurs
        self.bias-=bdv*self.lr
        for i in range(len(self.weights)):
            self.weights[i]-=wdv[i]*self.lr 
        




# basic testing, we test with a linear function 

#init dataset

numitems=20
dataset=[]
def algo(x):
    y=2*x
    return y 
for i in range(numitems):
    x=rand.randrange(-10,10)
    y=algo(x)
    dataset.append([[x],y])



repeats=10
act1=Linear()
loss1=MSE()
numinputs=1 
lr=0.01
'''
n1=neuron(numinputs, act1, loss1, lr)

print(n1.forward(dataset[3][0]))

for i in range(repeats):
    print("Repeat:", i)
    for data in dataset:
        n1.train(data)

print(n1.forward(dataset[3][0]),dataset[3][1])
'''




class layer:
    def __init__(self, numinputs, numneurons, actfn, lossfn, lr):

        self.neurons=[]
        # neurons in layer
        for i in range(numneurons):
            n=neuron(numinputs, actfn, lossfn, lr)
            self.neurons.append(n)
    def forward(self, inputs):
        self.outputs=[]
        for neuron in self.neurons:
            output=neuron.forward(inputs)
            self.outputs.append(output)
        return self.outputs


        
    def train(self, data):
        expectations=data[1]
        inputs=data[0]

        for neuron, expected in zip(self.neurons, expectations):
            sub_data=[inputs, expected]
            neuron.train(sub_data)


    # Unfinished
    def compoundtrain(self, inputs, backpropderivatives):
        expectations=data[1]
        inputs=data[0]

        for neuron, expected in zip(self.neurons, expectations):
            sub_data=[inputs, expected]
            neuron.compoundtrain(sub_data)




numitems2=20
dataset2=[]
def algo2(x):
    y1=x+1
    y2=x+2 
    y3=x+3
    return [y1,y2, y3] 
for i in range(numitems2):
    x=rand.randrange(-10,10)
    y=algo2(x)
    dataset2.append([[x],y])


repeats2=10
act2=Linear()
loss2=MSE()
numinputs2=1 
lr2=0.01


l1=layer(numinputs, 3, act1, loss1, lr)

print(l1.forward(dataset2[3][0]))

for i in range(repeats2):
    print("Repeat:", i)
    for data in dataset2:
        l1.train(data)

print(l1.forward(dataset2[3][0]),dataset2[3][1])



