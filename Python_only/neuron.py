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
    def derive(self,output):
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

        return wdv

    #for interoperability with layers
    def compoundtrain(self, inputs, prediction, backprop_derivative):
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
        
        return wdv




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


    # for final layer 
    def train(self, i_o, expectations):
        predicted=i_o[1]
        inputs=i_o[0]

        lossdv=0
        for neuron, expected, prediction in zip(self.neurons, expectations, predicted):
            lossdv+=neuron.lossfn.derive(prediction, expected)
        lossdv=lossdv/len(self.neurons)
        print(lossdv)
        backdvs=[]
        
        for neuron, prediction in zip(self.neurons, predicted):
            
            dv=neuron.compoundtrain(inputs, prediction, lossdv)
            print("dv:", dv)
            backdvs.append(dv)
        return backdvs


    # Unfinished
    def compoundtrain(self, i_o, backpdv):
        predicted=i_o[1]
        inputs=i_o[0]

        
        backdvs=[]
        for neuron, prediction, neuron_no in zip(self.neurons, predicted, range(len(self.neurons))):
            bpdv=0
            for i in backpdv:
                print(i[neuron_no])
                bpdv+=i[neuron_no]
            dv=neuron.compoundtrain(inputs, prediction, bpdv)
            backdvs.append(dv)
        return backdvs




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


repeats2=1
act2=ReLU()
loss2=MSE()
numinputs2=1 
lr2=0.001


l1=layer(numinputs, 3, act2, loss1, lr)
l2=layer(3, 3, act1, loss1, lr)

o1=l1.forward(dataset2[3][0])
o2=l2.forward(o1)
print(o1,o2)

for i in range(repeats2):
    print("Repeat:", i)
    for data in dataset2:
        input=data[0]
        output1=l1.forward(input)
        output2=l2.forward(output1)
        expected=data[1]
        print("Output:",output1,output2)
        print("expected:", expected)

        results=[input, output1, output2]
        data1=results[1:]
        data2=results[0:2]
        print("data:",data1,data2)
        backderiv=l2.train(data1, expected)
        print("backderiv", backderiv)
        print(l1.compoundtrain(data2, backderiv))


o3=l1.forward(dataset2[3][0])
o4=l2.forward(o3)
print(o4, dataset2[3][1])
'''


# now we test two neurons, one after the other 
act3=ReLU()
loss3=MSE()

l2=neuron(1, 1, act1, loss1, lr)
l3=neuron(1, 1, act3, loss3, lr)

# input -> n3 -> n2 -> output 
input=dataset2[3][0]
r1=l3.forward(input)
r2=l2.forward(r1)
composite_result=[[input],[r1],[r2]]

'''
