# DENSE NEURAL NETWORKS

---
## What is it?

A dense neural network is a neural network where in each layer, every output of the previous layer is connected to each neuron of the current layer.
Basically **everything is connected**

In my understanding, this is the most basic type of neural network.

But what exactly is a neuron?

---

## The neuron

A neuron in this case is something that takes in inputs, decides how important each input is, then uses that information to generate an output.
In this case, I have the unconventional view of taking the activation functions to be part of the individual neuron, though, as I have heard, they are normally treated as their own layers.

How exactly does this work?

Each neuron takes in multiple inputs. Each input is multiplied by a corresponding weight, which acts as its *importance*.
These weighted inputs are then summed up, a bias added, then taken through an activation function to give an output.
The activation functions can vary depending on the need, but just mostly act as checking if the result has met a certain threshold or converting the output to a certain format.

The final formula of a neuron looks something like this.

$${\large y= activation((\sum_{i=0}^{n} x_{i}\cdot w_{i})+b) }$$

