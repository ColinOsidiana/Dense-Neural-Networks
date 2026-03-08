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

$${\large y= activation(z) }$$

Where:

$${\large z= (\sum_{i=0}^{n} x_{i}\cdot w_{i})+b }$$

And activation is any relevant activation function.

### Weights
Weights are multiplied with their corresponding inputs, and basically determine how much influence a certain input will have on the result of the neuron.

### Biases
The biases offset the weighted sum of all the inputs by a certain amount, influencing the output.


## Activation Functions

Activation functions introduce non-linearity into a neural network. Without them, stacking
multiple layers is mathematically equivalent to a single linear transformation — no matter
how deep the network, it could only learn linear relationships. Activation functions are
what give neural networks the ability to approximate any function.

---

### ReLU — Rectified Linear Unit

The default choice for hidden layers. Fast, simple, and effective. Outputs the input
directly if positive, zero otherwise. Suffers from the dying neuron problem — if a
neuron's input is always negative, its gradient is always zero and it never learns.
```math
f(x) = \max(0, x)
```
```math
f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}
```

---

### Leaky ReLU

A fix for the dying neuron problem. Instead of a hard zero for negative inputs, it allows
a small gradient controlled by $\alpha$ (typically 0.01). The neuron can still learn even
when its input is negative.
```math
f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}
```
```math
f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \leq 0 \end{cases}
```

---

### Sigmoid

Squashes any input into a value between 0 and 1, making it interpretable as a probability.
Used for binary classification output layers. Rarely used in hidden layers because it
suffers from vanishing gradients — when inputs are very large or very small, the gradient
approaches zero and learning stops.
```math
f(x) = \frac{1}{1 + e^{-x}}
```
```math
f'(x) = f(x)(1 - f(x))
```

---

### Tanh — Hyperbolic Tangent

Similar to sigmoid but output is between -1 and 1, making it zero-centred. This makes
it better than sigmoid for hidden layers because gradients are less likely to all push
in the same direction. Still suffers from vanishing gradients at the extremes. Used
inside LSTM and GRU gates.
```math
f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```
```math
f'(x) = 1 - f(x)^2
```

---

### Softmax

Converts a vector of raw scores into a probability distribution — all outputs are between
0 and 1 and sum to exactly 1. Used exclusively as the final layer in multi-class
classification. The numerically stable version subtracts the maximum before exponentiating
to prevent overflow.
```math
f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
```

Numerically stable version:
```math
f(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{n} e^{x_j - \max(x)}}
```

---

### GELU — Gaussian Error Linear Unit

The modern replacement for ReLU in Transformer architectures. Where ReLU hard-gates
inputs at zero, GELU weights inputs by the probability that they are positive under a
Gaussian distribution — producing a smooth, non-monotonic curve. Used in BERT, GPT-2,
and most large language models. $\Phi(x)$ is the cumulative distribution function of
the standard normal distribution.
```math
f(x) = x \cdot \Phi(x)
```

Practical approximation used in code:
```math
f(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)
```

---

### SiLU — Sigmoid Linear Unit (Swish)

A smooth, non-monotonic activation that multiplies the input by its own sigmoid. Like
GELU, it allows small negative values to pass through rather than zeroing them completely.
Used in modern vision models and as the gating component inside SwiGLU.
```math
f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
```
```math
f'(x) = f(x) + \sigma(x)(1 - f(x))
```

---

### SwiGLU

Not a single activation but a gated feedforward unit. One linear projection is passed
through SiLU, then multiplied element-wise ($\odot$) with a second linear projection.
The second projection acts as a learned gate that controls how much of the first
projection passes through. Used in LLaMA, Qwen, PaLM, and most modern LLMs. Consistently
outperforms plain GELU in large models.
```math
f(x) = \text{SiLU}(W_1 x) \odot W_2 x
```

---

### Quick Reference

| Activation | Output range | Use for |
|---|---|---|
| ReLU | $[0, \infty)$ | Default hidden layers in CNNs and dense networks |
| Leaky ReLU | $(-\infty, \infty)$ | When dying ReLU is suspected |
| Sigmoid | $(0, 1)$ | Binary classification output only |
| Tanh | $(-1, 1)$ | LSTM and GRU gates, RNNs |
| Softmax | $(0, 1)$, sums to 1 | Multi-class classification output only |
| GELU | $\approx(-0.17, \infty)$ | Transformer hidden layers |
| SiLU | $\approx(-0.28, \infty)$ | Modern vision models, gating |
| SwiGLU | $(-\infty, \infty)$ | LLM feedforward layers |



---
## Backpropagation
This is the training method we will be using. This method takes the partial derivative of the loss with regards to each weight, and adjusts the weight according to this gradient multiplied by the learning rate:

For a certain weight:
$$ w_{i}=w_{i}-\frac{\partial loss}{\partial w_{i}} \times \eta $$



---
## The plan

The goal in this repo is to make a basic dense neural network that trains to fit a graph of $ y=2x , y=x^{2} , y=log(x) $
At that point, I would then try to find a real world problem to solve with this (not likely though).
