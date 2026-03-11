# DENSE - NEURAL NETWORKS

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
## Loss
Loss measures how wrong a model is compared to what the right solution is. This enables us to later let the model learn through backpropagation.


Loss functions measure how wrong the model's predictions are. During training, the
optimizer tries to minimize this value by adjusting the weights. The choice of loss
function depends entirely on the task — regression, binary classification, or
multi-class classification. A wrong choice of loss function will cause the network
to optimize for the wrong thing entirely, regardless of how good the architecture is.

---

### MSE — Mean Squared Error

The default loss for regression tasks. Computes the average of the squared differences
between predictions and true values. Squaring does two things: it makes all errors
positive, and it penalises large errors disproportionately — a prediction that is 10
units off contributes 100 to the loss, while one that is 2 units off contributes only
4. This makes MSE sensitive to outliers.
```math
L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
```

Gradient with respect to each prediction:
```math
\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)
```

---

### MAE — Mean Absolute Error

Also used for regression. Instead of squaring the errors it takes the absolute value,
which means all errors are penalised equally regardless of size. This makes MAE more
robust to outliers than MSE — a single extreme prediction does not dominate the loss.
The downside is that the gradient is constant (always +1 or -1) which can cause the
optimizer to overshoot near the minimum.
```math
L = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
```

Gradient with respect to each prediction:
```math
\frac{\partial L}{\partial \hat{y}_i} = \frac{1}{n} \cdot \text{sign}(\hat{y}_i - y_i)
```

---

### Huber Loss

A hybrid of MSE and MAE. For errors smaller than a threshold $\delta$ it behaves like
MSE — smooth and differentiable near zero. For errors larger than $\delta$ it behaves
like MAE — linear and robust to outliers. This gives you the best of both: smooth
gradients near the minimum and outlier resistance far from it. The threshold $\delta$
is a hyperparameter you set before training.
```math
L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta\left(|y - \hat{y}| - \frac{1}{2}\delta\right) & \text{otherwise} \end{cases}
```

---

### BCE — Binary Cross-Entropy

The standard loss for binary classification — tasks where the output is one of two
classes. The model outputs a single probability $\hat{y} \in (0,1)$ via a sigmoid
activation. BCE measures how far that probability is from the true label (0 or 1).
When the true label is 1, the loss is $-\log(\hat{y})$ — it heavily penalises the
model for predicting a low probability when the answer was yes. When the true label
is 0, the loss is $-\log(1-\hat{y})$.
```math
L = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\right]
```

Gradient with respect to each prediction:
```math
\frac{\partial L}{\partial \hat{y}_i} = -\frac{1}{n}\left(\frac{y_i}{\hat{y}_i} - \frac{1 - y_i}{1 - \hat{y}_i}\right)
```

---

### CCE — Categorical Cross-Entropy

The standard loss for multi-class classification — tasks where the output is one of
$C$ classes. The model outputs a probability distribution over all classes via softmax.
CCE measures how far that distribution is from the true distribution, which is a
one-hot vector (1 for the correct class, 0 for all others). In practice only the
log probability of the correct class contributes to the loss — all other terms are
multiplied by zero.
```math
L = -\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})
```

Simplified for one-hot labels where only the true class $c^*$ is 1:
```math
L = -\frac{1}{n}\sum_{i=1}^{n} \log(\hat{y}_{i,c^*})
```

---

### KL Divergence — Kullback-Leibler Divergence

Measures how different one probability distribution is from another. Not typically
used as a training loss for basic classification, but essential in variational
autoencoders, knowledge distillation, and reinforcement learning from human feedback.
$P$ is the true distribution and $Q$ is the model's predicted distribution. It is
asymmetric — $KL(P \| Q) \neq KL(Q \| P)$ — which means the order you pass the
distributions in matters.
```math
KL(P \| Q) = \sum_{x} P(x) \log\frac{P(x)}{Q(x)}
```

Properties:
```math
KL(P \| Q) \geq 0 \quad \text{always}
```
```math
KL(P \| Q) = 0 \quad \text{if and only if } P = Q
```

---

### Focal Loss

A modification of BCE designed for severely imbalanced datasets — for example, object
detection where 99% of image regions contain no object. Standard BCE treats every
example equally, so the model gets overwhelmed by the easy majority class and ignores
the rare minority class. Focal loss down-weights easy examples by multiplying BCE by
$(1 - \hat{y})^\gamma$. When the model is already confident and correct, this factor
is near zero and the loss contribution is tiny. Hard, misclassified examples dominate
training instead.
```math
L = -\frac{1}{n}\sum_{i=1}^{n}(1 - \hat{y}_i)^\gamma \left[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\right]
```

Where $\gamma$ (gamma) is the focusing parameter, typically set to 2.

---

### Contrastive Loss

Used for training embedding models. Rather than predicting a class, the model learns
to map similar inputs close together and dissimilar inputs far apart in embedding space.
$D$ is the Euclidean distance between two embeddings, $y$ is 1 if the pair is similar
and 0 if dissimilar, and $m$ is a margin — dissimilar pairs only contribute to the
loss if they are closer than the margin.
```math
L = \frac{1}{n}\sum_{i=1}^{n} y_i D_i^2 + (1 - y_i)\max(0, m - D_i)^2
```

---

### Quick Reference

| Loss | Task | Output activation |
|---|---|---|
| MSE | Regression | None (linear) |
| MAE | Regression, robust to outliers | None (linear) |
| Huber | Regression, outliers present | None (linear) |
| BCE | Binary classification | Sigmoid |
| CCE | Multi-class classification | Softmax |
| KL Divergence | Distillation, VAEs, RLHF | Softmax |
| Focal | Imbalanced classification | Sigmoid |
| Contrastive | Embedding / similarity learning | L2 normalisation |


--- 
## Backpropagation
This is the training method we will be using. This method takes the partial derivative of the loss with regards to each weight, and adjusts the weight according to this gradient multiplied by the learning rate:

For a certain weight:
```math 
w_{i}=w_{i}-\frac{\partial loss}{\partial w_{i}} \times \eta
```




---
## The plan

The goal in this repo is to make a basic dense neural network that trains to fit a graph of 

```
```
```math
y=2x

```
```math 

y=x^{2}

```
```math

y=log(x) 

```


At that point, I would then try to find a real world problem to solve with this (not likely though).


For ease of coding, we shall use ReLU for activation, MSE for loss, and pure gradient descent for the Backpropagation, as I donno how to make an optimiser, or how exactly an optimiser works :O 
