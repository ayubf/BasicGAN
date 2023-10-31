# Types of Layers

Within the two neural networks in this project there are a few types of layers. Each layer performs a different operation given an input and yeilding an output using weights and bias terms of each node within the layer. These layers include:

## 1. [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
* Linear transformation: takes a vector as an input and multiplies each value by the weight and adds bias term
* Applies linear operation on an input vector such that given $W$ as the weight and $b$ as the bias term, 

$$
    \text{Output} = \{ W \cdot x + b  \: | \: x \in \text{input} \}
$$

## 2. [LeakyReLU](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU)
* LeakyReLU: takes a vector as input and replaces all values less than the leak parameter are replaced by multiplying them by the leak parameter.
* Applies non-linear operation on an input vector such that given $l$ as the leak parameter,

$$

\text{Output} = \{ \left\{
\begin{array}{ll}
      x  & \text{if} \: x \geq 0 \\
    l \cdot x & \text{otherwise} \\
\end{array}
\right. |\:x \in \text{input} \}
$$


## 3. [Sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)
* Sigmoid function: takes any real number as input and squahses it into a number between 0 and 1
* Applies non-linear operation on an input vector such that,


$$
\text{Output} = \{ \: \frac{1}{1+ e^{-x}} \: | \: x \in \text{input} \}
$$

## 4. [Tanh](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html)
* Hyperbolic tangent function: takes any real number as input and squashes it between -1 and 1
* Applies a non-linear operation on an input vector such that,
$$
\text{Output} = \{\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}} \: | \: x \in \text{input} \}
$$



