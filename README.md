# AI-ML-DL-01-Mini-Neural-Network-Library
Udacity Self-Driving Car Engineer Nanodegree: MiniFlow.

## Forward Propagation

MiniFlow has two methods to help you define and then run values through your graphs: ``topological_sort()`` and ``forward_pass()``.

``topological_sort()`` returns a sorted list of nodes in which all of the calculations can run in series.

``forward_pass()`` actually runs the network and outputs a value.

## Linear Transform

Linear algebra nicely reflects the idea of transforming values between layers in a graph.

```python
def forward(self):
  X = self.inbound_nodes[0].value
  W = self.inbound_nodes[1].value
  b = self.inbound_nodes[2].value
  self.value = np.dot(X, W) + b
```

## Sigmoid Function

Neural networks take advantage of alternating transforms and activation functions to better categorize outputs. 

The sigmoid function is among the most common activation functions.

## Cost Function

I will calculate the cost using the mean squared error (MSE).

```python
def forward(self):
  y = self.inbound_nodes[0].value.reshape(-1, 1)
  a = self.inbound_nodes[1].value.reshape(-1, 1)
  m = self.inbound_nodes[0].value.shape[0]
  diff = y - a
  self.value = np.mean(diff**2)
```

## Gradient Descent

We adjust the old ``x`` pushing it in the direction of ``gradx`` with the force ``learning_rate``, by subtracting ``learning_rate * gradx``.

```python
def gradient_descent_update(x, gradx, learning_rate):
    x = x - learning_rate * gradx
    return x
```

## Backpropagation

The ``backward`` method sums the derivative (it's a normal derivative when there's only one variable) with respect to the only input over all the output nodes.

```python
def backward(self):
  # Initialize the gradients to 0.
  self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
  # Sum the derivative with respect to the input over all the outputs.
  for n in self.outbound_nodes:
    grad_cost = n.gradients[self]
    sigmoid = self.value
    self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost
```

## Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) is a version of Gradient Descent where on each forward pass a batch of data is randomly sampled from total dataset.

```python
def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.
    Arguments:
        `trainables`: A list of `Input` nodes representing weights/biases.
        `learning_rate`: The learning rate.
    """
    # Performs SGD
    # Loop over the trainables
    for t in trainables:
        # Change the trainable's value by subtracting the learning rate
        # multiplied by the partial of the cost with respect to this trainable.
        partial = t.gradients[t]
        t.value -= learning_rate * partial
```

First, the partial of the cost (C) with respect to the trainable ``t`` is accessed.

Second, the value of the trainable is updated.
