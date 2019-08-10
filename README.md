# CarND-03-Neural-Network-MiniFlow

Udacity Self-Driving Car Engineer Nanodegree: MiniFlow.

## Further Reading

- [Partial derivatives](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivatives/v/partial-derivatives-introduction)
- [Gradients](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/gradient-and-directional-derivatives/v/gradient)
- [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.fowl6fvfk)
- [Vector, Matrix, and Tensor Derivatives](http://cs231n.stanford.edu/vecDerivs.pdf)

## Forward Propagation

```
def forward(self):
    x_value = self.inbound_nodes[0].value
    y_value = self.inbound_nodes[1].value
    self.value = x_value + y_value
```

## Linear Transform

Linear algebra nicely reflects the idea of transforming values between layers in a graph.

```python
def forward(self):
    inputs = self.inbound_nodes[0].value
    weights = self.inbound_nodes[1].value
    bias = self.inbound_nodes[2].value
    self.value = bias
    for x, w in zip(inputs, weights):
        self.value += x * w
```

```python
def forward(self):
    X = self.inbound_nodes[0].value
    W = self.inbound_nodes[1].value
    b = self.inbound_nodes[2].value
    self.value = np.dot(X, W) + b
```

## Sigmoid Function

```python
def _sigmoid(self, x):
    return 1. / (1. + np.exp(-x)) # the `.` ensures that `1` is a float

def forward(self):
    input_value = self.inbound_nodes[0].value
    self.value = self._sigmoid(input_value)
```

## MSE Cost

```python
def forward(self):
    """
    Calculates the mean squared error.
    """
    # NOTE: We reshape these to avoid possible matrix/vector broadcast
    # errors.
    #
    # For example, if we subtract an array of shape (3,) from an array of shape
    # (3,1) we get an array of shape(3,3) as the result when we want
    # an array of shape (3,1) instead.
    #
    # Making both arrays (3,1) insures the result is (3,1) and does
    # an elementwise subtraction as expected.
    
    y = self.inbound_nodes[0].value.reshape(-1, 1)
    a = self.inbound_nodes[1].value.reshape(-1, 1)
    m = self.inbound_nodes[0].value.shape[0]

    diff = y - a
    self.value = np.mean(diff**2)
```

## Gradient Descent

Empirically, Learning rate in the range 0.1 to 0.0001 work well. 
The range 0.001 to 0.0001 is popular, as 0.1 and 0.01 are sometimes too large.

```python
def gradient_descent_update(x, gradx, learning_rate):
    x = x - learning_rate * gradx
    return x
```

## Backpropagation

A composition of functions `MSE(Linear(Sigmoid(Linear(X, W1, b1)), W2, b2), y)`

```python
class Sigmoid(Node)
    def backward(self):
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
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

## Example

Create a neural network.

```python
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = MSE(y, l2)
```

Train 
```python
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Reset value of X and y Inputs
        X.value = X_batch
        y.value = y_batch

        # Step 2
        forward_and_backward(graph)

        # Step 3
        sgd_update(trainables)

        loss += graph[-1].value

    print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))
```

