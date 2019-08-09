# CarND-03-Neural-Network-MiniFlow

Udacity Self-Driving Car Engineer Nanodegree: MiniFlow.

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

## Cost Function

I will calculate the cost using the mean squared error (MSE).

```python
    def forward(self):
        """
        Calculates the mean squared error.
        """
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)

        self.m = self.inbound_nodes[0].value.shape[0]
        # Save the computed output for backward.
        self.diff = y - a
        self.value = np.mean(self.diff**2)

    def backward(self):
        """
        Calculates the gradient of the cost.
        """
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff
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

