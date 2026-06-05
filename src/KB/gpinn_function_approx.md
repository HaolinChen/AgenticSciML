# Gradient-Enhanced Physics-Informed Neural Networks (gPINN) for Function Approximation

**Keywords**: ["function_approximation", "PINN", "MLP", "adam", "mse", "tensorflow"]

**Problem:** Standard neural networks for function approximation can have limited accuracy even with many training points. This method addresses function approximation by leveraging gradient information of the target function in addition to function values. The approach improves accuracy by enforcing that both the function values and their derivatives match the true solution.

**Issues addressed:**
- Limited accuracy with standard neural network function approximation
- Inefficient use of training data (requiring many points to achieve good accuracy)
- Poor approximation of function derivatives even when function values are reasonably predicted

## Key Method

The gradient-enhanced neural network (gNN) extends standard neural network training by incorporating gradient information into the loss function. While a standard neural network minimizes:

```
L = (1/n) * Σ|u(xi) - û(xi)|²
```

The gNN uses an enhanced loss function:

```
L = (1/n) * Σ|u(xi) - û(xi)|² + wg * (1/n) * Σ|∇u(xi) - ∇û(xi)|²
```

where:
- `u(xi)` is the true function value at point xi
- `û(xi)` is the neural network prediction
- `∇u(xi)` and `∇û(xi)` are the true and predicted gradients
- `wg` is a weighting parameter for the gradient loss term

The key innovation is that by enforcing both function values and derivatives to match, the neural network learns a more accurate representation with fewer training points. The gradient loss term penalizes fluctuations and helps the network generalize better.

## Implementation

```python
import numpy as np
import deepxde as dde
from deepxde.backend import tf

# Define the target function to approximate
def func(x):
    """Target function: -(1.4 - 3x) * sin(18x) on [0,1]"""
    return -(1.4 - 3 * x) * np.sin(18 * x)

# Standard Neural Network (NN) approach
def NNfunc(x, y):
    """
    Loss function for standard NN
    Returns the residual between network output and target function
    y: neural network output
    """
    return y + (1.4 - 3 * x) * tf.sin(18 * x)

# Gradient-enhanced Neural Network (gNN) approach
def gNNfunc(x, y):
    """
    Loss function for gNN including gradient information
    Returns both function residual and derivative residual
    """
    # Compute derivative of network output using automatic differentiation
    dy_x = dde.grad.jacobian(y, x)

    return [
        # Function value residual
        y + (1.4 - 3 * x) * tf.sin(18 * x),
        # Derivative residual: dy/dx should match d/dx[-(1.4-3x)sin(18x)]
        dy_x + 18 * (1.4 - 3 * x) * tf.cos(18 * x) - 3 * tf.sin(18 * x),
    ]

# Define geometry (domain)
geom = dde.geometry.Interval(0, 1)

# Standard NN setup
data = dde.data.PDE(
    geom,
    NNfunc,           # Loss function
    [],               # No boundary conditions
    13,               # Number of training points in domain
    2,                # Number of points on boundary
    "uniform",        # Uniform sampling
    solution=func,    # Exact solution for error computation
    num_test=100      # Number of test points
)

# Neural network architecture: 1 input -> 3 hidden layers (20 neurons each) -> 1 output
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN([1] + [20] * 3 + [1], activation, initializer)

# Compile and train standard NN
NNmodel = dde.Model(data, net)
NNmodel.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = NNmodel.train(epochs=10000)

# Gradient-enhanced NN setup
data_gnn = dde.data.PDE(
    geom,
    gNNfunc,          # Enhanced loss with gradient
    [],
    13,
    2,
    "uniform",
    solution=func,
    num_test=100
)

net_gnn = dde.maps.FNN([1] + [20] * 3 + [1], activation, initializer)

# Compile gNN with loss weights
gNNmodel = dde.Model(data_gnn, net_gnn)
gNNmodel.compile(
    "adam",
    lr=0.001,
    metrics=["l2 relative error"],
    loss_weights=[1, 0.01]  # Weight for [function loss, gradient loss]
)
losshistory_gnn, train_state_gnn = gNNmodel.train(epochs=10000)

# Make predictions
x_test = geom.uniform_points(1000)
u_pred_nn = NNmodel.predict(x_test)
u_pred_gnn = gNNmodel.predict(x_test)

# Predict derivatives using automatic differentiation
du_pred_nn = NNmodel.predict(x_test, operator=lambda x, y: dde.grad.jacobian(y, x))
du_pred_gnn = gNNmodel.predict(x_test, operator=lambda x, y: dde.grad.jacobian(y, x))
```

## Critical Parameters

- **Network Architecture**:
  - Input dimension: 1
  - Hidden layers: 3 layers with 20 neurons each
  - Output dimension: 1
  - Activation function: `tanh`
  - Weight initialization: Glorot uniform

- **Training Configuration**:
  - Optimizer: Adam
  - Learning rate: 0.001
  - Training epochs: 10,000
  - Number of training points: 13 interior points + 2 boundary points

- **Loss Weights** (for gNN):
  - Function residual weight: 1.0
  - Gradient residual weight: 0.01
  - The gradient weight `wg` is critical and problem-dependent. In this example, `wg = 0.01` works well, but values of 0.1 and 1.0 were also tested

- **Performance Improvement**:
  - gNN achieves approximately 1% L² relative error with 15 training points
  - Standard NN has >10% error with the same number of points
  - gNN provides approximately one order of magnitude improvement in both function and derivative accuracy
