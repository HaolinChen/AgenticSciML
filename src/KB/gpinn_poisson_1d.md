# Gradient-Enhanced Physics-Informed Neural Networks (gPINN) for 1D Poisson Equation

**Keywords**: ["PDE", "elliptic", "second-order", "forward-problem", "poisson", "1D", "dirichlet", "PINN", "MLP", "strong-form", "adam", "mse", "deepxde"]

**Problem:** Standard Physics-Informed Neural Networks (PINNs) often have limited accuracy even with many training points when solving PDEs. This method addresses the 1D Poisson equation -∆u = f on the interval [0, π] with Dirichlet boundary conditions, where the source term f contains multiple frequency components (sine terms with frequencies 1, 2, 3, 4, and 8). The challenge is to accurately capture these multi-frequency features with fewer collocation points.

**Issues addressed:**
- Limited accuracy of standard PINNs requiring many training points
- Slow convergence rates in PDE residual minimization
- Poor prediction accuracy for solution derivatives
- Inefficient use of training data in physics-informed learning

## Key Method

Gradient-enhanced PINN (gPINN) improves upon standard PINN by incorporating gradient information of the PDE residual into the loss function. The key insight is that if the PDE residual f(x) is zero, then its spatial derivatives ∇f(x) should also be zero.

For the 1D Poisson equation -u'' = f(x), the standard PINN loss is:

L_PINN = L_f = (1/N) Σ |(-u''(x_i) - f(x_i))|²

The gPINN extends this by adding the gradient of the residual:

L_gPINN = L_f + w · L_g

where L_g = (1/N) Σ |d/dx(-u''(x) - f(x))|² = (1/N) Σ |(-u''' - f'(x))|²

The additional gradient term enforces that the third derivative of the neural network solution matches the derivative of the source term, providing extra constraints that improve accuracy with the same number of training points.

**Key parameters:**
- Weight w for gradient loss term (critical: determines balance between residual and gradient losses)
- Number of residual collocation points
- Network architecture (depth and width)
- Hard vs. soft boundary condition enforcement

## Implementation

### PINN Implementation

```python
import deepxde as dde
from deepxde.backend import tf
import numpy as np

# Define the PDE residual for standard PINN
def PINNpde(x, y):
    # Compute second derivative using Hessian (automatic differentiation)
    dy_xx = dde.grad.hessian(y, x)

    # Define the multi-frequency source term f(x)
    # f = sin(x) + 2sin(2x) + 3sin(3x) + 4sin(4x) + 8sin(8x)
    f = 8 * tf.sin(8 * x)
    for i in range(1, 5):
        f += i * tf.sin(i * x)

    # Return PDE residual: -u'' - f = 0
    return -dy_xx - f


# Define the analytical solution (for testing/validation)
def func(x):
    sol = x + 1 / 8 * np.sin(8 * x)
    for i in range(1, 5):
        sol += 1 / i * np.sin(i * x)
    return sol


# Define geometry and PDE data
geom = dde.geometry.Interval(0, np.pi)

# Create PDE dataset:
# - 15 uniform collocation points inside domain
# - 0 boundary points (using hard constraints instead)
# - solution function for computing error metrics
data = dde.data.PDE(geom, PINNpde, [], 15, 0, "uniform", solution=func, num_test=100)

# Define neural network architecture
# 4 hidden layers with 20 neurons each
layer_size = [1] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)


# Hard constraint: enforce Dirichlet BCs through output transformation
# This ensures u(0) = 0 and u(π) = π exactly
def output_transform(x, y):
    # u(x) = x + tanh(x) * tanh(π-x) * N(x)
    # where N(x) is the raw network output
    # The product tanh(x) * tanh(π-x) vanishes at both boundaries
    # The linear term x ensures u(π) = π
    return x + tf.math.tanh(x) * tf.math.tanh(np.pi - x) * y


net.apply_output_transform(output_transform)

# Compile and train the PINN model
PINNmodel = dde.Model(data, net)
PINNmodel.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = PINNmodel.train(epochs=20000)

# Save results
dde.saveplot(losshistory, train_state, issave=True, isplot=False)
```

### gPINN Implementation

```python
# Define the PDE residual for gPINN with gradient information
def gPINNpde(x, y):
    # Compute second derivative
    dy_xx = dde.grad.hessian(y, x)
    # Compute third derivative (gradient of second derivative)
    dy_xxx = dde.grad.jacobian(dy_xx, x)

    # Define the source term f(x)
    f = 8 * tf.sin(8 * x)
    for i in range(1, 5):
        f += i * tf.sin(i * x)

    # Define the derivative of the source term f'(x)
    # f'(x) = cos(x) + 4cos(2x) + 9cos(3x) + 16cos(4x) + 64cos(8x)
    df_x = (
        tf.cos(x)
        + 4 * tf.cos(2 * x)
        + 9 * tf.cos(3 * x)
        + 16 * tf.cos(4 * x)
        + 64 * tf.cos(8 * x)
    )

    # Return two loss terms:
    # 1. Standard PDE residual: -u'' - f = 0
    # 2. Gradient of PDE residual: -u''' - f' = 0
    return [-dy_xx - f, -dy_xxx - df_x]


# Geometry and network architecture (same as PINN)
geom = dde.geometry.Interval(0, np.pi)

data = dde.data.PDE(geom, gPINNpde, [], 15, 0, "uniform", solution=func, num_test=100)

layer_size = [1] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)


# Apply the same output transformation for hard BC enforcement
def output_transform(x, y):
    return x + tf.math.tanh(x) * tf.math.tanh(np.pi - x) * y


net.apply_output_transform(output_transform)

# Compile gPINN model with loss weights
# loss_weights = [1, 0.01] means:
# - Weight 1 for the PDE residual loss
# - Weight 0.01 for the gradient of residual loss
gPINNmodel = dde.Model(data, net)
gPINNmodel.compile(
    "adam", lr=0.001, metrics=["l2 relative error"], loss_weights=[1, 0.01]
)
losshistory, train_state = gPINNmodel.train(epochs=20000)

dde.saveplot(losshistory, train_state, issave=True, isplot=False)
```

### Prediction and Visualization

```python
# Generate points for prediction
x = geom.uniform_points(1000)

# Predict solutions
pinn_pred = PINNmodel.predict(x)
gpinn_pred = gPINNmodel.predict(x)
exact = func(x)

# Predict derivatives using operator argument
pinn_deriv = PINNmodel.predict(x, operator=lambda x, y: dde.grad.jacobian(y, x))
gpinn_deriv = gPINNmodel.predict(x, operator=lambda x, y: dde.grad.jacobian(y, x))

# Ground truth derivative
def du_x(x):
    return 1 + np.cos(x) + np.cos(2 * x) + np.cos(3 * x) + np.cos(4 * x) + np.cos(8 * x)

exact_deriv = du_x(x)
```

## Critical Parameters

1. **Gradient loss weight (w = 0.01)**: Most critical parameter in gPINN. Controls the balance between minimizing the PDE residual and its gradient. Too large (e.g., w=1) can hurt performance; too small approaches standard PINN. Optimal value is problem-dependent and requires tuning.

2. **Number of collocation points (15)**: Fewer points needed compared to standard PINN to achieve same accuracy due to additional gradient information.

3. **Network architecture ([1] + [20] * 3 + [1])**: 4 hidden layers with 20 neurons each. Sufficient capacity for the multi-frequency solution.

4. **Activation function (tanh)**: Smooth activation enabling computation of higher-order derivatives through automatic differentiation.

5. **Output transformation**: Hard constraint enforcement for boundary conditions. Eliminates need for boundary loss term and improves accuracy compared to soft constraints.

6. **Learning rate (0.001)**: Standard Adam learning rate providing stable convergence.

7. **Training epochs (20000)**: Sufficient iterations for convergence of both PINN and gPINN.

8. **Optimizer (Adam)**: First-order gradient-based optimizer suitable for this problem scale.

**Performance improvements:**
- gPINN achieves ~10x lower L² relative error for u compared to PINN with same number of points
- gPINN achieves ~100x lower L² relative error for u' compared to PINN
- gPINN converges faster due to additional gradient constraints
- Computational cost: ~1.7x that of standard PINN (due to third derivative computation)
