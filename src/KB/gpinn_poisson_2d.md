# Gradient-Enhanced Physics-Informed Neural Networks (gPINN) for 2D Poisson Equation

**Keywords**: ["PDE", "elliptic", "nonlinear", "forward-problem", "poisson", "2D", "dirichlet", "PINN", "MLP", "strong-form", "adam", "mse", "pytorch"]

**Problem:** Solving the 2D Poisson equation with steep gradients and highly localized features. Standard PINNs often have limited accuracy even with many training points, particularly struggling with solutions that have sharp peaks or steep gradients in localized regions. The problem involves finding u(x,y) satisfying -Δu = f on the domain [0,1]² with zero Dirichlet boundary conditions, where the exact solution u(x,y) = 2^(4a) x^a (1-x)^a y^a (1-y)^a with a=10 has a sharp peak at the center of the domain.

**Issues addressed:** Solutions with steep gradients, localized features (sharp peaks), high-frequency components, parameter-sensitive problems. gPINN significantly reduces errors in regions with steep gradients where standard PINNs struggle. For the 2D Poisson problem with a=10, gPINN achieves accurate predictions across the entire domain while PINN has large errors near the center peak.

## Key Method

Gradient-enhanced physics-informed neural networks (gPINNs) improve the accuracy of PINNs by incorporating gradient information of the PDE residual into the loss function. The core idea is that if the PDE residual f is zero everywhere, then its spatial derivatives must also be zero.

For the 2D Poisson equation -Δu = f, the standard PINN loss is:
- L_f = (1/N) Σ |∂²u/∂x² + ∂²u/∂y² + f(x,y)|²

gPINN adds two additional gradient loss terms:
- L_g1 = w_g1 (1/N) Σ |∂³u/∂x³ + ∂³u/∂x∂y² - ∂f/∂x|²
- L_g2 = w_g2 (1/N) Σ |∂³u/∂x²∂y + ∂³u/∂y³ - ∂f/∂y|²

Total loss: L = L_f + L_g1 + L_g2

The gradient terms enforce that the derivatives of the PDE residual are also minimized, which reduces fluctuations in the residual and leads to more accurate solutions. The weight hyperparameters (w_g1, w_g2) control the relative importance of the gradient losses and must be tuned for optimal performance.

## Implementation

```python
import numpy as np
import deepxde as dde
from deepxde.backend import tf

# Set the exponent parameter a=10 for creating steep gradients
a = 10

# Define the source term f(x,y) for the 2D Poisson equation
# This is derived from the exact solution u = (16*x*y*(1-x)*(1-y))^a
def f(x, y):
    # Second derivative in x direction
    u_xx = (
        16**a
        * a
        * (a * (1 - 2 * x) ** 2 - 2 * x**2 + 2 * x - 1)
        * ((x - 1) * x * (y - 1) * y) ** a
        / ((x - 1) ** 2 * x**2)
    )
    # Second derivative in y direction
    u_yy = (
        16**a
        * a
        * (a * (1 - 2 * y) ** 2 - 2 * y**2 + 2 * y - 1)
        * ((x - 1) * x * (y - 1) * y) ** a
        / ((y - 1) ** 2 * y**2)
    )
    # Return -Δu for the PDE: -Δu = f
    return -u_xx - u_yy

# Define the x-derivative of the source term for gradient enhancement
def f_x(x, y):
    return -(
        (
            16**a
            * a
            * (2 * x - 1)
            * ((x - 1) * x * (y - 1) * y) ** a
            * (
                -2 * a * (2 * a - 1) * (x - 1) ** 2 * x**2 * y
                + (a - 1) * a * (x - 1) ** 2 * x**2
                + (a - 1) * y**4 * (a * (1 - 2 * x) ** 2 - 2 * (x - 1) * x - 2)
                - 2 * (a - 1) * y**3 * (a * (1 - 2 * x) ** 2 - 2 * (x - 1) * x - 2)
                + y**2
                * (
                    (2 * a * (x - 1) * x + a) ** 2
                    + a * (-2 * (x - 1) * x * ((x - 1) * x + 3) - 3)
                    + 2 * (x - 1) * x
                    + 2
                )
            )
        )
        / ((x - 1) ** 3 * x**3 * (y - 1) ** 2 * y**2)
    )

# Define the y-derivative of the source term for gradient enhancement
def f_y(x, y):
    return -(
        (
            16**a
            * a
            * (2 * y - 1)
            * ((x - 1) * x * (y - 1) * y) ** a
            * (
                (a - 1) * x**4 * (a * (1 - 2 * y) ** 2 - 2 * (y - 1) * y - 2)
                - 2 * (a - 1) * x**3 * (a * (1 - 2 * y) ** 2 - 2 * (y - 1) * y - 2)
                + x**2
                * (
                    (2 * a * (y - 1) * y + a) ** 2
                    + a * (-2 * (y - 1) * y * ((y - 1) * y + 3) - 3)
                    + 2 * (y - 1) * y
                    + 2
                )
                - 2 * a * (2 * a - 1) * x * (y - 1) ** 2 * y**2
                + (a - 1) * a * (y - 1) ** 2 * y**2
            )
        )
        / ((x - 1) ** 2 * x**2 * (y - 1) ** 3 * y**3)
    )

# Output transform to enforce Dirichlet boundary conditions (hard constraint)
# This ensures u=0 on all boundaries automatically
def output_transform(x, y):
    x_in = x[:, 0:1]
    y_in = x[:, 1:2]
    # Multiply network output by zero at boundaries
    return x_in * y_in * (1 - x_in) * (1 - y_in) * y

# Define the PDE residual and gradient residuals for gPINN
def pde(x, y):
    x_in = x[:, 0:1]
    y_in = x[:, 1:2]

    # Standard PINN: second derivatives for Poisson equation
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, i=1, j=1)

    # gPINN enhancement: third derivatives for gradient of residual
    du_xxx = dde.grad.jacobian(du_xx, x, j=0)
    du_xxy = dde.grad.jacobian(du_xx, x, j=1)
    du_yyy = dde.grad.jacobian(du_yy, x, j=1)
    du_yyx = dde.grad.jacobian(du_yy, x, j=0)

    # Return three residuals: [PDE residual, x-gradient residual, y-gradient residual]
    return [
        du_xx + du_yy + f(x_in, y_in),           # Standard PINN loss
        du_xxx + du_yyx + f_x(x_in, y_in),       # Gradient in x direction
        du_xxy + du_yyy + f_y(x_in, y_in),       # Gradient in y direction
    ]

# Define the computational domain
geom = dde.geometry.Rectangle([0, 0], [1, 1])

# Create PDE data object with 400 training points in the domain
data = dde.data.PDE(geom, pde, [], num_domain=400)

# Define the neural network: 2 inputs, 3 hidden layers with 20 neurons each, 1 output
# Uses tanh activation and Glorot normal initialization
net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

# Apply hard boundary constraint transformation
net.apply_output_transform(output_transform)

# Create the gPINN model
gPINNmodel = dde.Model(data, net)

# Compile with Adam optimizer and gradient loss weights
# loss_weights=[1, 1e-5, 1e-5]: weights for [PDE loss, x-gradient loss, y-gradient loss]
# The gradient weights (1e-5) are critical for balancing the loss terms
gPINNmodel.compile("adam", lr=1.0e-3, loss_weights=[1, 1e-5, 1e-5])

# Train the model
losshistory, train_state = gPINNmodel.train(epochs=20000, callbacks=[])

# Generate test points on a uniform grid
def gen_test_x(num):
    x = np.linspace(0, 1, num)
    y = np.linspace(0, 1, num)
    l = []
    for i in range(len(y)):
        for j in range(len(x)):
            l.append([x[j], y[i]])
    return np.array(l)

# Exact solution for validation
def sol(t):
    x = t[:, 0:1]
    y = t[:, 1:2]
    return (16 * x * y * (1 - x) * (1 - y)) ** a

# Compute L2 relative error
x = gen_test_x(100)
print(
    "L2 relative error of u",
    dde.metrics.l2_relative_error(sol(x), gPINNmodel.predict(x)),
)
```

## Critical Parameters

1. **Gradient loss weights**: `loss_weights=[1, 1e-5, 1e-5]`
   - The weights for gradient loss terms are crucial for gPINN performance
   - For this 2D Poisson problem, optimal weight is 1e-5
   - Too large weights (e.g., 1) can degrade performance
   - Too small weights reduce to standard PINN
   - Requires grid search/tuning for each problem

2. **Number of residual points**: `num_domain=400`
   - gPINN achieves better accuracy than PINN with the same number of points
   - Can achieve comparable accuracy with fewer points than PINN

3. **Network architecture**: `[2] + [20] * 3 + [1]`
   - 3 hidden layers with 20 neurons each
   - Sufficient for capturing steep gradients with gradient enhancement

4. **Activation function**: `"tanh"`
   - Smooth activation needed for computing third-order derivatives

5. **Learning rate**: `lr=1.0e-3`
   - Standard Adam learning rate works well

6. **Training epochs**: `epochs=20000`
   - Sufficient for convergence with gradient enhancement

7. **Parameter a**: `a=10`
   - Controls steepness of the solution
   - Larger a creates sharper peaks and more challenging problems
   - This value creates a highly localized peak at domain center
