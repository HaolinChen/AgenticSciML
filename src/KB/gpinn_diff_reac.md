# Gradient-Enhanced Physics-Informed Neural Networks (gPINN) for Diffusion-Reaction Equation

**Keywords**: [PDE, parabolic, nonlinear, forward-problem, diffusion, reaction-diffusion, 1D, periodic, PINN, MLP, strong-form, adam, mse, tensorflow, deepxde]

**Problem:** Standard Physics-Informed Neural Networks (PINNs) enforce only the PDE residual to be zero during training, which can result in limited accuracy even with many training points. The gradient-enhanced PINN (gPINN) improves upon this by also enforcing the gradients of the PDE residual to be zero, leveraging the insight that if the PDE residual f(x) = 0, then its derivatives ∇f(x) must also equal zero. This approach provides better accuracy with fewer training points and improved predictions of solution derivatives.

**Issues addressed:**
- **Limited accuracy with standard PINNs**: gPINN achieves up to two orders of magnitude better accuracy than PINN with the same number of training points
- **Poor convergence**: gPINN has faster convergence rate than standard PINN
- **Inaccurate derivative predictions**: gPINN significantly improves accuracy of predicted spatial and temporal derivatives (∂u/∂x and ∂u/∂t)
- **Residual fluctuation**: By penalizing the gradient of the residual, gPINN reduces fluctuation and brings the residual closer to zero

## Key Method

The core innovation of gPINN is augmenting the standard PINN loss function with additional terms that penalize the gradients of the PDE residual.

For a general PDE of the form:
```
f(x; ∂u/∂x₁, ..., ∂u/∂xₐ; ∂²u/∂x₁∂x₁, ...; λ) = 0
```

**Standard PINN loss:**
```
L = wf·Lf + wb·Lb
```

where Lf is the PDE residual loss and Lb is the boundary condition loss.

**gPINN loss:**
```
L = wf·Lf + wb·Lb + Σᵢ wgᵢ·Lgᵢ
```

where the gradient loss terms are:
```
Lgᵢ(θ; Tgᵢ) = (1/|Tgᵢ|) Σ_{x∈Tgᵢ} |∂f/∂xᵢ|²
```

The key hyperparameter is the weight wg for the gradient loss terms. The optimal value is problem-dependent and determined via grid search.

For the 1D+time diffusion-reaction equation:
```
∂u/∂t = D·∂²u/∂x² + R(x,t)
```

gPINN adds two gradient loss terms:
- Lgₓ: gradient with respect to spatial variable x
- Lgt: gradient with respect to time variable t

This enforces:
- ∂f/∂x = 0
- ∂f/∂t = 0

in addition to the original PDE residual f = 0.

## Implementation

### PINN PDE Residual (Standard Approach)

```python
# Standard PINN: Only enforces the PDE residual
def PINNpde(x, y):
    """
    Compute PDE residual for diffusion-reaction equation.

    Args:
        x: Input coordinates [x_in, t_in] where x ∈ [-π, π], t ∈ [0, 1]
        y: Neural network output (predicted solution u)

    Returns:
        PDE residual: ∂u/∂t - ∂²u/∂x² - R(x,t)
    """
    x_in = x[:, 0:1]  # Spatial coordinate
    t_in = x[:, 1:2]  # Temporal coordinate

    # First-order time derivative
    dy_t = dde.grad.jacobian(y, x, j=1)

    # Second-order spatial derivative
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)

    # Source/reaction term
    r = tf.exp(-t_in) * (
        3 * tf.sin(2 * x_in) / 2
        + 8 * tf.sin(3 * x_in) / 3
        + 15 * tf.sin(4 * x_in) / 4
        + 63 * tf.sin(8 * x_in) / 8
    )

    # PDE residual: ∂u/∂t - ∂²u/∂x² - R = 0
    return [dy_t - dy_xx - r]
```

### gPINN PDE Residual (Gradient-Enhanced Approach)

```python
# gPINN: Enforces PDE residual AND its gradients
def gPINNpde(x, y):
    """
    Compute PDE residual and its gradients for gradient-enhanced training.

    Returns:
        [residual, ∂residual/∂x, ∂residual/∂t]
    """
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]

    # --- Original PDE residual components ---
    dy_t = dde.grad.jacobian(y, x, j=1)      # ∂u/∂t
    dy_xx = dde.grad.hessian(y, x, i=0, j=0) # ∂²u/∂x²
    r = tf.exp(-t_in) * (
        3 * tf.sin(2 * x_in) / 2
        + 8 * tf.sin(3 * x_in) / 3
        + 15 * tf.sin(4 * x_in) / 4
        + 63 * tf.sin(8 * x_in) / 8
    )

    # --- Gradient of residual with respect to x ---
    dy_tx = dde.grad.hessian(y, x, i=0, j=1)   # ∂²u/∂x∂t
    dy_xxx = dde.grad.jacobian(dy_xx, x, j=0)  # ∂³u/∂x³
    dr_x = tf.exp(-t_in) * (
        63 * tf.cos(8 * x_in)
        + 15 * tf.cos(4 * x_in)
        + 8 * tf.cos(3 * x_in)
        + 3 * tf.cos(2 * x_in)
    )  # ∂R/∂x

    # --- Gradient of residual with respect to t ---
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)   # ∂²u/∂t²
    dy_xxt = dde.grad.jacobian(dy_xx, x, j=1)  # ∂³u/∂x²∂t
    dr_t = -r  # ∂R/∂t = -R (since R = exp(-t)·...)

    # Return: [f, ∂f/∂x, ∂f/∂t]
    # where f = ∂u/∂t - ∂²u/∂x² - R
    return [
        dy_t - dy_xx - r,           # Original PDE residual
        dy_tx - dy_xxx - dr_x,      # Spatial gradient of residual
        dy_tt - dy_xxt - dr_t       # Temporal gradient of residual
    ]
```

### Hard Constraint for Boundary and Initial Conditions

```python
def icfunc(x):
    """
    Initial condition at t=0:
    u(x, 0) = Σᵢ₌₁⁴ sin(ix)/i + sin(8x)/8
    """
    return (
        tf.sin(8 * x) / 8
        + tf.sin(1 * x) / 1
        + tf.sin(2 * x) / 2
        + tf.sin(3 * x) / 3
        + tf.sin(4 * x) / 4
    )

def output_transform(x, y):
    """
    Transform network output to automatically satisfy:
    - Boundary conditions: u(-π, t) = u(π, t) = 0
    - Initial condition: u(x, 0) = icfunc(x)

    This eliminates the need for boundary/initial condition loss terms.
    """
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]

    # u = (x² - π²)(1 - exp(-t))·N(x,t) + icfunc(x)
    # where N(x,t) is the neural network output
    return (x_in - np.pi) * (x_in + np.pi) * (1 - tf.exp(-t_in)) * y + icfunc(x_in)
```

### Network Architecture and Training Setup

```python
# Domain definition
geom = dde.geometry.Interval(-np.pi, np.pi)  # Spatial domain: [-π, π]
timedomain = dde.geometry.TimeDomain(0, 1)    # Time domain: [0, 1]
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Training data configuration for gPINN
data = dde.data.TimePDE(
    geomtime,
    gPINNpde,      # Use gPINN PDE function (returns 3 residuals)
    [],            # No explicit boundary conditions (handled by output_transform)
    num_domain=50, # Number of collocation points for PDE residual
    solution=solution,  # Analytical solution for evaluation
    num_test=10000
)

# Neural network: 3 hidden layers with 20 neurons each
layer_size = [2] + [20] * 3 + [1]  # Input: (x,t), Output: u
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

# Apply output transformation to enforce BC/IC exactly
net.apply_output_transform(output_transform)

# Create model
gPINNmodel = dde.Model(data, net)

# Compile with loss weights
# loss_weights=[1, 0.1, 0.1] corresponds to weights for:
# [PDE residual, spatial gradient loss, temporal gradient loss]
gPINNmodel.compile(
    "adam",
    lr=0.0001,
    metrics=["l2 relative error"],
    loss_weights=[1, 0.1, 0.1]  # Key difference from PINN: weights for gradient terms
)

# Train
losshistory, train_state = gPINNmodel.train(epochs=100000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
```

### Comparison: Standard PINN vs gPINN

```python
# For PINN: Use PINNpde (1 residual) and loss_weights not specified
PINNmodel.compile("adam", lr=0.0001, metrics=["l2 relative error"])

# For gPINN: Use gPINNpde (3 residuals) and specify loss_weights
gPINNmodel.compile(
    "adam",
    lr=0.0001,
    metrics=["l2 relative error"],
    loss_weights=[1, 0.1, 0.1]  # [PDE, ∂PDE/∂x, ∂PDE/∂t]
)
```

## Critical Parameters

1. **Gradient loss weights** (`loss_weights=[1, wg_x, wg_t]`):
   - Controls the relative importance of gradient terms
   - For diffusion-reaction: `wg = 0.1` works well and is insensitive to exact value
   - For Poisson equation: optimal `wg = 0.01` (problem-dependent)
   - Determined by grid search

2. **Number of collocation points** (`num_domain`):
   - gPINN achieves 1% L² error with only 40 training points
   - Standard PINN requires >100 points for same accuracy
   - gPINN requires ~2-3× fewer points than PINN

3. **Network architecture**:
   - Input: 2 dimensions (x, t)
   - Hidden layers: 3 layers × 20 neurons
   - Activation: tanh
   - Output: 1 dimension (u)

4. **Learning rate**:
   - Standard: `lr = 0.0001` for Adam optimizer
   - For higher accuracy: use smaller learning rate (e.g., `1e-6`) with more iterations

5. **Training epochs**:
   - Standard: 100,000 iterations
   - For convergence to <0.01% error: up to 5×10⁶ iterations

6. **Output transformation**: Essential for automatically satisfying boundary/initial conditions, eliminating need for soft constraint losses

7. **Computational cost**:
   - gPINN is ~1.74× more expensive than PINN per iteration
   - But requires fewer points for same accuracy, often resulting in net speedup
