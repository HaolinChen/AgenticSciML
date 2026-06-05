# Gradient-Enhanced Physics-Informed Neural Networks for Inverse Diffusion-Reaction System

**Keywords**: ["PDE", "parabolic", "nonlinear", "forward-problem", "inverse-problem", "parameter-estimation", "reaction-diffusion", "1D", "dirichlet", "PINN", "MLP", "strong-form", "adam", "mse", "deepxde"]

**Problem:** This method addresses the inverse problem of inferring a space-dependent reaction rate k(x) in a one-dimensional diffusion-reaction system given sparse measurements of the solution u. The governing PDE is λ(∂²u/∂x²) - k(x)u = f, where λ = 0.01 is the diffusion coefficient, u is the solute concentration, f = sin(2πx) is the source term, and k(x) = 0.1 + exp[-0.5(x - 0.5)²/0.15²] is the unknown space-dependent reaction rate to be inferred. The challenge is to simultaneously learn both the solution field u(x) and the unknown parameter field k(x) from limited observational data.

**Issues addressed:** Standard PINNs have limited accuracy for inverse problems when inferring spatially-varying parameters, especially with sparse data. The gradient-enhanced approach improves accuracy of both the solution field and the inferred parameter field by incorporating gradient information of the PDE residual into the loss function. This method is particularly effective for inverse problems where the unknown parameter is a function rather than a constant, leading to more accurate parameter estimation with fewer training points.

## Key Method

The gradient-enhanced PINN (gPINN) extends the standard PINN formulation by adding the gradient of the PDE residual with respect to spatial coordinates as additional loss terms. For the diffusion-reaction inverse problem:

**Standard PINN residual:**
- f = λ(∂²u/∂x²) - k·u - sin(2πx)

**gPINN adds the gradient of the residual:**
- ∂f/∂x = λ(∂³u/∂x³) - k(∂u/∂x) - u(∂k/∂x) - 2π·cos(2πx)

The neural network outputs two components: the solution u and the unknown parameter k. By enforcing both the PDE residual and its spatial derivative to be zero, gPINN achieves better convergence and more accurate inference of the unknown reaction rate function k(x).

The method uses a hard constraint approach for boundary conditions by transforming the network output to automatically satisfy u(0) = u(1) = 0. The network is trained with 8 observation points of u and 8 PDE residual points.

## Implementation

```python
# Define the true unknown reaction rate function
def k(x):
    return 0.1 + np.exp(-0.5 * (x - 0.5) ** 2 / 0.15 ** 2)

# Diffusion coefficient
l = 0.01  # λ in the PDE

# Generate training data from observations
def gen_traindata(num):
    xvals = np.linspace(0, 1, num)  # Sample points
    yvals = sol(xvals)  # Reference solution values
    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))

# Hard constraint for boundary conditions u(0) = u(1) = 0
def output_transform(x, y):
    # y[:, 0:1] is u, y[:, 1:2] is k
    # Transform u to satisfy BCs: u_transformed = x(1-x)*u_network
    return tf.concat((x * (1 - x) * y[:, 0:1], y[:, 1:2]), axis=1)

# Define the geometry
geom = dde.geometry.Interval(0, 1)

# Observation data: 8 measurements of u
ob_x, ob_u = gen_traindata(8)
observe_u = dde.PointSetBC(ob_x, ob_u, component=0)

# Boundary conditions (enforced through output_transform)
bc = dde.DirichletBC(geom, sol, lambda _, on_boundary: on_boundary, component=0)
```

```python
# gPINN: PDE residual with gradient enhancement
def pde(x, y):
    u = y[:, 0:1]  # Solution component
    k = y[:, 1:2]  # Unknown reaction rate component

    # Compute derivatives of u
    du_x = dde.grad.jacobian(y, x, i=0)      # ∂u/∂x
    du_xx = dde.grad.hessian(y, x, component=0)  # ∂²u/∂x²
    du_xxx = dde.grad.jacobian(du_xx, x)     # ∂³u/∂x³

    # Compute derivative of k
    dk_x = dde.grad.jacobian(y, x, i=1)      # ∂k/∂x

    # Return both PDE residual and its gradient
    return [
        # Original PDE residual: λ·∂²u/∂x² - k·u - sin(2πx) = 0
        l * du_xx - k * u - tf.sin(2 * np.pi * x),

        # Gradient of residual: λ·∂³u/∂x³ - k·∂u/∂x - u·∂k/∂x - 2π·cos(2πx) = 0
        l * du_xxx - k * du_x - u * dk_x - 2 * np.pi * tf.cos(2 * np.pi * x),
    ]
```

```python
# Create PDE data with sparse training points
data = dde.data.PDE(
    geom,
    pde,
    bcs=[bc, observe_u],
    num_domain=8,         # 8 residual points in domain
    num_boundary=2,       # 2 boundary points
    train_distribution="uniform",
    num_test=1000,
)

# Neural network architecture: 2-output network (u and k)
# PFNN (Parallel Fully-connected Neural Network) with 3 parallel branches
net = dde.maps.PFNN([1, [20, 20], [20, 20], [20, 20], 2], "tanh", "Glorot uniform")

# Create and compile model
gPINNmodel = dde.Model(data, net)

# Loss weights: [PDE_residual, gradient_residual, boundary, observation]
# The gradient term has weight 0.01 (smaller than PDE residual weight of 1)
gPINNmodel.compile("adam", lr=0.0001, metrics=[], loss_weights=[1, 0.01, 1, 1])

# Train the model
losshistory, train_state = gPINNmodel.train(epochs=200000, callbacks=[])
```

## Critical Parameters

- **Diffusion coefficient**: λ = 0.01
- **Number of observation points**: 8 (sparse measurements of u)
- **Number of PDE residual points**: 8 in the domain
- **Loss weights**: [1, 0.01, 1, 1] for [PDE residual, gradient residual, boundary, observation]
  - The gradient loss weight (0.01) is critical for balancing the contribution of gradient information
- **Network architecture**: PFNN with 3 parallel branches of [20, 20] neurons each, outputting 2 components (u and k)
- **Activation function**: tanh
- **Optimizer**: Adam with learning rate 0.0001
- **Training epochs**: 200,000
- **Boundary condition enforcement**: Hard constraint via output transformation u_transformed = x(1-x)·u_network
- **Domain**: x ∈ [0, 1] with Dirichlet BCs u(0) = u(1) = 0
