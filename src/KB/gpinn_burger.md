# Gradient-Enhanced Physics-Informed Neural Networks (gPINN) for Burgers' Equation

**Keywords**: ["PDE", "parabolic", "nonlinear", "forward-problem", "burgers", "1D", "dirichlet", "PINN", "MLP", "self-adaptive", "strong-form", "adam", "lbfgs", "mse", "deepxde"]

**Problem:** Gradient-enhanced physics-informed neural networks (gPINNs) improve the accuracy of standard PINNs by leveraging gradient information of the PDE residual. The method is demonstrated on the 1D Burgers' equation, a nonlinear parabolic PDE with steep gradients. Standard PINNs often have limited accuracy even with many training points, particularly for problems with solutions containing steep gradients or discontinuities. gPINN addresses this by enforcing not only the PDE residual to be zero but also the derivatives of the PDE residual with respect to the spatial and temporal variables.

**Issues addressed:** Solutions with steep gradients, convergence challenges in standard PINNs, high residual errors in regions with sharp solution features, requirement for fewer training points to achieve comparable accuracy.

## Key Method

The core innovation of gPINN is adding gradient-based loss terms to the standard PINN formulation. For a PDE residual f(x), gPINN enforces both f(x) = 0 and ∇f(x) = 0.

For the 1D Burgers' equation:
- PDE: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
- The gradient loss terms include derivatives of the PDE residual with respect to both x and t
- Loss function: L = Lf + wgx·Lgx + wgt·Lgt

Where:
- Lf: Standard PDE residual loss
- Lgx: Gradient loss with respect to spatial variable (∂f/∂x)²
- Lgt: Gradient loss with respect to temporal variable (∂f/∂t)²
- wgx, wgt: Weight parameters for gradient losses

The method is combined with Residual-based Adaptive Refinement (RAR), which adaptively adds training points where the PDE residual is largest during training. This combination (gPINN + RAR) is particularly effective for PDEs with steep gradients like Burgers' equation.

## Implementation

```python
# Output transform to enforce boundary and initial conditions
def output_transform(x, y):
    """
    Hard constraint to satisfy:
    - Boundary conditions: u(-1,t) = u(1,t) = 0
    - Initial condition: u(x,0) = -sin(πx)
    """
    x_in = x[:, 0:1]  # Spatial coordinate
    t_in = x[:, 1:2]  # Temporal coordinate

    # Construct output that automatically satisfies BC and IC
    # (1-x)(1+x) ensures u=0 at x=±1
    # (1-exp(-t)) ensures smooth transition from initial condition
    return (1 - x_in) * (1 + x_in) * (1 - tf.exp(-t_in)) * y - tf.sin(np.pi * x_in)


# PDE definition for gPINN: returns residual and its gradients
def pde(x, y):
    """
    Define Burgers' equation and its gradient-enhanced terms.
    Returns three residuals:
    1. Standard PDE residual: ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x² = 0
    2. Gradient w.r.t. x: ∂/∂x(PDE residual) = 0
    3. Gradient w.r.t. t: ∂/∂t(PDE residual) = 0
    """
    # First-order derivatives
    dy_x = dde.grad.jacobian(y, x, j=0)   # ∂u/∂x
    dy_t = dde.grad.jacobian(y, x, j=1)   # ∂u/∂t

    # Second-order derivatives
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)  # ∂²u/∂x²

    # Third-order derivatives for gradient enhancement
    dy_tx = dde.grad.hessian(y, x, i=0, j=1)   # ∂²u/∂t∂x
    dy_xxx = dde.grad.jacobian(dy_xx, x, j=0)  # ∂³u/∂x³

    # Fourth-order derivatives for temporal gradient
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)   # ∂²u/∂t²
    dy_xxt = dde.grad.jacobian(dy_xx, x, j=1)  # ∂³u/∂x²∂t

    # Return list of residuals: [PDE, ∂PDE/∂x, ∂PDE/∂t]
    return [
        dy_t + y * dy_x - 0.01 / np.pi * dy_xx,  # Standard Burgers' equation
        dy_tx + (dy_x * dy_x + y * dy_xx) - 0.01 / np.pi * dy_xxx,  # ∂/∂x(PDE)
        dy_tt + dy_t * dy_x + y * dy_tx - 0.01 / np.pi * dy_xxt,    # ∂/∂t(PDE)
    ]


# Domain definition
geom = dde.geometry.Interval(-1, 1)  # Spatial domain: [-1, 1]
timedomain = dde.geometry.TimeDomain(0, 1)  # Temporal domain: [0, 1]
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Create PDE data with no explicit boundary/initial conditions (handled by output_transform)
data = dde.data.TimePDE(
    geomtime, pde, [],
    num_domain=1500,    # Number of residual points
    num_boundary=0,     # BC handled by output_transform
    num_initial=0       # IC handled by output_transform
)

# Neural network: 3 hidden layers, 32 neurons each
net = dde.maps.FNN([2] + [32] * 3 + [1], "tanh", "Glorot normal")
net.apply_output_transform(output_transform)

# Create model
gPINNRARmodel = dde.Model(data, net)

# Training phase 1: Adam optimizer with gradient-enhanced loss
# Loss weights: [PDE residual, ∂PDE/∂x, ∂PDE/∂t]
gPINNRARmodel.compile("adam", lr=1.0e-3, loss_weights=[1, 0.0001, 0.0001])
losshistory, train_state = gPINNRARmodel.train(epochs=20000)

# Training phase 2: L-BFGS-B for fine-tuning
gPINNRARmodel.compile("L-BFGS-B", loss_weights=[1, 0.0001, 0.0001])
losshistory, train_state = gPINNRARmodel.train()


# Residual-based Adaptive Refinement (RAR) loop
for i in range(40):
    # Sample random points and compute PDE residual
    X = geomtime.random_points(100000)
    err_eq = np.abs(gPINNRARmodel.predict(X, operator=pde))[0]

    err = np.mean(err_eq)
    print("Mean residual: %.3e" % (err))

    # Select top 10 points with largest residual error
    err_eq = torch.tensor(err_eq)
    x_ids = torch.topk(err_eq, 10, dim=0)[1].numpy()

    # Add high-error points to training set
    for elem in x_ids:
        print("Adding new point:", X[elem], "\n")
        data.add_anchors(X[elem])

    # Retrain with expanded training set
    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    gPINNRARmodel.compile("adam", lr=1e-3, loss_weights=[1, 0.0001, 0.0001])
    losshistory, train_state = gPINNRARmodel.train(
        epochs=10000, disregard_previous_best=True, callbacks=[early_stopping]
    )

    # Fine-tune with L-BFGS-B
    gPINNRARmodel.compile("L-BFGS-B", loss_weights=[1, 0.0001, 0.0001])
    losshistory, train_state = gPINNRARmodel.train()

    # Evaluate accuracy
    X, y_true = gen_testdata()
    y_pred = gPINNRARmodel.predict(X)
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
```

## Critical Parameters

- **Viscosity coefficient**: ν = 0.01/π (defines diffusion strength in Burgers' equation)
- **Network architecture**: 3 hidden layers with 32 neurons each
- **Activation function**: tanh
- **Initial training points**: 1500 uniformly distributed residual points
- **Loss weights for gradient terms**: [1, 0.0001, 0.0001] for [PDE, ∂PDE/∂x, ∂PDE/∂t]
  - The gradient loss weights (0.0001) are critical and problem-dependent
  - Too large weights can destabilize training; too small provides no benefit
- **Optimizer**: Two-stage approach
  - Adam with learning rate 1e-3 for 20,000 epochs (initial training)
  - L-BFGS-B for fine-tuning (no learning rate needed)
- **RAR parameters**:
  - Number of refinement iterations: 40
  - Points added per iteration: 10
  - Candidate pool for point selection: 100,000 random points
  - Early stopping: min_delta=1e-4, patience=2000
- **Domain**: x ∈ [-1, 1], t ∈ [0, 1]
- **Boundary conditions**: Dirichlet (u = 0 at x = ±1)
- **Initial condition**: u(x,0) = -sin(πx)
