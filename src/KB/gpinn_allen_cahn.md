# Gradient-Enhanced Physics-Informed Neural Networks (gPINN) with Residual-Based Adaptive Refinement for Allen-Cahn Equation

**Keywords**: [PDE, parabolic, nonlinear, forward-problem, allen-cahn, 1D, dirichlet, PINN, MLP, self-adaptive, adaptive-sampling, adaptive-weights, strong-form, l2-regularization, adam, lbfgs, mse, deepxde]

**Problem:** Standard Physics-Informed Neural Networks (PINNs) often have limited accuracy even with many training points, particularly for PDEs with solutions containing steep gradients or sharp features. The Allen-Cahn equation with small diffusion coefficient (D = 0.001) produces solutions with multiple very steep regions that are challenging for standard PINNs to capture accurately. This method addresses the need for improved accuracy with fewer training points by leveraging additional gradient information of the PDE residual.

**Issues addressed:**
- Limited accuracy of standard PINNs despite using many training points
- Difficulty capturing solutions with steep gradients and sharp transitions
- Slow convergence requiring excessive residual points
- Non-uniform error distribution with large errors concentrated in regions of steep gradients
- Inefficient residual point distribution for problems with localized features

## Key Method

The gradient-enhanced PINN (gPINN) improves upon standard PINNs by enforcing not only that the PDE residual f equals zero, but also that the gradient of the PDE residual ∇f equals zero. This is based on the principle that if f(x) = 0 everywhere, then ∂f/∂xi = 0 must also hold for all spatial and temporal dimensions.

For the Allen-Cahn equation:
```
∂u/∂t = D ∂²u/∂x² + 5(u - u³)
```

The standard PINN minimizes:
```
L_f = |∂u/∂t - D ∂²u/∂x² - 5(u - u³)|²
```

gPINN adds gradient loss terms by differentiating the PDE residual with respect to both x and t:
```
L_gx = |∂/∂x[∂u/∂t - D ∂²u/∂x² - 5(u - u³)]|² = |∂²u/∂t∂x - D ∂³u/∂x³ - 5(3u²∂u/∂x - ∂u/∂x)|²
L_gt = |∂/∂t[∂u/∂t - D ∂²u/∂x² - 5(u - u³)]|² = |∂²u/∂t² - D ∂³u/∂x²∂t - 5(3u²∂u/∂t - ∂u/∂t)|²
```

Total loss: L = L_f + w_gx * L_gx + w_gt * L_gt

The method is further enhanced with Residual-based Adaptive Refinement (RAR):
1. Train network with initial uniform residual points
2. Evaluate PDE residual at many random candidate points
3. Identify top-k points with largest residual error
4. Add these points to training set as "anchor points"
5. Retrain and repeat until convergence or error threshold met

This adaptive sampling automatically concentrates points in regions of steep gradients where the solution is hardest to learn.

## Implementation

```python
import numpy as np
import deepxde as dde
from deepxde.backend import tf

# Hard boundary condition enforcement via output transformation
# Ensures u(-1,t) = u(1,t) = -1 and u(x,0) = x²cos(πx) automatically
def output_transform(x, y):
    """
    Apply hard constraints for initial and boundary conditions.
    x[:, 0:1]: spatial coordinate x
    x[:, 1:2]: temporal coordinate t
    y: raw network output

    Returns: u(x,t) = t*(1+x)*(1-x)*y + x²cos(πx)
    - At t=0: u = x²cos(πx) (initial condition)
    - At x=±1: boundary term vanishes, satisfies u(-1,t) = u(1,t) = -1
    """
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]
    return t_in * (1 + x_in) * (1 - x_in) * y + tf.square(x_in) * tf.cos(np.pi * x_in)


# gPINN formulation: PDE residual + gradient residuals
def gPINNpde(x, y):
    """
    Compute PDE residual and its gradients for gPINN.

    Allen-Cahn equation: ∂u/∂t = D*∂²u/∂x² + 5(u³ - u)
    where D = 0.001

    Returns three residuals:
    1. Main PDE: du_t - 0.001*du_xx + 5*(u³ - u)
    2. Gradient w.r.t. x: ∂(PDE)/∂x = du_tx - 0.001*du_xxx + 5*(3u²*du_x - du_x)
    3. Gradient w.r.t. t: ∂(PDE)/∂t = du_tt - 0.001*du_xxt + 5*(3u²*du_t - du_t)
    """
    u = y

    # First derivatives
    du_x = dde.grad.jacobian(y, x, j=0)    # ∂u/∂x
    du_t = dde.grad.jacobian(y, x, j=1)    # ∂u/∂t

    # Second derivatives
    du_xx = dde.grad.hessian(y, x, i=0, j=0)    # ∂²u/∂x²
    du_tx = dde.grad.hessian(y, x, i=0, j=1)    # ∂²u/∂x∂t
    du_tt = dde.grad.hessian(y, x, i=1, j=1)    # ∂²u/∂t²

    # Third derivatives for gradient terms
    du_xxx = dde.grad.jacobian(du_xx, x, j=0)   # ∂³u/∂x³
    du_xxt = dde.grad.jacobian(du_xx, x, j=1)   # ∂³u/∂x²∂t

    # Return: [main PDE residual, gradient residual w.r.t. x, gradient residual w.r.t. t]
    return [
        du_t - 0.001 * du_xx + 5 * (u ** 3 - u),                    # Main PDE
        du_tx - 0.001 * du_xxx + 5 * (3 * u ** 2 * du_x - du_x),   # ∂(PDE)/∂x
        du_tt - 0.001 * du_xxt + 5 * (3 * u ** 2 * du_t - du_t),   # ∂(PDE)/∂t
    ]


# Domain definition
geom = dde.geometry.Interval(-1, 1)              # Spatial domain: x ∈ [-1, 1]
timedomain = dde.geometry.TimeDomain(0, 1)       # Temporal domain: t ∈ [0, 1]
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Create PDE data with initial residual points
data = dde.data.TimePDE(
    geomtime,
    gPINNpde,        # PDE with gradient terms
    [],              # No explicit boundary conditions (handled by output_transform)
    num_domain=500   # Initial number of residual points
)

# Neural network: 5 hidden layers with 64 neurons each
net = dde.maps.FNN(
    [2] + [64] * 4 + [1],    # Input: (x, t), Hidden: 64×4, Output: u
    "tanh",                   # Activation function
    "Glorot normal"           # Weight initialization
)
net.apply_output_transform(output_transform)

# Create model
gPINNRARmodel = dde.Model(data, net)

# Initial training with Adam optimizer
# loss_weights: [1.0 for main PDE, 0.0001 for ∂f/∂x, 0.0001 for ∂f/∂t]
gPINNRARmodel.compile("adam", lr=1.0e-3, loss_weights=[1, 0.0001, 0.0001])
losshistory, train_state = gPINNRARmodel.train(epochs=20000)

# Fine-tuning with L-BFGS-B optimizer
gPINNRARmodel.compile("L-BFGS-B", loss_weights=[1, 0.0001, 0.0001])
losshistory, train_state = gPINNRARmodel.train()

# Residual-based Adaptive Refinement (RAR) loop
for i in range(100):  # Up to 100 refinement iterations
    # Sample many candidate points
    X = geomtime.random_points(100000)

    # Evaluate PDE residual at all candidate points
    # Returns residuals for [main PDE, ∂f/∂x, ∂f/∂t]
    err_eq = np.abs(gPINNRARmodel.predict(X, operator=gPINNpde))[0]

    # Monitor mean residual
    err = np.mean(err_eq)
    print("Mean residual: %.3e" % (err))

    # Identify top 30 points with largest residual error
    x_ids = torch.topk(torch.tensor(err_eq), 30, dim=0)[1].numpy()

    # Add high-error points as anchor points to training set
    for elem in x_ids:
        print("Adding new point:", X[elem], "\n")
        data.add_anchors(X[elem])

    # Retrain with augmented training set
    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    gPINNRARmodel.compile("adam", lr=1e-3, loss_weights=[1, 0.0001, 0.0001])
    losshistory, train_state = gPINNRARmodel.train(
        epochs=10000,
        disregard_previous_best=True,
        callbacks=[early_stopping]
    )

    # Fine-tune with L-BFGS-B
    gPINNRARmodel.compile("L-BFGS-B", loss_weights=[1, 0.0001, 0.0001])
    losshistory, train_state = gPINNRARmodel.train()

    # Evaluate accuracy on test data
    X_test, y_true = gen_testdata()
    y_pred = gPINNRARmodel.predict(X_test)
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
```

## Critical Parameters

1. **Loss weights for gradient terms**: [1, 0.0001, 0.0001]
   - Main PDE residual weight: 1.0
   - Gradient residual weights: 0.0001 for both ∂f/∂x and ∂f/∂t
   - Critical: Must be tuned for optimal performance; too large can destabilize training

2. **Network architecture**: 5 hidden layers × 64 neurons
   - Deeper and wider than standard PINN examples due to complexity of steep gradients
   - Activation: tanh (smooth, supports higher-order derivatives)

3. **Initial residual points**: 500
   - Starting uniform distribution before adaptive refinement

4. **RAR parameters**:
   - Candidate points per iteration: 100,000 (for residual evaluation)
   - Points added per iteration: 30 (top errors)
   - Maximum refinement iterations: 100
   - Early stopping: min_delta=1e-4, patience=2000

5. **Two-stage optimization**:
   - Stage 1: Adam with lr=1e-3 for 20,000 epochs (initial) or 10,000 epochs (RAR iterations)
   - Stage 2: L-BFGS-B for fine-tuning (no learning rate needed)

6. **Output transform**: Hard constraint enforcement
   - Automatically satisfies initial and boundary conditions
   - Eliminates need for boundary loss terms

7. **PDE parameters**:
   - Diffusion coefficient D = 0.001 (small, creates steep gradients)
   - Reaction term coefficient: 5
   - Domain: x ∈ [-1, 1], t ∈ [0, 1]
