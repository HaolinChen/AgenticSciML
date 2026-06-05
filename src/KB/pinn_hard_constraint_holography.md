# Physics-Informed Neural Networks with Hard Constraints for Inverse Design (hPINN)

**Keywords**: ["PDE", "elliptic", "inverse-problem", "helmholtz", "2D", "dirichlet", "periodic", "PINN", "MLP", "strong-form", "penalty-method", "augmented-lagrangian", "adam", "lbfgs", "mse", "deepxde"]

**Problem:** Topology optimization and inverse design problems are constrained by partial differential equations (PDEs) and inequality constraints. Traditional PINNs use soft constraints (loss functions) for PDEs, which leads to a trade-off: the PDE-based loss and the design objective function compete during optimization. Simply optimizing the sum of both losses results in solutions that do not satisfy the PDEs accurately. The challenge is to enforce PDE constraints as hard constraints while optimizing the design objective.

**Issues addressed:** Traditional PINNs with soft constraints fail for inverse design because: (1) PDE constraints and design objectives are not consistent and cannot be minimized to zero simultaneously, (2) solutions may not satisfy PDEs when penalty coefficients are too small, (3) optimization becomes ill-conditioned when penalty coefficients are too large. hPINN addresses these by using penalty methods and augmented Lagrangian methods to enforce hard constraints, ensuring PDE satisfaction while achieving good design objectives.

## Key Method

hPINN (Physics-Informed Neural Networks with Hard Constraints) solves PDE-constrained inverse design problems by:

1. **Hard-constraint boundary conditions**: Dirichlet and periodic BCs are imposed exactly by modifying the neural network architecture rather than using loss functions.

2. **Penalty Method**: Converts constrained optimization to a sequence of unconstrained problems with increasing penalty coefficients μ^k = β^k μ^0. Each iteration uses the previous solution as initialization to mitigate ill-conditioning.

3. **Augmented Lagrangian Method**: Adds Lagrange multiplier terms to the loss function:
   - Loss = J + μ_F L_F + μ_h L_h + λ_F · F[u] + λ_h · h
   - Multipliers are updated: λ^k = λ^(k-1) + 2μ^(k-1) · residual
   - Provides better convergence than penalty method by avoiding excessive penalty coefficients

4. **Network Architecture Modifications**:
   - Dirichlet BC: u(x) = g(x) + ℓ(x)N(x) where ℓ(x)=0 on boundary
   - Periodic BC: Replace input x_j with Fourier basis {cos(2πx_j/P), sin(2πx_j/P)}

## Implementation

### Hard-Constraint Dirichlet Boundary Conditions

```python
def output_transform(inputs, outputs):
    """Enforce zero Dirichlet BC at top and bottom boundaries."""
    x, y = inputs[:, :1], inputs[:, 1:]

    # Permittivity constrained to [1, 12]
    eps = 1 + 11 * tf.math.sigmoid(outputs[:, -1:])

    # Zero Dirichlet BC: ℓ(y) = (1 - e^(a-y))(1 - e^(y-b))
    # ℓ(y) = 0 at y=a and y=b, ℓ(y) > 0 for a < y < b
    a, b = BOX[0][1] - DPML, BOX[1][1] + DPML
    E = (1 - tf.math.exp(a - y)) * (1 - tf.math.exp(y - b)) * outputs[:, :2]

    return tf.concat((E, eps), axis=1)
```

### Hard-Constraint Periodic Boundary Conditions

```python
def feature_transform(inputs):
    """Enforce periodic BC in x-direction using Fourier basis."""
    # Period in x-direction
    P = BOX[1][0] - BOX[0][0] + 2 * DPML
    w = 2 * np.pi / P
    x, y = w * inputs[:, :1], inputs[:, 1:]

    # Replace x with Fourier basis functions
    # Using 6 harmonics (12 basis functions)
    return tf.concat(
        (
            tf.math.cos(x), tf.math.sin(x),
            tf.math.cos(2*x), tf.math.sin(2*x),
            tf.math.cos(3*x), tf.math.sin(3*x),
            tf.math.cos(4*x), tf.math.sin(4*x),
            tf.math.cos(5*x), tf.math.sin(5*x),
            tf.math.cos(6*x), tf.math.sin(6*x),
            y,
            tf.math.cos(OMEGA * y),  # Extra features for planewave pattern
            tf.math.sin(OMEGA * y),
        ),
        axis=1,
    )
```

### PDE Residual with Perfectly Matched Layer (PML)

```python
def pde(inputs, outputs, X, ReE, ImE, eps):
    """Helmholtz equation with PML for electromagnetic holography.

    PDE: ∇²E + εω²E = -iωJ
    With PML coordinate stretching in x and y directions.
    """
    # Compute PML coefficients A1, B1, A2, B2, A3, B3, A4, B4
    # These represent 1/(1 + iσ_x/ω)² and derivatives
    A1, B1, A2, B2, A3, B3, A4, B4 = PML(X)

    # Compute derivatives using automatic differentiation
    dReE_x = dde.grad.jacobian(outputs, inputs, i=ReE, j=0)
    dReE_y = dde.grad.jacobian(outputs, inputs, i=ReE, j=1)
    dReE_xx = dde.grad.hessian(outputs, inputs, component=ReE, i=0, j=0)
    dReE_yy = dde.grad.hessian(outputs, inputs, component=ReE, i=1, j=1)
    dImE_x = dde.grad.jacobian(outputs, inputs, i=ImE, j=0)
    dImE_y = dde.grad.jacobian(outputs, inputs, i=ImE, j=1)
    dImE_xx = dde.grad.hessian(outputs, inputs, component=ImE, i=0, j=0)
    dImE_yy = dde.grad.hessian(outputs, inputs, component=ImE, i=1, j=1)

    ReE = outputs[:, ReE : ReE + 1]
    ImE = outputs[:, ImE : ImE + 1]

    # Real part of Helmholtz equation with PML
    loss_Re = (
        (A1 * dReE_xx + A2 * dReE_x + A3 * dReE_yy + A4 * dReE_y) / OMEGA
        - (B1 * dImE_xx + B2 * dImE_x + B3 * dImE_yy + B4 * dImE_y) / OMEGA
        + eps * OMEGA * ReE
    )

    # Imaginary part of Helmholtz equation with PML
    loss_Im = (
        (A1 * dImE_xx + A2 * dImE_x + A3 * dImE_yy + A4 * dImE_y) / OMEGA
        + (B1 * dReE_xx + B2 * dReE_x + B3 * dReE_yy + B4 * dReE_y) / OMEGA
        + eps * OMEGA * ImE
        + J(X)  # Source term
    )

    # Return 4 outputs for augmented Lagrangian (duplicated for multipliers)
    return loss_Re, loss_Im, loss_Re, loss_Im
```

### Augmented Lagrangian Training Loop

```python
def augmented_Lagrangian(model, geom, geom2, mu, beta):
    """Augmented Lagrangian method for enforcing PDE as hard constraint."""
    # Initialize multipliers
    x = model.data.train_x[np.sum(model.data.num_bcs):]
    lambda_Re, lambda_Im = np.zeros((len(x), 1)), np.zeros((len(x), 1))

    for i in range(1, 10):
        # Update Lagrange multipliers based on PDE residuals
        residual_Re, residual_Im, _, _ = model.predict(x, operator=pde_domain)
        lambda_Re += mu * residual_Re
        lambda_Im += mu * residual_Im

        # Increase penalty coefficient
        mu *= beta
        print(f"Iteration {i}: mu = {mu}")

        # Define Lagrangian loss terms
        def loss_Lagrangian_Re(_, y):
            return tf.reduce_mean(lambda_Re * y)

        def loss_Lagrangian_Im(_, y):
            return tf.reduce_mean(lambda_Im * y)

        # Total loss: objective + penalty + Lagrangian terms
        # loss_weights[0:2]: penalty on PDE residuals (0.5*mu each)
        # loss_weights[3:4]: Lagrangian multiplier terms
        # loss_weights[4]: target objective (intensity matching)
        loss_weights = [0.5 * mu] * 2 + [1, 1] + [1]
        loss = ["MSE", "MSE", loss_Lagrangian_Re, loss_Lagrangian_Im, "MSE"]

        # Compile and train with L-BFGS optimizer
        model.compile("L-BFGS-B", loss=loss, loss_weights=loss_weights)
        losshistory, train_state = model.train(disregard_previous_best=True)

        # Save intermediate results
        save_epsilon(geom2, model, f"epsilon{i}.dat")
```

### Main Training Procedure

```python
# Network architecture: PFNN with 4 blocks of 3 hidden layers, 48 neurons each
net = dde.maps.PFNN([2] + [[48] * 3] * 4 + [3], "tanh", "Glorot normal")
net.apply_feature_transform(feature_transform)  # Periodic BC
net.apply_output_transform(output_transform)    # Dirichlet BC

# Define target objective: match intensity pattern in design region
losses = [
    dde.OperatorBC(geom, target_bc, lambda x, _: geom3_small.inside(x)),
]

# Create PDE data with residual points
dx = 0.05
data = dde.data.PDE(
    geom,
    pde_domain,  # PDE residual
    losses,      # Target objective
    num_domain=int(geom.area / dx ** 2),
    num_boundary=int(geom.perimeter / dx),
)
model = dde.Model(data, net)

# Initial training with Adam
mu = 2
loss_weights = [0.5 * mu] * 2 + [0, 0] + [1]  # For augmented Lagrangian
model.compile("adam", lr=0.001, loss_weights=loss_weights)
model.train(epochs=20000)

# Switch to L-BFGS for refinement
model.compile("L-BFGS-B", loss_weights=loss_weights)
model.train()

# Apply augmented Lagrangian method for hard constraint
augmented_Lagrangian(model, geom, geom2, mu, beta=2)
```

## Critical Parameters

- **Network architecture**: PFNN with 4 blocks × 3 layers × 48 neurons, tanh activation
- **Fourier basis**: 6-12 harmonics for periodic BC (more harmonics improve accuracy)
- **Initial penalty coefficient** μ₀ = 2
- **Penalty increase factor** β = 2
- **Augmented Lagrangian iterations**: 9 outer iterations
- **Optimizers**: Adam (lr=0.001, 20000 epochs) → L-BFGS-B for refinement
- **Residual points**: ~17000 points (spacing ~0.05)
- **Permittivity range**: ε ∈ [1, 12] via sigmoid transformation
- **PML depth**: 1 wavelength, σ₀ = -ln(10⁻²⁰)/(4d³/3)
- **Frequency**: ω = 2π (wavelength = 1)
