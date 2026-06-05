# Physics-Informed Neural Networks with Hard Constraints for Stokes Flow Topology Optimization

**Keywords**: [PDE, elliptic, nonlinear, inverse-problem, optimization, stokes, 2D, regular, dirichlet, PINN, MLP, strong-form, augmented-lagrangian, l2-regularization, adam, lbfgs, mse, deepxde]

**Problem:** Topology optimization for fluids in Stokes flow to minimize dissipated power while satisfying PDE constraints (Stokes equations) and inequality constraints (fluid volume fraction). The challenge is that the objective function and PDE constraints compete with each other during optimization - unlike standard PINNs where data and PDE losses are consistent, here they are not generally compatible, making it difficult to both satisfy PDEs exactly and achieve good design objectives simultaneously.

**Issues addressed:** Ill-conditioning when penalty coefficients are too large, convergence difficulties in constrained optimization, inability to satisfy hard constraints (PDEs and inequalities) with standard soft constraint approaches.

## Key Method

The method uses **hard-constrained Physics-Informed Neural Networks (hPINN)** with the **augmented Lagrangian method** for topology optimization. Key innovations:

1. **Hard constraint boundary conditions**: Dirichlet BCs are embedded directly into the neural network architecture through output transformations, ensuring exact satisfaction without additional loss terms.

2. **Augmented Lagrangian method**: Unlike soft constraints (which use fixed penalty coefficients) or standard penalty methods (which can become ill-conditioned), the augmented Lagrangian adds Lagrange multiplier terms that mimic the gradient contribution of penalty terms from previous iterations. This allows convergence without requiring penalty coefficients to go to infinity.

3. **Separate networks for physics and design**: Independent neural networks approximate the velocity field (u, v), pressure (p), and design parameter ρ (density function indicating solid vs fluid regions).

4. **Volume constraint handling**: Inequality constraint on fluid volume fraction is enforced through the augmented Lagrangian framework with adaptive multipliers.

The Stokes flow problem is formulated as:
- Governing PDEs: -ν∆u + ∇p = αu (generalized Stokes with Darcy term), ∇·u = 0
- Design parameter: ρ ∈ [0,1] where ρ=0 is solid, ρ=1 is fluid
- Inverted permeability: α(ρ) = ᾱ + (α - ᾱ)ρ(1+q)/(ρ+q) with smooth interpolation
- Objective: Minimize dissipated power J = ∫(½∇u:∇u + ½αu²)dxdy
- Constraint: ∫ρ dxdy ≤ γ (volume fraction limit)

## Implementation

```python
# Problem setup: Stokes flow with Darcy forcing term
GAMMA = 0.9  # Volume fraction constraint

def alpha(rho):
    """Inverted permeability interpolation between solid and fluid"""
    alpha_max, alpha_min = 2.5 * 10 ** 4, 0  # Solid vs fluid permeability
    q = 0.1  # Controls sharpness of transition
    return alpha_max + (alpha_min - alpha_max) * rho * (1 + q) / (rho + q)

def pde(inputs, outputs):
    """Stokes equations with Darcy term as PDE residuals"""
    # Extract derivatives using automatic differentiation
    du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
    dv_y = dde.grad.jacobian(outputs, inputs, i=1, j=1)
    du_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(outputs, inputs, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(outputs, inputs, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(outputs, inputs, component=1, i=1, j=1)
    dp_x = dde.grad.jacobian(outputs, inputs, i=2, j=0)
    dp_y = dde.grad.jacobian(outputs, inputs, i=2, j=1)

    # Darcy forcing term: f = α(ρ)·u
    f = alpha(outputs[:, 3:]) * outputs[:, :2]
    fx, fy = f[:, :1], f[:, 1:]

    # Stokes momentum equations: -ν∆u + ∇p - f = 0 (scaled)
    loss1 = (-(du_xx + du_yy) + dp_x - fx) * 0.01  # x-momentum
    loss2 = (-(dv_xx + dv_yy) + dp_y - fy) * 0.01  # y-momentum
    loss3 = (du_x + dv_y) * 1e2  # Continuity equation

    # Return losses for augmented Lagrangian (6 terms: 3 penalty + 3 Lagrangian)
    return loss1, loss2, loss3, loss1, loss2, loss3

def volume(inputs, outputs, X):
    """Extract density ρ for volume constraint"""
    return outputs[:, 3:4]

def loss_volume(_, y):
    """Penalty loss for volume constraint: max(0, mean(ρ) - γ)²"""
    return tf.math.square(tf.math.maximum(0.0, tf.reduce_mean(y) - GAMMA))

def dissipated_power(inputs, outputs, X):
    """Objective function: dissipated power in Stokes flow"""
    du = dde.grad.jacobian(outputs, inputs, i=0)
    dv = dde.grad.jacobian(outputs, inputs, i=1)
    # Power = ½(|∇u|² + |∇v|²) + ½α|u|²
    p1 = tf.math.reduce_sum(
        tf.math.square(du) + tf.math.square(dv), axis=1, keepdims=True
    )
    u2 = tf.math.reduce_sum(tf.math.square(outputs[:, :2]), axis=1, keepdims=True)
    p2 = alpha(outputs[:, 3:]) * u2
    return 0.5 * (p1 + p2)

def loss_power(_, y):
    """Minimize average dissipated power"""
    return tf.reduce_mean(y)
```

```python
def output_transform(inputs, outputs):
    """Embed hard Dirichlet BCs into network architecture"""
    x, y = inputs[:, :1], inputs[:, 1:]
    bc = 16 * x * (1 - x) * y * (1 - y)  # Zero at boundary

    # u: velocity in x-direction with BC u=1 at boundary
    u0 = 1
    u = tf.math.abs(u0 + bc * outputs[:, :1])

    # v: velocity in y-direction with BC v=0 at boundary
    v = bc * outputs[:, 1:2]

    # p: pressure with BC p=0 at right boundary (x=1)
    p = (1 - x) * outputs[:, 2:3]

    # ρ: density constrained to [0,1], solid at center, fluid at boundary
    center = tf.math.square(x - 0.5) + tf.math.square(y - 0.5)
    rho = center * (
        bc * outputs[:, 3:] + (1 - bc) * (1 + 1e-6 / 0.25) / (center + 1e-6)
    )
    rho = tf.math.maximum(0.0, tf.math.minimum(1.0, rho))

    return tf.concat((u, v, p, rho), axis=1)
```

```python
def augmented_Lagrangian(model, geom, mu_PDE, mu_V, beta):
    """Augmented Lagrangian method for hard constraint enforcement"""
    x = model.data.train_x[np.sum(model.data.num_bcs) :]  # Interior points
    x_inside = model.data.train_x[: model.data.num_bcs[0]]

    # Initialize Lagrange multipliers
    lambla1 = np.zeros((len(x), 1))
    lambla2 = np.zeros((len(x), 1))
    lambla3 = np.zeros((len(x), 1))
    lambdaV = 0
    mus = [[mu_PDE, mu_V, lambdaV]]

    for i in range(1, 10):  # Outer iterations
        # Update multipliers based on residuals
        residual1, residual2, residual3, _, _, _ = model.predict(x, operator=pde)
        lambla1 += 2 / 3 * mu_PDE * residual1
        lambla2 += 2 / 3 * mu_PDE * residual2
        lambla3 += 2 / 3 * mu_PDE * residual3

        # Update volume constraint multiplier
        dV = np.mean(model.predict(x_inside)[:, 3:4]) - GAMMA
        lambdaV = max(lambdaV + 2 * mu_V * dV, 0)

        # Increase penalty coefficients
        mu_PDE *= beta
        mu_V *= beta
        mus.append([mu_PDE, mu_V, lambdaV])
        print(f"Iteration {i}: mu = {mu_PDE}, {mu_V}, lambdaV = {lambdaV}\n")

        # Define loss functions with current multipliers
        def loss_PDE1(_, y):
            return tf.reduce_mean(lambla1 * y)

        def loss_PDE2(_, y):
            return tf.reduce_mean(lambla2 * y)

        def loss_PDE3(_, y):
            return tf.reduce_mean(lambla3 * y)

        def loss_V1(_, y):
            if lambdaV > 0:
                return tf.math.square(tf.reduce_mean(y) - GAMMA)
            return loss_volume(None, y)

        def loss_V2(_, y):
            return tf.reduce_mean(y) - GAMMA

        # Combined loss with penalty and Lagrangian terms
        loss_weights = [mu_PDE / 3] * 3 + [1] * 3 + [mu_V, lambdaV, 1]
        loss = (
            ["MSE"] * 3
            + [loss_PDE1, loss_PDE2, loss_PDE3]
            + [loss_V1, loss_V2, loss_power]
        )

        model.compile("L-BFGS-B", loss=loss, loss_weights=loss_weights)
        losshistory, train_state = model.train(disregard_previous_best=True)

        # Save intermediate results
        save_solution(geom, model, f"solution{i}")
```

```python
def main():
    """Main training loop for topology optimization"""
    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    # Network: 4 parallel fully-connected blocks, each 4 layers × 64 neurons
    net = dde.maps.PFNN([2] + [[64] * 4] * 4 + [4], "tanh", "Glorot normal")
    net.apply_output_transform(output_transform)

    # Define loss components (volume constraint appears twice for augmented Lagrangian)
    losses = [
        dde.OperatorBC(geom, volume, lambda x, _: not geom.on_boundary(x)),
        dde.OperatorBC(geom, volume, lambda x, _: not geom.on_boundary(x)),
        dde.OperatorBC(geom, dissipated_power, lambda x, _: not geom.on_boundary(x)),
    ]

    dx = 0.01
    data = dde.data.PDE(
        geom,
        pde,
        losses,
        num_domain=int(geom.area / dx ** 2),  # ~10,000 points
        num_boundary=int(geom.perimeter / dx),
    )
    model = dde.Model(data, net)

    # Initial training with soft constraints
    mu_PDE, mu_V = 0.1, 1e4
    loss_weights = [mu_PDE / 3] * 3 + [0] * 3 + [mu_V, 0] + [1]
    loss = ["MSE"] * 3 + ["zero"] * 3 + [loss_volume, "zero", loss_power]

    # Two-stage optimization: Adam then L-BFGS
    model.compile("adam", lr=0.0001, loss=loss, loss_weights=loss_weights)
    losshistory, train_state = model.train(epochs=20000)

    model.compile("L-BFGS-B", loss=loss, loss_weights=loss_weights)
    losshistory, train_state = model.train()
    save_solution(geom, model, "solution0")

    # Apply augmented Lagrangian method for hard constraints
    augmented_Lagrangian(model, geom, mu_PDE, mu_V, beta=2)

    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
```

## Critical Parameters

- **GAMMA = 0.9**: Fluid volume fraction constraint (90% of domain must be fluid)
- **alpha_max = 2.5×10⁴, alpha_min = 0**: Inverted permeability range (solid vs fluid)
- **q = 0.1**: Interpolation sharpness parameter between solid/fluid phases
- **Network architecture**: 4 parallel blocks of [64]×4 layers, tanh activation
- **num_domain ≈ 10,000**: Number of interior collocation points (dx=0.01)
- **mu_PDE = 0.1, mu_V = 10⁴**: Initial penalty coefficients for PDE and volume constraints
- **beta = 2**: Penalty coefficient growth factor in augmented Lagrangian
- **Outer iterations**: 9 augmented Lagrangian iterations
- **Optimizer**: Adam (lr=1e-4, 20k epochs) followed by L-BFGS-B
- **PDE loss scaling**: 0.01 for momentum equations, 100 for continuity equation
- **DeepXDE version**: 0.9.1 (specific version required for compatibility)
