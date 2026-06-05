# Self-Adaptive Physics-Informed Neural Networks for Burgers Equation

**Keywords**: [PDE, parabolic, nonlinear, forward-problem, burgers, 1D, dirichlet, PINN, MLP, self-adaptive, adaptive-weights, strong-form, adam, lbfgs, mse, tensorflow]

**Problem:** Solving the viscous Burgers equation, a nonlinear PDE that develops sharp gradients (shock-like behavior) due to the convective term. While baseline PINNs can solve relatively smooth PDEs, they struggle with solutions containing discontinuities or sharp transitions. The multi-part loss function in standard PINNs can suffer from imbalance where gradient descent latches onto some components at the expense of others.

**Issues addressed:**
- Training imbalance in multi-part loss functions
- Difficulty capturing sharp gradients and shock-like features
- Suboptimal focus on difficult regions with steep solution transitions
- Slow convergence in early training stages

## Key Method

Self-Adaptive PINNs (SA-PINNs) apply **point-wise trainable weights** to create a soft attention mechanism where the network automatically learns which training points are most difficult and require more focus. This is fundamentally different from global weighting schemes that apply uniform weights to entire loss components.

**Core Mechanism:**
- Each training point gets its own adaptive weight: λᵢ for i = 1,...,N
- Weights are optimized via **gradient ascent** while network weights undergo **gradient descent**
- Creates a saddle-point optimization seeking to minimize losses while maximizing weights
- Points with larger residuals automatically get larger weight increases

**Loss Function:**
L(w, λᵣ, λ₀) = L₀(w, λ₀) + Lᵣ(w, λᵣ) + Lᵦ(w)

where:
- L₀ = (1/2) Σᵢ m(λᵢ₀)|u(xᵢ,0) - u₀(xᵢ)|²  (initial condition, weighted)
- Lᵣ = (1/2) Σᵢ m(λᵢᵣ)|PDE_residual|²  (collocation points, weighted)
- Lᵦ = (1/2) Σᵢ |boundary_condition|²  (boundaries, unweighted)

**Update Rules:**
- w^(k+1) = w^k - η ∇_w L(w^k, λ^k)
- λ^(k+1) = λ^k + ρ ∇_λ L(w^k, λ^k)

The gradient ∇_λ L is proportional to the squared residuals, so larger errors drive larger weight increases, automatically focusing training on stubborn regions.

## Implementation

### Network Architecture

```python
# Network: 2 inputs (x,t) -> 8 hidden layers (20 neurons) -> 1 output (u)
layer_sizes = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

# Weight/bias size tracking for L-BFGS
sizes_w = []
sizes_b = []
for i, width in enumerate(layer_sizes):
    if i != 1:
        sizes_w.append(int(width * layer_sizes[1]))
        sizes_b.append(int(width if i != 0 else layer_sizes[1]))

def neural_net(layer_sizes):
    """
    Construct MLP with tanh activation

    Uses Glorot (Xavier) normal initialization for stable training
    No activation on output layer (linear output for u)
    """
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))

    # Hidden layers with tanh activation
    for width in layer_sizes[1:-1]:
        model.add(layers.Dense(
            width, activation=tf.nn.tanh,
            kernel_initializer="glorot_normal"))

    # Output layer (linear)
    model.add(layers.Dense(
            layer_sizes[-1], activation=None,
            kernel_initializer="glorot_normal"))

    return model

u_model = neural_net(layer_sizes)
```

### Self-Adaptive Loss Function

```python
def loss(x_f_batch, t_f_batch, x0, t0, u0, x_lb, t_lb, x_ub, t_ub,
         col_weights, u_weights):
    """
    Compute weighted loss for Burgers equation

    Parameters:
    -----------
    x_f_batch, t_f_batch: Collocation points for PDE residual
    x0, t0, u0: Initial condition points and values
    x_lb, t_lb: Left boundary points (x=-1)
    x_ub, t_ub: Right boundary points (x=1)
    col_weights: Adaptive weights for collocation points (N_f x 1)
    u_weights: Adaptive weights for initial condition (N_0 x 1)

    Returns:
    --------
    total_loss, mse_0_u, mse_f_u: Total loss and individual components
    """
    # Compute PDE residual at collocation points
    f_u_pred = f_model(x_f_batch, t_f_batch)

    # Initial condition prediction
    u0_pred = u_model(tf.concat([x0, t0], 1))

    # Boundary predictions
    u_lb_pred, _ = u_x_model(x_lb, t_lb)
    u_ub_pred, _ = u_x_model(x_ub, t_ub)

    # Weighted initial condition loss
    # u_weights acts as point-wise attention mask
    mse_0_u = tf.reduce_mean(tf.square(u_weights*(u0 - u0_pred)))

    # Boundary loss (Dirichlet: u = 0 at boundaries)
    # Not weighted in this implementation
    mse_b_u = tf.reduce_mean(tf.square(u_lb_pred - 0)) + \
              tf.reduce_mean(tf.square(u_ub_pred - 0))

    # Weighted PDE residual loss
    # col_weights provides adaptive attention to difficult collocation points
    mse_f_u = tf.reduce_mean(tf.square(col_weights*f_u_pred))

    return mse_0_u + mse_b_u + mse_f_u, mse_0_u, mse_f_u
```

### Burgers Equation PDE Residual

```python
@tf.function
def f_model(x, t):
    """
    Burgers equation residual: u_t + u*u_x - (0.01/π)*u_xx = 0

    This is a convection-diffusion equation with:
    - Convective term u*u_x (nonlinear, creates shocks)
    - Diffusive term (0.01/π)*u_xx (small viscosity, limited smoothing)
    """
    u = u_model(tf.concat([x, t], 1))

    # Compute spatial and temporal derivatives using automatic differentiation
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)

    # Burgers equation with viscosity ν = 0.01/π
    f_u = u_t + u*u_x - (0.01/tf.constant(math.pi))*u_xx

    return f_u

@tf.function
def u_x_model(x, t):
    """
    Compute u and u_x (spatial derivative)
    Used for boundary condition evaluation
    """
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)
    return u, u_x
```

### Gradient Computation for Dual Optimization

```python
@tf.function
def grad(model, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,
         x_lb, t_lb, x_ub, t_ub, col_weights, u_weights):
    """
    Compute gradients for both network weights and adaptive weights

    Returns gradients for:
    - Network parameters (descent direction)
    - Adaptive weights (ascent direction)
    """
    with tf.GradientTape(persistent=True) as tape:
        loss_value, mse_0, mse_f = loss(
            x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,
            x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)

        # Gradient w.r.t. network weights (for descent)
        grads = tape.gradient(loss_value, u_model.trainable_variables)

        # Gradient w.r.t. adaptive weights (for ascent)
        # These gradients are proportional to squared residuals
        grads_col = tape.gradient(loss_value, col_weights)
        grads_u = tape.gradient(loss_value, u_weights)

    return loss_value, mse_0, mse_f, grads, grads_col, grads_u
```

### Training Loop with Adaptive Weighting

```python
def fit(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub,
        col_weights, u_weights, tf_iter, newton_iter):
    """
    Two-stage training with self-adaptive weights

    Stage 1: Adam optimization (adaptive weights active)
    Stage 2: L-BFGS refinement (adaptive weights frozen)
    """
    batch_sz = N_f  # Full batch by default
    n_batches = N_f // batch_sz

    # Separate optimizers for network and adaptive weights
    # beta_1=0.90 for Adam (less momentum than default 0.9)
    tf_optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=.90)
    tf_optimizer_coll = tf.keras.optimizers.Adam(lr=0.005, beta_1=.90)
    tf_optimizer_u = tf.keras.optimizers.Adam(lr=0.005, beta_1=.90)

    print("Starting Adam training")
    start_time = time.time()

    # Stage 1: Adam with adaptive weight training
    for epoch in range(tf_iter):
        for i in range(n_batches):
            x0_batch = x0
            t0_batch = t0
            u0_batch = u0

            x_f_batch = x_f[i*batch_sz:(i*batch_sz + batch_sz),]
            t_f_batch = t_f[i*batch_sz:(i*batch_sz + batch_sz),]

            loss_value, mse_0, mse_f, grads, grads_col, grads_u = grad(
                u_model, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,
                x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)

            # Gradient descent for network weights
            tf_optimizer.apply_gradients(zip(grads, u_model.trainable_variables))

            # Gradient ASCENT for adaptive weights
            # Negative sign flips descent to ascent
            tf_optimizer_coll.apply_gradients(zip([-grads_col], [col_weights]))
            tf_optimizer_u.apply_gradients(zip([-grads_u], [u_weights]))

        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Time: %.2f' % (epoch, elapsed))
            tf.print(f"mse_0: {mse_0}  mse_f: {mse_f}  total: {loss_value}")
            start_time = time.time()

    # Stage 2: L-BFGS fine-tuning (weights frozen)
    print("Starting L-BFGS training")
    loss_and_flat_grad = get_loss_and_flat_grad(
        x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,
        x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)

    lbfgs(loss_and_flat_grad, get_weights(u_model), Struct(),
          maxIter=newton_iter, learningRate=0.8)
```

### Data Setup and Training Execution

```python
# Domain boundaries
lb = np.array([-1.0])  # Left boundary
ub = np.array([1.0])   # Right boundary

# Number of training points
N0 = 100      # Initial condition points
N_b = 25      # Boundary points (25 per boundary = 50 total)
N_f = 10000   # Collocation points

# Initialize adaptive weights
# col_weights: initialized to 100 (high initial attention to physics)
# u_weights: random uniform (will be trained to increase as needed)
col_weights = tf.Variable(tf.reshape(tf.repeat(100.0, N_f), (N_f, -1)))
u_weights = tf.Variable(tf.random.uniform([N0, 1]))

# Load Burgers equation data (from Raissi et al.)
data = scipy.io.loadmat('burgers_shock.mat')
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['usol']
Exact_u = np.real(Exact)

# Sample random initial condition points
idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x,:]
u0 = tf.cast(Exact_u[idx_x, 0:1], dtype=tf.float32)

# Sample boundary time points
idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t,:]

# Generate collocation points via Latin Hypercube Sampling
# LHS provides better space-filling than random sampling
X_f = lb + (ub-lb)*lhs(2, N_f)
x_f = tf.convert_to_tensor(X_f[:,0:1], dtype=tf.float32)
t_f = tf.convert_to_tensor(np.abs(X_f[:,1:2]), dtype=tf.float32)

# Construct training point sets
X0 = np.concatenate((x0, 0*x0), 1)          # Initial: (x0, 0)
X_lb = np.concatenate((0*tb + lb[0], tb), 1)  # Left: (-1, tb)
X_ub = np.concatenate((0*tb + ub[0], tb), 1)  # Right: (1, tb)

# Convert to tensors
x0 = tf.cast(X0[:,0:1], dtype=tf.float32)
t0 = tf.cast(X0[:,1:2], dtype=tf.float32)
x_lb = tf.convert_to_tensor(X_lb[:,0:1], dtype=tf.float32)
t_lb = tf.convert_to_tensor(X_lb[:,1:2], dtype=tf.float32)
x_ub = tf.convert_to_tensor(X_ub[:,0:1], dtype=tf.float32)
t_ub = tf.convert_to_tensor(X_ub[:,1:2], dtype=tf.float32)

# Train (modified iterations for demonstration: 100 Adam + 100 L-BFGS)
# Paper uses: 10000 Adam + 10000 L-BFGS
fit(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights,
    tf_iter=100, newton_iter=100)

# Evaluate on full domain
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact_u.T.flatten()[:,None]

u_pred, f_u_pred = predict(X_star)
error_u = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
print('Error u: %e' % (error_u))
```

## Critical Parameters

1. **Adaptive weight initialization**
   - Collocation weights: 100.0 (uniform, high initial value)
   - Initial condition weights: U(0,1) (uniform random)
   - High initial collocation weights ensure physics residual is prioritized early

2. **Learning rates**
   - Network (Adam): 0.005 with β₁=0.90
   - Collocation weights (Adam): 0.005 with β₁=0.90
   - Initial condition weights (Adam): 0.005 with β₁=0.90
   - L-BFGS: 0.8
   - Equal learning rates balance network and weight optimization

3. **Network architecture**
   - [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]: 8 hidden layers, 20 neurons each
   - Deeper network than baseline (6 layers) for better expressiveness
   - Tanh activation with Glorot initialization

4. **Training points**
   - Initial condition: N₀ = 100
   - Boundary points: Nᵦ = 50 (25 per boundary)
   - Collocation points: Nᵣ = 10,000
   - Latin Hypercube Sampling for better coverage than random sampling

5. **Two-stage optimization**
   - Adam iterations: 10,000 (paper), 100 (demo)
   - L-BFGS iterations: 10,000 (paper), 100 (demo)
   - Adaptive weights only trained during Adam phase
   - L-BFGS refines solution with fixed attention pattern

6. **Burgers equation parameters**
   - Viscosity: ν = 0.01/π ≈ 0.00318 (low viscosity, near-shock behavior)
   - Domain: x ∈ [-1,1], t ∈ [0,1]
   - Initial condition: u(x,0) = -sin(πx)
   - Boundary conditions: u(±1,t) = 0 (Dirichlet)

7. **Mask function**
   - Implicit linear mask: m(λ) = λ
   - Weight gradients: ∇_λ L = (1/2) * squared_residual
   - Monotonic increase ensures weights grow with residuals

8. **Performance metrics**
   - SA-PINN L2 error: ~4.80e-4 (with 10k Adam + 10k L-BFGS)
   - Baseline PINN: ~6.7e-4
   - Improvement: 28% reduction in error
   - Training time: Similar to baseline (96ms/iteration on V100 GPU)
   - Sharp discontinuity at x=0 captured with high local weights
