# Self-Adaptive Physics-Informed Neural Networks for Helmholtz Equation

**Keywords**: [PDE, elliptic, linear, forward-problem, helmholtz, 2D, dirichlet, PINN, MLP, self-adaptive, adaptive-weights, strong-form, adam, lbfgs, mse, tensorflow]

**Problem:** Solving the 2D Helmholtz equation, which describes wave and diffusion processes in spatial domains. While the equation is linear, high-frequency oscillatory solutions (large wavenumber) can be challenging for neural networks to approximate. Baseline PINNs struggle with multi-part loss imbalance where certain loss components (e.g., boundaries vs residuals) dominate training, preventing accurate overall solutions.

**Issues addressed:**
- Loss component imbalance in multi-objective optimization
- Difficulty capturing high-frequency oscillatory patterns
- Non-uniform difficulty across the spatial domain
- Suboptimal attention allocation in regions with varying solution complexity

## Key Method

Self-Adaptive PINNs (SA-PINNs) implement **individualized trainable weights** for each training point using a soft attention mechanism. Unlike global weighting schemes that scale entire loss components uniformly, SA-PINNs allow the network to discover and focus on difficult regions automatically through point-wise adaptive weights.

**Core Concept:**
The loss function applies trainable masks m(λᵢ) to individual point residuals, where λᵢ are optimized via gradient ascent:
- Network seeks to **minimize** weighted losses: min_w L(w, λ)
- Weights seek to **maximize** losses: max_λ L(w, λ)
- Equilibrium: weights increase at high-residual points, forcing network improvement

**Loss Formulation:**
L(w, λᵣ) = Lᵣ(w, λᵣ) + Lᵦ(w)

where:
- Lᵣ = (1/2) Σᵢ m(λᵢᵣ)|∇²u + k²u - q|²  (PDE residual, weighted)
- Lᵦ = (1/2) Σⱼ |u_boundary - 0|²  (Dirichlet BCs, unweighted)
- m(λ) is a monotonic increasing function (e.g., identity or polynomial)

**Gradient Dynamics:**
Since ∇_λ L ∝ |residual|², points with larger errors automatically receive larger weight updates, implementing a form of automatic curriculum learning where the network progressively focuses on stubborn regions.

**For Helmholtz:** The method discovers that non-flat regions with high-frequency oscillations require more attention, while flat regions near zero can be learned with minimal weight.

## Implementation

### Network Definition

```python
# Network architecture: 2 inputs (x,y) -> 4 hidden layers (50 neurons) -> 1 output (u)
layer_sizes = [2, 50, 50, 50, 50, 1]

# Size tracking for L-BFGS weight flattening
sizes_w = []
sizes_b = []
for i, width in enumerate(layer_sizes):
    if i != 1:
        sizes_w.append(int(width * layer_sizes[1]))
        sizes_b.append(int(width if i != 0 else layer_sizes[1]))

def neural_net(layer_sizes):
    """
    Construct fully-connected network with tanh activation

    Architecture optimized for 2D spatial problems
    Glorot initialization provides good starting weights
    """
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))

    # Hidden layers with tanh activation
    for width in layer_sizes[1:-1]:
        model.add(layers.Dense(
            width, activation=tf.nn.tanh,
            kernel_initializer="glorot_normal"))

    # Linear output layer
    model.add(layers.Dense(
            layer_sizes[-1], activation=None,
            kernel_initializer="glorot_normal"))

    return model

u_model = neural_net(layer_sizes)
```

### Self-Adaptive Loss Function

```python
def loss(x_f, y_f, x_lb, y_lb, x_ub, y_ub, x_rb, y_rb, x_lftb, y_lftb,
         col_weights):
    """
    Compute weighted loss for 2D Helmholtz equation

    Parameters:
    -----------
    x_f, y_f: Collocation points for PDE residual
    x_lb, y_lb: Lower boundary (y = -1)
    x_ub, y_ub: Upper boundary (y = 1)
    x_rb, y_rb: Right boundary (x = 1)
    x_lftb, y_lftb: Left boundary (x = -1)
    col_weights: Adaptive weights for collocation points (N_f x 1)

    Returns:
    --------
    total_loss, boundary_loss, residual_loss
    """
    # PDE residual at collocation points
    f_u_pred = f_model(x_f, y_f)

    # Boundary predictions (Dirichlet: u = 0 on all boundaries)
    u_lb_pred = u_model(tf.concat([x_lb, y_lb], 1))
    u_ub_pred = u_model(tf.concat([x_ub, y_ub], 1))
    u_rb_pred = u_model(tf.concat([x_rb, y_rb], 1))
    u_lftb_pred = u_model(tf.concat([x_lftb, y_lftb], 1))

    # Boundary loss (unweighted, all boundaries should be zero)
    mse_b_u = tf.reduce_mean(tf.square(u_lb_pred - 0)) + \
              tf.reduce_mean(tf.square(u_ub_pred - 0)) + \
              tf.reduce_mean(tf.square(u_rb_pred - 0)) + \
              tf.reduce_mean(tf.square(u_lftb_pred - 0))

    # Weighted PDE residual loss
    # col_weights provide adaptive attention to different regions
    mse_f_u = tf.reduce_mean(tf.square(col_weights*f_u_pred))

    return mse_b_u + mse_f_u, mse_b_u, mse_f_u
```

### Helmholtz PDE Residual with Forcing Term

```python
def f_model(x, y):
    """
    Helmholtz equation residual: u_xx + u_yy + k²u - q(x,y) = 0

    The forcing term q(x,y) is constructed to give analytical solution:
    u(x,y) = sin(a₁πx)sin(a₂πy)

    For a₁=1, a₂=4, k²=1, this creates high-frequency oscillations in y
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)

        # Network prediction
        u = u_model(tf.concat([x, y], 1))

        # First derivatives
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)

    # Second derivatives (requires separate gradient calls)
    u_xx = tape.gradient(u_x, x)
    u_yy = tape.gradient(u_y, y)

    del tape

    # Problem parameters
    a1 = 1.0   # Frequency in x direction
    a2 = 4.0   # Frequency in y direction (high frequency)
    ksq = 1.0  # Helmholtz wavenumber squared

    # Forcing term constructed from analytical solution
    # forcing = -(a₁π)²sin(a₁πx)sin(a₂πy) - (a₂π)²sin(a₁πx)sin(a₂πy) + k²sin(a₁πx)sin(a₂πy)
    #         = [k² - (a₁π)² - (a₂π)²] * sin(a₁πx)sin(a₂πy)
    forcing = - (a1*math.pi)**2*np.sin(a1*math.pi*x)*np.sin(a2*math.pi*y) - \
                (a2*math.pi)**2*np.sin(a1*math.pi*x)*np.sin(a2*math.pi*y) + \
                ksq*np.sin(a1*math.pi*x)*np.sin(a2*math.pi*y)

    # Helmholtz residual
    f_u = u_xx + u_yy + ksq*u - forcing

    return f_u
```

### Training Loop with Adaptive Weights

```python
def fit(x_f, y_f, x_lb, y_lb, x_ub, y_ub, x_rb, y_rb, x_lftb, y_lftb,
        col_weights, tf_iter, newton_iter):
    """
    Two-stage training with self-adaptive collocation weights

    Only collocation points have adaptive weights in this implementation
    Boundary conditions use fixed unit weights
    """
    batch_sz = N_f
    n_batches = N_f // batch_sz

    # Optimizers with high momentum (beta_1=0.99)
    tf_optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=.99)
    tf_optimizer_coll = tf.keras.optimizers.Adam(lr=0.001, beta_1=.99)

    print("Starting Adam training")
    start_time = time.time()

    # Stage 1: Adam optimization with adaptive weights
    for epoch in range(tf_iter):
        for i in range(n_batches):
            x_f_batch = x_f[i*batch_sz:(i*batch_sz + batch_sz),]
            y_f_batch = y_f[i*batch_sz:(i*batch_sz + batch_sz),]

            # Compute loss and gradients
            with tf.GradientTape(persistent=True) as tape:
                loss_value, mse_b, mse_f = loss(
                    x_f, y_f, x_lb, y_lb, x_ub, y_ub, x_rb, y_rb,
                    x_lftb, y_lftb, col_weights)

                # Gradients for network (descent)
                grads = tape.gradient(loss_value, u_model.trainable_variables)

                # Gradients for adaptive weights (ascent)
                grads_col = tape.gradient(loss_value, col_weights)

            # Apply gradient descent to network
            tf_optimizer.apply_gradients(zip(grads, u_model.trainable_variables))

            # Apply gradient ASCENT to adaptive weights (negative sign)
            tf_optimizer_coll.apply_gradients(zip([-grads_col], [col_weights]))

            del tape

        # Progress logging
        elapsed = time.time() - start_time
        print('It: %d, Time: %.2f' % (epoch, elapsed))
        tf.print(f"mse_b: {mse_b}  mse_f: {mse_f}  total: {loss_value}")
        start_time = time.time()

    # Stage 2: L-BFGS refinement (weights frozen)
    print("Starting L-BFGS training")
    loss_and_flat_grad = get_loss_and_flat_grad(
        x_f, y_f, x_lb, y_lb, x_ub, y_ub, x_rb, y_rb, x_lftb, y_lftb, col_weights)

    lbfgs(loss_and_flat_grad, get_weights(u_model), Struct(),
          maxIter=newton_iter, learningRate=0.8)
```

### Data Preparation and Execution

```python
# Domain boundaries
lb = np.array([-1.0])
ub = np.array([1.0])
rb = np.array([1.0])
lftb = np.array([-1.0])

# Number of training points
N0 = 200       # Not used (no initial condition for steady-state)
N_b = 100      # Boundary points per edge
N_f = 100000   # Collocation points

# Initialize adaptive weights (random uniform)
col_weights = tf.Variable(tf.random.uniform([N_f, 1]))
u_weights = tf.Variable(100*tf.random.uniform([N0, 1]))  # Not used

# Create fine mesh for evaluation
nx, ny = (1001, 1001)
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
xv, yv = np.meshgrid(x, y)

# Analytical solution: u(x,y) = sin(πx)sin(4πy)
Exact_u = np.sin(math.pi*xv)*np.sin(4*math.pi*yv)

# Reshape for sampling
x = np.reshape(x, (-1,1))
y = np.reshape(y, (-1,1))

# Sample boundary points
idx_y = np.random.choice(y.shape[0], N_b, replace=False)
yb = y[idx_y, :]

# Generate collocation points via Latin Hypercube Sampling
X_f = lb + (ub-lb)*lhs(2, N_f)
x_f = tf.convert_to_tensor(X_f[:,0:1], dtype=tf.float32)
y_f = tf.convert_to_tensor(X_f[:,1:2], dtype=tf.float32)

# Construct boundary point sets
X_lb = np.concatenate((yb, 0*yb + lb[0]), 1)      # Lower: (x, -1)
X_ub = np.concatenate((yb, 0*yb + ub[0]), 1)      # Upper: (x, 1)
X_rb = np.concatenate((0*yb + rb[0], yb), 1)      # Right: (1, y)
X_lftb = np.concatenate((0*yb + lftb[0], yb), 1)  # Left: (-1, y)

# Convert to tensors
x_lb = tf.convert_to_tensor(X_lb[:,0:1], dtype=tf.float32)
y_lb = tf.convert_to_tensor(X_lb[:,1:2], dtype=tf.float32)
x_ub = tf.convert_to_tensor(X_ub[:,0:1], dtype=tf.float32)
y_ub = tf.convert_to_tensor(X_ub[:,1:2], dtype=tf.float32)
x_rb = tf.convert_to_tensor(X_rb[:,0:1], dtype=tf.float32)
y_rb = tf.convert_to_tensor(X_rb[:,1:2], dtype=tf.float32)
x_lftb = tf.convert_to_tensor(X_lftb[:,0:1], dtype=tf.float32)
y_lftb = tf.convert_to_tensor(X_lftb[:,1:2], dtype=tf.float32)

# Train the model
fit(x_f, y_f, x_lb, y_lb, x_ub, y_ub, x_rb, y_rb, x_lftb, y_lftb,
    col_weights, tf_iter=10000, newton_iter=10000)

# Evaluate on full mesh
X, Y = np.meshgrid(x, y)
X_star = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
u_star = Exact_u.flatten()[:,None]

u_pred, f_u_pred = predict(X_star)
error_u = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
print('Error u: %e' % (error_u))
```

## Critical Parameters

1. **Adaptive weight initialization**
   - Collocation weights: U(0,1) - uniform random initialization
   - Only collocation points have adaptive weights in this problem
   - Boundary weights implicitly set to 1 (not trainable)

2. **Learning rates**
   - Network (Adam): 0.001 with β₁=0.99 (high momentum)
   - Collocation weights (Adam): 0.001 with β₁=0.99
   - L-BFGS: 0.8
   - Lower learning rate than other examples (0.001 vs 0.005) for stability

3. **Network architecture**
   - [2, 50, 50, 50, 50, 1]: 4 hidden layers with 50 neurons each
   - Matches architecture from Wang et al. learning rate annealing paper
   - Tanh activation with Glorot normal initialization

4. **Training points**
   - Collocation points: Nᵣ = 100,000 (large for accurate PDE residual)
   - Boundary points: Nᵦ = 400 (100 per edge × 4 edges)
   - No initial condition (steady-state problem)
   - Latin Hypercube Sampling for space-filling collocation distribution

5. **Two-stage training**
   - Adam: 10,000 iterations (adaptive weights active)
   - L-BFGS: 10,000 iterations (adaptive weights frozen)
   - Total: 20,000 iterations (vs 40,000 in baseline from comparison paper)
   - 50% fewer iterations with better or comparable accuracy

6. **Helmholtz equation parameters**
   - Wavenumber squared: k² = 1.0
   - Frequency parameters: a₁ = 1, a₂ = 4
   - Domain: (x,y) ∈ [-1,1]²
   - Boundary conditions: u = 0 on all four edges (Dirichlet)
   - Analytical solution: u(x,y) = sin(πx)sin(4πy)
   - High-frequency oscillations in y-direction (4 cycles vs 1 in x)

7. **Mask function**
   - Linear mask: m(λ) = λ (implicit identity)
   - Weight gradients: ∇_λ L ∝ |PDE_residual|²
   - Regions with high residuals get stronger weight increases

8. **Spatial adaptation pattern**
   - Flat regions (near zeros of sin functions): low weights
   - High-amplitude oscillatory regions: high weights
   - Non-uniform attention matches solution complexity
   - Self-discovered pattern stable across random restarts

9. **Performance metrics**
   - SA-PINN L2 error: ~3.2e-3 ± 2.2e-4
   - Baseline PINN: ~1.4e-1 (with larger network, more iterations)
   - Learning rate annealing methods: ~2.54e-3 to 2.74e-2
   - Competitive accuracy with fewer training iterations
   - Automatically identifies difficult regions without manual tuning

10. **Evaluation mesh**
    - Grid size: 1001 × 1001 = 1,002,001 points
    - Fine mesh captures high-frequency oscillations accurately
    - Cubic interpolation for visualization
