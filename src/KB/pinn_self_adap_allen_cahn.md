# Self-Adaptive Physics-Informed Neural Networks for Allen-Cahn Equation

**Keywords**: [PDE, parabolic, nonlinear, forward-problem, allen-cahn, reaction-diffusion, 1D, periodic, PINN, MLP, self-adaptive, adaptive-weights, strong-form, adam, lbfgs, mse, tensorflow]

**Problem:** Solving the Allen-Cahn reaction-diffusion equation, which is a "stiff" PDE with sharp space and time transitions commonly encountered in phase-field models. The baseline PINN fails to converge on stiff PDEs due to imbalance in multi-part loss functions, where gradient descent may focus on some loss components at the expense of others. The Allen-Cahn equation with periodic boundary conditions is particularly challenging for PINNs to approximate accurately.

**Issues addressed:**
- Training imbalance in multi-part loss functions where some loss components dominate others
- Difficulty fitting initial conditions in time-irreversible diffusion processes
- Poor convergence for stiff PDEs with sharp spatial and temporal transitions
- Inability to automatically identify and focus on difficult regions of the solution

## Key Method

Self-Adaptive PINNs (SA-PINNs) apply **trainable weights to individual training points** (rather than entire loss components) using a soft multiplicative attention mechanism. Each collocation point, initial condition point, and boundary point receives its own adaptive weight that is trained alongside the network weights.

**Core Principle:** The network is trained to **minimize losses while maximizing weights**, creating a saddle-point optimization problem where:
- Network weights **w** undergo gradient descent: minimize L(w, λ)
- Adaptive weights **λ** undergo gradient ascent: maximize L(w, λ)

**Loss Function:**
L(w, λᵣ, λᵦ, λ₀) = Lᵣ(w, λᵣ) + Lᵦ(w, λᵦ) + L₀(w, λ₀)

where:
- Lᵣ = (1/2) Σᵢ m(λᵢᵣ)|PDE_residual|²  (collocation points)
- Lᵦ = (1/2) Σᵢ m(λᵢᵦ)|boundary_condition|²  (boundary points)
- L₀ = (1/2) Σᵢ m(λᵢ₀)|initial_condition|²  (initial points)
- m(λ) is a strictly increasing mask function (e.g., polynomial or sigmoid)

**Gradient Updates:**
- w^(k+1) = w^k - η ∇_w L  (descent)
- λ^(k+1) = λ^k + ρ ∇_λ L  (ascent)

Since m'(λ) > 0, larger losses automatically produce larger weight updates, forcing the network to focus on difficult regions.

**For Allen-Cahn:** The method discovers that early time regions and certain spatial regions require more attention, consistent with the time-irreversible nature of diffusion processes.

## Implementation

### Network Definition and Weight Management

```python
# Network architecture: 2 inputs (x,t) -> 4 hidden layers (128 neurons) -> 1 output (u)
layer_sizes = [2, 128, 128, 128, 128, 1]

# Helper structures for L-BFGS optimization
sizes_w = []
sizes_b = []
for i, width in enumerate(layer_sizes):
    if i != 1:
        sizes_w.append(int(width * layer_sizes[1]))
        sizes_b.append(int(width if i != 0 else layer_sizes[1]))

def neural_net(layer_sizes):
    """
    Constructs fully-connected neural network with tanh activation
    """
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
    for width in layer_sizes[1:-1]:
        model.add(layers.Dense(
            width, activation=tf.nn.tanh,
            kernel_initializer="glorot_normal"))  # Xavier initialization
    model.add(layers.Dense(
            layer_sizes[-1], activation=None,  # Linear output layer
            kernel_initializer="glorot_normal"))
    return model

u_model = neural_net(layer_sizes)
```

### Self-Adaptive Loss Function

```python
def loss(x_f_batch, t_f_batch, x0, t0, u0, x_lb, t_lb, x_ub, t_ub,
         col_weights, u_weights):
    """
    Compute weighted loss with self-adaptive weights

    x_f_batch, t_f_batch: Collocation points for PDE residual
    x0, t0, u0: Initial condition points and values
    x_lb, t_lb, x_ub, t_ub: Boundary points (periodic BCs)
    col_weights: Adaptive weights for collocation points (N_f x 1)
    u_weights: Adaptive weights for initial condition (N_0 x 1)
    """
    # PDE residual at collocation points
    f_u_pred = f_model(x_f_batch, t_f_batch)

    # Initial condition prediction
    u0_pred = u_model(tf.concat([x0, t0], 1))

    # Boundary condition predictions
    u_lb_pred, u_x_lb_pred = u_x_model(u_model, x_lb, t_lb)
    u_ub_pred, u_x_ub_pred = u_x_model(u_model, x_ub, t_ub)

    # Initial condition loss with adaptive weights
    mse_0_u = tf.reduce_mean(tf.square(u_weights*(u0 - u0_pred)))

    # Periodic boundary condition loss (u and u_x match at boundaries)
    mse_b_u = tf.reduce_mean(tf.square(tf.math.subtract(u_lb_pred, u_ub_pred))) + \
              tf.reduce_mean(tf.square(tf.math.subtract(u_x_lb_pred, u_x_ub_pred)))

    # PDE residual loss with adaptive weights
    mse_f_u = tf.reduce_mean(tf.square(col_weights * f_u_pred[0]))

    return mse_0_u + mse_b_u + mse_f_u, mse_0_u, mse_b_u, mse_f_u
```

### Allen-Cahn PDE Residual

```python
@tf.function
def f_model(x, t):
    """
    Allen-Cahn PDE residual: u_t - 0.0001*u_xx + 5*u³ - 5*u = 0

    This is a reaction-diffusion equation with:
    - Small diffusion coefficient (0.0001) leading to sharp transitions
    - Cubic nonlinearity (u³) creating stiffness
    """
    u = u_model(tf.concat([x, t], 1))

    # Compute derivatives using automatic differentiation
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)

    # Allen-Cahn coefficients
    c1 = tf.constant(.0001, dtype=tf.float32)  # Diffusion coefficient
    c2 = tf.constant(5.0, dtype=tf.float32)     # Reaction coefficient

    # PDE residual
    f_u = u_t - c1*u_xx + c2*u*u*u - c2*u

    return f_u

@tf:function
def u_x_model(u_model, x, t):
    """
    Compute u and u_x for periodic boundary conditions
    """
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)
    return u, u_x
```

### Gradient Computation with Dual Descent/Ascent

```python
@tf.function
def grad(model, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,
         x_lb, t_lb, x_ub, t_ub, col_weights, u_weights):
    """
    Compute gradients for network weights (descent) and adaptive weights (ascent)
    """
    with tf.GradientTape(persistent=True) as tape:
        loss_value, mse_0, mse_b, mse_f = loss(
            x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,
            x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)

        # Gradient descent for network weights
        grads = tape.gradient(loss_value, u_model.trainable_variables)

        # Gradient ascent for adaptive weights (negative sign applied in optimizer)
        grads_col = tape.gradient(loss_value, col_weights)
        grads_u = tape.gradient(loss_value, u_weights)

        # Gradients for analysis (not used in optimization)
        gradients_u = tape.gradient(mse_0, u_model.trainable_variables)
        gradients_f = tape.gradient(mse_f, u_model.trainable_variables)

    return loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u, gradients_u, gradients_f
```

### Two-Stage Training Loop

```python
def fit(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub,
        col_weights, u_weights, tf_iter, newton_iter):
    """
    Two-stage training: Adam for adaptive weights, then L-BFGS for refinement

    tf_iter: Number of Adam iterations
    newton_iter: Number of L-BFGS iterations
    """
    batch_sz = N_f  # Full batch by default
    n_batches = N_f // batch_sz

    # Create separate optimizers for network and adaptive weights
    tf_optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
    tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
    tf_optimizer_u = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)

    print("Starting Adam training")
    start_time = time.time()

    # Stage 1: Adam optimization with adaptive weight training
    for epoch in range(tf_iter):
        for i in range(n_batches):
            x0_batch = x0
            t0_batch = t0
            u0_batch = u0

            x_f_batch = x_f[i*batch_sz:(i*batch_sz + batch_sz),]
            t_f_batch = t_f[i*batch_sz:(i*batch_sz + batch_sz),]

            loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u, g_u, g_f = grad(
                u_model, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,
                x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)

            # Gradient descent for network
            tf_optimizer.apply_gradients(zip(grads, u_model.trainable_variables))

            # Gradient ASCENT for adaptive weights (note negative sign)
            tf_optimizer_weights.apply_gradients(zip([-grads_col, -grads_u],
                                                     [col_weights, u_weights]))

        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Time: %.2f' % (epoch, elapsed))
            tf.print(f"mse_0: {mse_0}  mse_b: {mse_b}  mse_f: {mse_f}  total: {loss_value}")
            start_time = time.time()

    # Stage 2: L-BFGS fine-tuning (adaptive weights held constant)
    print("Starting L-BFGS training")
    loss_and_flat_grad = get_loss_and_flat_grad(
        x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,
        x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)

    lbfgs(loss_and_flat_grad, get_weights(u_model), Struct(),
          maxIter=newton_iter, learningRate=0.8)
```

### Data Preparation and Training Execution

```python
# Domain boundaries
lb = np.array([-1.0])
ub = np.array([1.0])

# Number of training points
N0 = 512      # Initial condition points
N_b = 100     # Boundary points
N_f = 20000   # Collocation points for PDE residual

# Initialize adaptive weights
# col_weights: uniform random initialization
# u_weights: higher initial values for initial conditions (100x larger)
col_weights = tf.Variable(tf.random.uniform([N_f, 1]))
u_weights = tf.Variable(100*tf.random.uniform([N0, 1]))

# Load Allen-Cahn data
data = scipy.io.loadmat('AC.mat')
t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['uu']
Exact_u = np.real(Exact)

# Sample initial condition points
idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x,:]
u0 = tf.cast(Exact_u[idx_x,0:1], dtype=tf.float32)

# Sample boundary time points
idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t,:]

# Generate collocation points using Latin Hypercube Sampling
X_f = lb + (ub-lb)*lhs(2, N_f)
x_f = tf.convert_to_tensor(X_f[:,0:1], dtype=tf.float32)
t_f = tf.convert_to_tensor(np.abs(X_f[:,1:2]), dtype=tf.float32)

# Construct point sets
X0 = np.concatenate((x0, 0*x0), 1)          # Initial: (x0, 0)
X_lb = np.concatenate((0*tb + lb[0], tb), 1)  # Left boundary: (-1, tb)
X_ub = np.concatenate((0*tb + ub[0], tb), 1)  # Right boundary: (1, tb)

x0 = tf.cast(X0[:,0:1], dtype=tf.float32)
t0 = tf.cast(X0[:,1:2], dtype=tf.float32)
x_lb = tf.convert_to_tensor(X_lb[:,0:1], dtype=tf.float32)
t_lb = tf.convert_to_tensor(X_lb[:,1:2], dtype=tf.float32)
x_ub = tf.convert_to_tensor(X_ub[:,0:1], dtype=tf.float32)
t_ub = tf.convert_to_tensor(X_ub[:,1:2], dtype=tf.float32)

# Train the model
fit(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights,
    tf_iter=10000, newton_iter=10000)

# Evaluate error
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact_u.T.flatten()[:,None]

u_pred, f_u_pred = predict(X_star)
error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
print('Error u: %e' % (error_u))
```

## Critical Parameters

1. **Adaptive weight initialization**
   - Collocation weights: U(0,1) - uniform random
   - Initial condition weights: U(0,100) - much higher to prioritize fitting early time
   - Initialization strategy reflects prior knowledge that initial conditions are critical for time-irreversible processes

2. **Learning rates**
   - Network weights (Adam): 0.005 with β₁=0.99
   - Adaptive weights (Adam): 0.005 with β₁=0.99
   - L-BFGS learning rate: 0.8
   - Equal learning rates for network and weights allows balanced co-training

3. **Network architecture**
   - [2, 128, 128, 128, 128, 1]: 4 hidden layers with 128 neurons each
   - Wider than typical PINNs to handle stiff PDE
   - Tanh activation with Glorot normal initialization

4. **Training points**
   - Initial condition: N₀ = 512 points
   - Boundary points: Nᵦ = 100 (periodic boundaries)
   - Collocation points: Nᵣ = 20,000
   - Large number of collocation points for accurate PDE residual

5. **Two-stage training**
   - Stage 1: 10,000 Adam iterations (adaptive weights active)
   - Stage 2: 10,000 L-BFGS iterations (adaptive weights frozen)
   - Adaptive weights only trained during Adam phase

6. **Allen-Cahn PDE parameters**
   - Diffusion coefficient: c₁ = 0.0001 (very small, creates stiffness)
   - Reaction coefficient: c₂ = 5.0
   - Domain: x ∈ [-1,1], t ∈ [0,1]
   - Periodic boundary conditions: u(-1,t) = u(1,t), uₓ(-1,t) = uₓ(1,t)
   - Initial condition: u(x,0) = x²cos(πx)

7. **Mask function**
   - Simple linear mask: m(λ) = λ (implicit in implementation)
   - Could use polynomial m(λ) = λᑫ or sigmoid for different attention sharpness

8. **Performance**
   - SA-PINN L2 error: ~2.1e-2
   - Baseline PINN: fails to converge (~96e-2)
   - Nonadaptive weighting: ~50e-2
   - Time-adaptive method: ~8.0e-2
   - Order of magnitude improvement over prior PINN methods
