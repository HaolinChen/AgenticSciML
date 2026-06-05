# Physics-Informed DeepONets for Diffusion-Reaction Systems

**Keywords**: [PDE, parabolic, nonlinear, forward-problem, reaction-diffusion, 1D, dirichlet, DeepONet, Gaussian-random-field, MLP, strong-form, adam, mse, jax]

**Problem:** DeepONets can approximate nonlinear operators between infinite-dimensional function spaces, but typically require large training datasets of paired input-output observations which are expensive to obtain. Standard DeepONets may not produce physically consistent predictions since outputs are not guaranteed to satisfy underlying PDEs. Physics-informed DeepONets address this by learning solution operators of parametric PDEs without any paired input-output data (except initial and boundary conditions) by incorporating PDE residuals as soft constraints during training.

**Issues addressed:**
- Eliminating the need for large paired input-output training datasets
- Ensuring physical consistency of predictions with governing PDEs
- Data efficiency (up to 100% reduction in required training data)
- Improving predictive accuracy when solving parametric PDEs
- Enabling fast surrogate modeling for parametric PDE systems

## Key Method

Physics-informed DeepONets extend the DeepONet architecture to solve parametric PDEs by learning the solution operator **G: u → s** that maps variable input functions (source terms, boundary conditions, coefficients, etc.) to corresponding PDE solutions. The architecture combines:

**DeepONet Structure:**
- **Branch network**: Encodes input functions u(x) evaluated at fixed sensor locations {x_i}
- **Trunk network**: Encodes output evaluation coordinates y
- **Operator approximation**: G_θ(u)(y) = Σ_k b_k(u(x_1),...,u(x_m)) · t_k(y)

**Physics-Informed Loss:**

The total loss combines data fitting and physics constraints:

L(θ) = L_operator(θ) + L_physics(θ)

where:
- **L_operator(θ)**: Enforces initial and boundary conditions
- **L_physics(θ)**: Penalizes PDE residual violations using automatic differentiation

For the diffusion-reaction system:

∂s/∂t = D ∂²s/∂x² + ks² + u(x)

The PDE residual is:

R_θ(x,t) = ∂G_θ(u)(x,t)/∂t - D ∂²G_θ(u)(x,t)/∂x² - k[G_θ(u)(x,t)]² - u(x)

**Key advantages:**
- No paired input-output training data required (only IC/BC needed)
- Physics consistency guaranteed through PDE residual minimization
- Continuous representation independent of mesh resolution
- Can predict O(10³) PDE solutions in fractions of a second (3 orders of magnitude faster than traditional solvers)
- Strong generalization to out-of-distribution inputs

## Implementation

### DeepONet Architecture with Branch and Trunk Networks

```python
# Define MLP network builder
def MLP(layers, activation=np.tanh):
    """
    Vanilla multi-layer perceptron
    layers: list of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
    activation: activation function (default: tanh)
    """
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            # Xavier/Glorot initialization
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params

    def apply(params, inputs):
        # Forward pass through hidden layers
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = activation(outputs)
        # Output layer (no activation)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs

    return init, apply
```

### Physics-Informed DeepONet Model Class

```python
class PI_DeepONet:
    def __init__(self, branch_layers, trunk_layers):
        """
        branch_layers: architecture for branch network [m, hidden1, ..., p]
        trunk_layers: architecture for trunk network [2, hidden1, ..., p]
        where m = number of input sensors, p = latent dimension
        """
        # Initialize branch and trunk networks
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=np.tanh)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=np.tanh)

        # Initialize parameters
        branch_params = self.branch_init(rng_key=random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key=random.PRNGKey(4321))
        params = (branch_params, trunk_params)

        # Setup Adam optimizer with exponential learning rate decay
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(
            optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.9)
        )
        self.opt_state = self.opt_init(params)

        self.itercount = itertools.count()

        # Loggers for tracking training progress
        self.loss_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    def operator_net(self, params, u, x, t):
        """
        DeepONet forward pass: computes G_θ(u)(x,t)
        params: (branch_params, trunk_params)
        u: input function evaluated at sensor locations, shape (m,)
        x, t: output evaluation coordinates (scalars)
        Returns: scalar prediction s(x,t)
        """
        branch_params, trunk_params = params
        y = np.stack([x, t])  # Combine spatial and temporal coordinates
        # Branch network encodes input function
        B = self.branch_apply(branch_params, u)
        # Trunk network encodes output coordinates
        T = self.trunk_apply(trunk_params, y)
        # Compute dot product to get operator output
        outputs = np.sum(B * T)
        return outputs
```

### PDE Residual Computation Using Automatic Differentiation

```python
def residual_net(self, params, u, x, t):
    """
    Compute PDE residual for diffusion-reaction system:
    R = ∂s/∂t - D·∂²s/∂x² - k·s² - u(x)

    Uses JAX automatic differentiation to compute derivatives
    """
    # Solution prediction
    s = self.operator_net(params, u, x, t)

    # Compute derivatives using automatic differentiation
    # First derivatives
    s_t = grad(self.operator_net, argnums=3)(params, u, x, t)  # ∂s/∂t
    s_x = grad(self.operator_net, argnums=2)(params, u, x, t)  # ∂s/∂x
    # Second derivative
    s_xx = grad(grad(self.operator_net, argnums=2), argnums=2)(params, u, x, t)  # ∂²s/∂x²

    # PDE residual: R = s_t - D·s_xx - k·s² - u(x)
    # Note: u(x) term is handled separately in loss computation
    res = s_t - 0.01 * s_xx - 0.01 * s**2
    return res
```

### Loss Functions

```python
def loss_bcs(self, params, batch):
    """
    Boundary/Initial condition loss: enforces zero IC/BC
    L_operator = (1/NP) Σ |G_θ(u)(x,t)|²
    """
    inputs, outputs = batch
    u, y = inputs  # u: input functions, y: BC/IC locations

    # Vectorized forward pass over batch
    s_pred = vmap(self.operator_net, (None, 0, 0, 0))(
        params, u, y[:,0], y[:,1]
    )

    # MSE loss (outputs are zeros for zero BC/IC)
    loss = np.mean((outputs.flatten() - s_pred)**2)
    return loss

def loss_res(self, params, batch):
    """
    Physics residual loss: enforces PDE satisfaction
    L_physics = (1/NQ) Σ |R_θ(x,t) - u(x)|²
    """
    inputs, outputs = batch
    u, y = inputs  # u: input functions, y: collocation points

    # Compute PDE residual at collocation points
    pred = vmap(self.residual_net, (None, 0, 0, 0))(
        params, u, y[:,0], y[:,1]
    )

    # MSE loss (outputs contain source term u(x) values)
    loss = np.mean((outputs.flatten() - pred)**2)
    return loss

def loss(self, params, bcs_batch, res_batch):
    """Total loss: L = L_operator + L_physics"""
    loss_bcs = self.loss_bcs(params, bcs_batch)
    loss_res = self.loss_res(params, res_batch)
    loss = loss_bcs + loss_res
    return loss
```

### Training Loop with JIT Compilation

```python
@partial(jit, static_argnums=(0,))
def step(self, i, opt_state, bcs_batch, res_batch):
    """
    Single optimization step (JIT compiled for speed)
    Computes gradients and updates parameters
    """
    params = self.get_params(opt_state)
    # Compute gradients of total loss
    g = grad(self.loss)(params, bcs_batch, res_batch)
    # Update parameters using Adam
    return self.opt_update(i, g, opt_state)

def train(self, bcs_dataset, res_dataset, nIter=10000):
    """
    Main training loop
    bcs_dataset: BC/IC data loader
    res_dataset: collocation points data loader
    nIter: number of training iterations
    """
    bcs_data = iter(bcs_dataset)
    res_data = iter(res_dataset)

    pbar = trange(nIter)
    for it in pbar:
        # Fetch mini-batches
        bcs_batch = next(bcs_data)
        res_batch = next(res_data)

        # Perform optimization step
        self.opt_state = self.step(
            next(self.itercount), self.opt_state, bcs_batch, res_batch
        )

        # Log losses every 100 iterations
        if it % 100 == 0:
            params = self.get_params(self.opt_state)
            loss_value = self.loss(params, bcs_batch, res_batch)
            loss_bcs_value = self.loss_bcs(params, bcs_batch)
            loss_res_value = self.loss_res(params, res_batch)

            self.loss_log.append(loss_value)
            self.loss_bcs_log.append(loss_bcs_value)
            self.loss_res_log.append(loss_res_value)

            pbar.set_postfix({
                'Loss': loss_value,
                'loss_bcs': loss_bcs_value,
                'loss_physics': loss_res_value
            })
```

### Data Generation Using Gaussian Random Fields

```python
def generate_one_training_data(key, P, Q):
    """
    Generate training data for one input function
    P: number of BC/IC points
    Q: number of collocation points
    Returns: BC/IC data and collocation point data
    """
    # Solve PDE numerically to get one sample function u(x)
    (x, t, UU), (u, y, s) = solve_ADR(key, Nx, Nt, P, length_scale)

    subkeys = random.split(key, 4)

    # Sample BC/IC points
    # Left boundary (x=0), right boundary (x=1), initial condition (t=0)
    x_bc1 = np.zeros((P // 3, 1))  # x = 0
    x_bc2 = np.ones((P // 3, 1))   # x = 1
    x_bc3 = random.uniform(key=subkeys[0], shape=(P // 3, 1))  # t = 0
    x_bcs = np.vstack((x_bc1, x_bc2, x_bc3))

    t_bc1 = random.uniform(key=subkeys[1], shape=(P//3 * 2, 1))  # for boundaries
    t_bc2 = np.zeros((P//3, 1))  # for initial condition
    t_bcs = np.vstack([t_bc1, t_bc2])

    # BC/IC training data (u repeated P times, zero boundary conditions)
    u_train = np.tile(u, (P, 1))
    y_train = np.hstack([x_bcs, t_bcs])
    s_train = np.zeros((P, 1))  # Zero BC/IC

    # Sample collocation points for PDE residual
    x_r_idx = random.choice(subkeys[2], np.arange(Nx), shape=(Q, 1))
    x_r = x[x_r_idx]
    t_r = random.uniform(subkeys[3], minval=0, maxval=1, shape=(Q, 1))

    # Collocation point data
    u_r_train = np.tile(u, (Q, 1))
    y_r_train = np.hstack([x_r, t_r])
    f_r_train = u[x_r_idx]  # Source term values at collocation points

    return u_train, y_train, s_train, u_r_train, y_r_train, f_r_train
```

### Main Training Script

```python
# Problem parameters
D = 0.01  # Diffusion coefficient
k = 0.01  # Reaction rate
length_scale = 0.2  # GRF length scale for input functions

# Grid resolution
Nx = 100  # Spatial points
Nt = 100  # Temporal points

# Training data configuration
N = 5000  # Number of input function samples
m = Nx    # Number of input sensors (where u(x) is evaluated)
P_train = 300  # BC/IC points per sample (100 per side)
Q_train = 100  # Collocation points per sample

# Generate training data from N different Gaussian random fields
key = random.PRNGKey(0)
keys = random.split(key, N)
u_train, y_train, s_train, u_r_train, y_r_train, f_r_train = vmap(
    generate_one_training_data, (0, None, None)
)(keys, P_train, Q_train)

# Reshape data for batch processing
u_bcs_train = np.float32(u_train.reshape(N * P_train, -1))
y_bcs_train = np.float32(y_train.reshape(N * P_train, -1))
s_bcs_train = np.float32(s_train.reshape(N * P_train, -1))

u_res_train = np.float32(u_r_train.reshape(N * Q_train, -1))
y_res_train = np.float32(y_r_train.reshape(N * Q_train, -1))
f_res_train = np.float32(f_r_train.reshape(N * Q_train, -1))

# Initialize model
branch_layers = [m, 50, 50, 50, 50, 50]  # m inputs → 50 latent features
trunk_layers = [2, 50, 50, 50, 50, 50]   # (x,t) → 50 latent features
model = PI_DeepONet(branch_layers, trunk_layers)

# Create data loaders
batch_size = 10000
bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size)
res_dataset = DataGenerator(u_res_train, y_res_train, f_res_train, batch_size)

# Train model
model.train(bcs_dataset, res_dataset, nIter=120000)

# Prediction
params = model.get_params(model.opt_state)
s_pred = model.predict_s(params, u_test, y_test)
error = np.linalg.norm(s_test - s_pred) / np.linalg.norm(s_test)
print(f'Relative L2 error: {error:.6f}')
```

## Critical Parameters

1. **Network Architecture**
   - Branch network: [100, 50, 50, 50, 50, 50] (5 hidden layers, 50 neurons each)
   - Trunk network: [2, 50, 50, 50, 50, 50] (5 hidden layers, 50 neurons each)
   - Activation: tanh (throughout all hidden layers)
   - Latent dimension p: 50 (output dimension of both branch and trunk)

2. **Training Data**
   - Number of input functions N: 5000 (sampled from Gaussian random fields)
   - Input sensors m: 100 (fixed locations where u(x) is evaluated)
   - BC/IC points P: 300 per sample (100 for left boundary, 100 for right boundary, 100 for initial condition)
   - Collocation points Q: 100 per sample (randomly sampled in domain)
   - GRF length scale: 0.2 (controls smoothness of input functions)

3. **Optimizer Configuration**
   - Algorithm: Adam with exponential learning rate decay
   - Initial learning rate: 1e-3
   - Decay rate: 0.9 every 2000 steps
   - Batch size: 10000
   - Training iterations: 120000

4. **PDE-Specific Parameters**
   - Diffusion coefficient D: 0.01
   - Reaction rate k: 0.01
   - Domain: x ∈ [0,1], t ∈ [0,1]
   - Boundary conditions: s(0,t) = s(1,t) = 0
   - Initial condition: s(x,0) = 0

5. **Loss Weighting**
   - L_operator and L_physics are equally weighted (coefficient = 1.0)
   - No additional balancing needed due to similar magnitudes

6. **Numerical Precision**
   - Data generation: float64 (for accurate GP sampling)
   - Training: float32 (for computational efficiency)
   - JAX backend with JIT compilation for performance

7. **Initialization**
   - Xavier/Glorot initialization for network weights
   - Different random seeds for branch (1234) and trunk (4321) networks
   - Ensures proper gradient flow at initialization

8. **Key Performance Metrics**
   - Training relative L2 error: ~0.007 (0.7%)
   - Inference speed: O(10³) PDEs solved in fraction of a second
   - 3 orders of magnitude speedup vs traditional PDE solvers
   - 80% improvement in accuracy vs standard DeepONet with 100% data reduction
