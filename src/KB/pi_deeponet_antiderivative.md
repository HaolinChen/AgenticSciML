# Physics-Informed DeepONet for Anti-derivative Operator Learning

**Keywords**: [ODE, forward-problem, antiderivative, 1D, DeepONet, MLP, strong-form, adam, mse, jax]

**Problem:** Traditional DeepONets require large paired input-output datasets to learn solution operators of parametric differential equations. This is expensive and often impractical. The goal is to learn the anti-derivative operator that maps input functions u(x) to their integrals s(x) without requiring extensive paired training data, relying instead on the governing ODE and initial conditions.

**Issues addressed:**
- Data scarcity in operator learning - reduces need for large paired input-output datasets
- Inconsistent predictions that violate physical laws in standard DeepONets
- Inability to learn operators with only boundary/initial conditions

## Key Method

**Physics-Informed DeepONet** extends the standard DeepONet architecture by incorporating physical constraints directly into the loss function through automatic differentiation. For the anti-derivative operator problem:

**Problem formulation:**
```
ds(x)/dx = u(x),  x ∈ [0, 1]
s(0) = 0
```

The solution operator is:
```
G: u(x) → s(x) = s(0) + ∫₀ˣ u(t)dt
```

**DeepONet Architecture:**
The operator G is approximated by:
```
G_θ(u)(y) = Σₖ₌₁ᵍ bₖ(u(x₁), u(x₂), ..., u(xₘ)) · tₖ(y)
```

where:
- **Branch network** `bₖ`: processes input function u evaluated at m sensor locations
- **Trunk network** `tₖ`: processes output query locations y
- Output is the dot product of branch and trunk outputs

**Loss Function:**
```
L(θ) = L_operator(θ) + L_physics(θ)
```

**Operator Loss** (enforces initial condition):
```
L_operator(θ) = (1/N) Σᵢ₌₁ᴺ |G_θ(u⁽ⁱ⁾)(0)|²
```

**Physics Loss** (enforces ODE):
```
L_physics(θ) = (1/(N·m)) Σᵢ₌₁ᴺ Σⱼ₌₁ᵐ |dG_θ(u⁽ⁱ⁾)(y)/dy|_{y=xⱼ} - u⁽ⁱ⁾(xⱼ)|²
```

**Key advantages:**
- Can train without paired input-output data, only initial conditions needed
- Predictions automatically satisfy the governing ODE
- Significantly improved data efficiency (10x less data for same accuracy)
- Once trained, can rapidly solve O(10³) ODEs in milliseconds

## Implementation

### DeepONet Architecture

```python
# Define MLP network builder
def MLP(layers, activation=np.tanh):
    '''Vanilla MLP network with specified layers and activation'''
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            # Xavier/Glorot initialization for stable training
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        # Initialize all layers
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params

    def apply(params, inputs):
        # Forward pass through all hidden layers
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = activation(outputs)
        # Output layer (no activation)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs

    return init, apply
```

### Physics-Informed DeepONet Model

```python
class PI_DeepONet:
    def __init__(self, branch_layers, trunk_layers):
        # Initialize branch network (processes input function u)
        # and trunk network (processes query locations y)
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=np.tanh)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=np.tanh)

        # Initialize network parameters with different random seeds
        branch_params = self.branch_init(rng_key = random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
        params = (branch_params, trunk_params)

        # Setup Adam optimizer with exponential learning rate decay
        # Initial LR: 1e-3, decay rate: 0.9 every 1000 iterations
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3,
                                                                      decay_steps=1000,
                                                                      decay_rate=0.9))
        self.opt_state = self.opt_init(params)

        self.itercount = itertools.count()

        # Loggers for tracking training
        self.loss_log = []
        self.loss_operator_log = []
        self.loss_physics_log = []

    def operator_net(self, params, u, y):
        """
        DeepONet forward pass: computes G_θ(u)(y)

        Args:
            params: (branch_params, trunk_params) network parameters
            u: input function evaluated at sensor locations, shape (m,)
            y: query location, shape (1,)

        Returns:
            s: predicted output G_θ(u)(y), scalar
        """
        branch_params, trunk_params = params
        # Branch network processes input function
        B = self.branch_apply(branch_params, u)
        # Trunk network processes query location
        T = self.trunk_apply(trunk_params, y)
        # Output is dot product of branch and trunk
        outputs = np.sum(B * T)
        return outputs

    def residual_net(self, params, u, y):
        """
        Compute ODE residual: ds/dy using automatic differentiation

        Args:
            params: network parameters
            u: input function
            y: query location

        Returns:
            s_y: derivative ds/dy at location y
        """
        # Use JAX automatic differentiation to compute gradient w.r.t. y
        s_y = grad(self.operator_net, argnums=2)(params, u, y)
        return s_y

    def loss_operator(self, params, batch):
        """
        Operator loss: enforces initial condition s(0) = 0

        Args:
            batch: ((u, y), s) where
                u: input functions, shape (N, m)
                y: query locations, shape (N, 1)
                s: target outputs, shape (N, 1)

        Returns:
            loss: mean squared error between predictions and targets
        """
        inputs, outputs = batch
        u, y = inputs
        # Vectorized forward pass for batch
        pred = vmap(self.operator_net, (None, 0, 0))(params, u, y)
        # MSE loss
        loss = np.mean((outputs.flatten() - pred.flatten())**2)
        return loss

    def loss_physics(self, params, batch):
        """
        Physics loss: enforces ODE ds/dx = u(x)

        Args:
            batch: ((u_r, y_r), s_r) where
                u_r: input functions, shape (N*m, m)
                y_r: collocation points, shape (N*m, 1)
                s_r: target residuals u(x), shape (N*m, 1)

        Returns:
            loss: mean squared error of ODE residual
        """
        inputs, outputs = batch
        u, y = inputs
        # Compute derivatives ds/dy at collocation points
        pred = vmap(self.residual_net, (None, 0, 0))(params, u, y)
        # MSE between ds/dy and u (should be equal by ODE)
        loss = np.mean((outputs.flatten() - pred.flatten())**2)
        return loss

    def loss(self, params, operator_batch, physics_batch):
        """
        Total loss: combination of operator and physics losses
        """
        loss_operator = self.loss_operator(params, operator_batch)
        loss_physics = self.loss_physics(params, physics_batch)
        loss = loss_operator + loss_physics
        return loss

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, operator_batch, physics_batch):
        """
        Single optimization step using JAX JIT compilation

        Args:
            i: iteration counter
            opt_state: optimizer state
            operator_batch: batch for operator loss
            physics_batch: batch for physics loss

        Returns:
            updated optimizer state
        """
        params = self.get_params(opt_state)
        # Compute gradients of total loss
        g = grad(self.loss)(params, operator_batch, physics_batch)
        # Update parameters using Adam
        return self.opt_update(i, g, opt_state)

    def train(self, operator_dataset, physics_dataset, nIter=10000):
        """
        Main training loop

        Args:
            operator_dataset: data for enforcing initial conditions
            physics_dataset: data for enforcing ODE residual
            nIter: number of training iterations
        """
        operator_data = iter(operator_dataset)
        physics_data = iter(physics_dataset)

        pbar = trange(nIter)
        for it in pbar:
            # Fetch batches for both losses
            operator_batch = next(operator_data)
            physics_batch = next(physics_data)

            # Perform optimization step
            self.opt_state = self.step(next(self.itercount),
                                      self.opt_state,
                                      operator_batch,
                                      physics_batch)

            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                # Compute and log losses
                loss_value = self.loss(params, operator_batch, physics_batch)
                loss_operator_value = self.loss_operator(params, operator_batch)
                loss_physics_value = self.loss_physics(params, physics_batch)

                self.loss_log.append(loss_value)
                self.loss_operator_log.append(loss_operator_value)
                self.loss_physics_log.append(loss_physics_value)

                # Display progress
                pbar.set_postfix({'Loss': loss_value,
                                  'loss_operator': loss_operator_value,
                                  'loss_physics': loss_physics_value})

    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        """
        Predict solution s at query points

        Args:
            params: trained network parameters
            U_star: input functions, shape (N, m)
            Y_star: query locations, shape (N, 1)

        Returns:
            s_pred: predicted outputs, shape (N,)
        """
        s_pred = vmap(self.operator_net, (None, 0, 0))(params, U_star, Y_star)
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_s_y(self, params, U_star, Y_star):
        """
        Predict derivative ds/dy at query points

        Args:
            params: trained network parameters
            U_star: input functions
            Y_star: query locations

        Returns:
            s_y_pred: predicted derivatives (should equal u)
        """
        s_y_pred = vmap(self.residual_net, (None, 0, 0))(params, U_star, Y_star)
        return s_y_pred
```

### Data Generation from Gaussian Process

```python
def generate_one_training_data(key, m=100, P=1):
    """
    Generate one training sample: random function u from Gaussian process

    Args:
        key: random key for reproducibility
        m: number of input sensors
        P: number of output sensors

    Returns:
        u_train: input function values at sensors
        y_train: output query locations
        s_train: output values (integral of u)
        u_r_train, y_r_train, s_r_train: data for physics loss
    """
    # Sample from Gaussian Process prior
    N = 512
    gp_params = (1.0, length_scale)  # output_scale=1.0, length_scale=0.2
    jitter = 1e-10
    X = np.linspace(0, 1, N)[:,None]

    # RBF kernel: k(x1, x2) = exp(-||x1 - x2||²/(2l²))
    K = RBF(X, X, gp_params)
    # Cholesky decomposition for sampling
    L = np.linalg.cholesky(K + jitter*np.eye(N))
    gp_sample = np.dot(L, random.normal(key, (N,)))

    # Create interpolation function for u(x)
    u_fn = lambda x, t: np.interp(t, X.flatten(), gp_sample)

    # Evaluate u at input sensor locations
    x = np.linspace(0, 1, m)
    u = vmap(u_fn, in_axes=(None,0))(0.0, x)

    # Solve ODE ds/dx = u using RK45 integrator
    y_train = random.uniform(key, (P,)).sort()
    # JAX odeint solves: ds/dt = u(t) with s(0) = 0
    s_train = odeint(u_fn, 0.0, np.hstack((0.0, y_train)))[1:]

    # Tile input for vectorization (repeat u for each output point)
    u_train = np.tile(u, (P, 1))

    # Data for physics loss: enforce ds/dx = u at collocation points
    u_r_train = np.tile(u, (m, 1))
    y_r_train = x  # collocation points = input sensors
    s_r_train = u  # target: ds/dx should equal u

    return u_train, y_train, s_train, u_r_train, y_r_train, s_r_train
```

### Training Script

```python
# Problem setup
N_train = 10000  # Number of random input functions
m = 100          # Number of input sensors (points where u is evaluated)
P_train = 1      # Number of output sensors per function
length_scale = 0.2  # Gaussian process length scale

# Network architecture
# Branch: processes input function u of length m
# Trunk: processes query location y (1D)
branch_layers = [m, 50, 50, 50, 50, 50]  # 5 hidden layers with 50 neurons
trunk_layers = [1, 50, 50, 50, 50, 50]   # 5 hidden layers with 50 neurons

# Initialize model
model = PI_DeepONet(branch_layers, trunk_layers)

# Generate training data
# For operator loss: enforce s(0) = 0
# For physics loss: enforce ds/dx = u at collocation points
keys = random.split(random.PRNGKey(0), N_train)
gen_fn = jit(lambda key: generate_one_training_data(key, m, P_train))
u_train, y_train, s_train, u_r_train, y_r_train, s_r_train = vmap(gen_fn)(keys)

# Reshape data
u_train = np.float32(u_train.reshape(N_train * P_train, -1))
y_train = np.float32(y_train.reshape(N_train * P_train, -1))
s_train = np.float32(s_train.reshape(N_train * P_train, -1))

u_r_train = np.float32(u_r_train.reshape(N_train * m, -1))
y_r_train = np.float32(y_r_train.reshape(N_train * m, -1))
s_r_train = np.float32(s_r_train.reshape(N_train * m, -1))

# Create data generators with batching
batch_size = 10000
operator_dataset = DataGenerator(u_train, y_train, s_train, batch_size)
physics_dataset = DataGenerator(u_r_train, y_r_train, s_r_train, batch_size)

# Train the model
# 40,000 iterations with learning rate decay
model.train(operator_dataset, physics_dataset, nIter=40000)

# Get trained parameters
params = model.get_params(model.opt_state)

# Make predictions
s_pred = model.predict_s(params, u_test, y_test)
s_y_pred = model.predict_s_y(params, u_test, y_test)  # ds/dy should equal u

# Compute errors
error_s = np.linalg.norm(s_test - s_pred) / np.linalg.norm(s_test)
error_u = np.linalg.norm(u_test[::P_test].flatten()[:,None] - s_y_pred) / \
          np.linalg.norm(u_test[::P_test].flatten()[:,None])

print(f"Relative L2 error for s: {error_s:.3e}")
print(f"Relative L2 error for u: {error_u:.3e}")
# Typical output: error_s ~ 3e-3, error_u ~ 4e-3
```

## Critical Parameters

1. **Network Architecture**
   - Branch layers: [100, 50, 50, 50, 50, 50] (5 hidden layers, 50 neurons each)
   - Trunk layers: [1, 50, 50, 50, 50, 50] (5 hidden layers, 50 neurons each)
   - Activation: tanh (smooth, differentiable, essential for computing derivatives)
   - Note: Shallower networks work better for vanilla DeepONet vs deeper networks

2. **Input/Output Sensors**
   - m = 100: Number of input sensors (points where u(x) is evaluated)
   - P_train = 1: Number of output query points per training sample
   - P_test = 100: Number of output points for testing
   - Input sensors uniformly spaced in [0, 1]

3. **Training Data**
   - N_train = 10,000: Number of random input functions
   - Sampled from Gaussian Process with RBF kernel
   - Length scale = 0.2: controls smoothness of random functions
   - No paired input-output data needed (only initial condition s(0)=0)

4. **Optimizer Settings**
   - Optimizer: Adam
   - Initial learning rate: 1e-3
   - Learning rate decay: exponential, rate 0.9 every 1000 iterations
   - Batch size: 10,000
   - Training iterations: 40,000
   - Key: Learning rate decay crucial for convergence

5. **Loss Components**
   - L_operator: enforces initial condition s(0) = 0
   - L_physics: enforces ODE residual ds/dx - u(x) = 0
   - Both losses weighted equally (coefficient = 1.0)
   - MSE (mean squared error) used for both components

6. **Gaussian Process Parameters**
   - Output scale: 1.0
   - Length scale: 0.2 (smaller = more irregular functions)
   - Kernel: RBF (exponential quadratic)
   - Fine grid size: N = 512 points for GP sampling
   - Jitter: 1e-10 for numerical stability in Cholesky decomposition

7. **Framework**
   - JAX: used for automatic differentiation and JIT compilation
   - Enables efficient gradient computation for physics loss
   - JIT compilation speeds up training significantly
   - Double precision (jax.config.update("jax_enable_x64", True))

8. **Performance Metrics**
   - Relative L2 error for s(x): ~3e-3
   - Relative L2 error for u(x) = ds/dx: ~4e-3
   - Inference time: milliseconds for single function
   - Can solve O(10³) ODEs in fraction of a second
