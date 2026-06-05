# DeepONet for Antiderivative Operator Learning

**Keywords**: [integral-equation, forward-problem, antiderivative, 1D, operator, DeepONet, Gaussian-random-field, MLP, adam, mse, jax]

**Problem:** Learning nonlinear operators that map between function spaces, specifically the antiderivative operator that maps an input function u(x) to its integral s(x). Traditional neural networks approximate functions (mapping vectors to vectors), but many scientific problems require approximating operators (mapping functions to functions). The challenge is to efficiently learn such operators from data without requiring the input functions to be on a structured grid.

**Issues addressed:**
- Inability of standard neural networks to learn operators (function-to-function mappings)
- Need for flexible operator learning without requiring structured/gridded input data
- Efficient generalization from relatively small datasets of function pairs
- Learning solution operators for parametric differential equations

## Key Method

DeepONet is based on the **Universal Approximation Theorem for Operators**, which states that any nonlinear continuous operator G can be approximated by:

$$G(u)(y) \approx \sum_{k=1}^{p} b_k(u(x_1), u(x_2), \ldots, u(x_m)) \cdot t_k(y)$$

**Architecture:**
- **Branch Network**: Encodes the input function u at fixed sensor locations x₁, x₂, ..., xₘ
  - Input: [u(x₁), u(x₂), ..., u(xₘ)] ∈ ℝᵐ
  - Output: [b₁, b₂, ..., bₚ] ∈ ℝᵖ

- **Trunk Network**: Encodes the query location y where the output function is evaluated
  - Input: y ∈ ℝᵈ
  - Output: [t₁, t₂, ..., tₚ] ∈ ℝᵖ

- **Output**: G(u)(y) = Σ bₖ · tₖ (dot product of branch and trunk outputs)

**Problem Setup:**
For the antiderivative operator:
$$\frac{ds(x)}{dx} = u(x), \quad x \in [0,1], \quad s(0) = 0$$

The operator maps:
$$G: u(x) \mapsto s(x) = \int_0^x u(t) dt$$

**Training Approach:**
- Generate N random input functions u from a Gaussian Random Field (GRF) with RBF kernel
- For each function, evaluate at m sensor locations and P output locations
- Minimize MSE loss: $\mathcal{L} = \frac{1}{NP}\sum_{i=1}^N\sum_{j=1}^P|G_\theta(u^{(i)})(y_j^{(i)}) - s(y_j^{(i)})|^2$

**Key Advantages:**
- Input functions only need to be sampled at consistent sensor locations (not on structured grids)
- Output can be queried at arbitrary locations
- Generalizes well with small datasets due to inductive bias of separating function encoding (branch) from location encoding (trunk)
- Can learn derivatives through automatic differentiation: u(x) = ds/dx

## Implementation

### RBF Kernel for Gaussian Random Field

```python
# Define RBF (Radial Basis Function) kernel for generating random functions
def RBF(x1, x2, params):
    """
    Exponential quadratic kernel for Gaussian process
    Args:
        x1, x2: input points
        params: (output_scale, lengthscales)
    Returns:
        Covariance matrix K(x1, x2)
    """
    output_scale, lengthscales = params
    # Compute squared distances scaled by lengthscale
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    # Return RBF kernel: k(x1,x2) = σ² exp(-r²/2)
    return output_scale * np.exp(-0.5 * r2)
```

### Data Generation from Gaussian Random Field

```python
def generate_one_training_data(key, m=100, P=1):
    """
    Generate one training sample: (u, y, s) where s is the antiderivative of u
    Args:
        key: random key for reproducibility
        m: number of input sensors (points where u is evaluated)
        P: number of output sensors (points where s is evaluated)
    Returns:
        u_train: input function values at m sensors, shape (P, m)
        y_train: output query locations, shape (P,)
        s_train: output function values (integrals), shape (P,)
    """
    # Sample GP prior at a fine grid (N=512 points)
    N = 512
    gp_params = (1.0, length_scale)  # (output_scale, lengthscale)
    jitter = 1e-10
    X = np.linspace(0, 1, N)[:,None]

    # Generate covariance matrix K and its Cholesky decomposition
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter*np.eye(N))

    # Sample from GP: u ~ N(0, K)
    gp_sample = np.dot(L, random.normal(key, (N,)))

    # Create interpolation function for the sampled GP
    u_fn = lambda x, t: np.interp(t, X.flatten(), gp_sample)

    # Input sensors: evaluate u at m uniformly spaced points
    x = np.linspace(0, 1, m)
    u = vmap(u_fn, in_axes=(None,0))(0.0, x)

    # Output sensors: evaluate s at P random points
    y_train = random.uniform(key, (P,)).sort()

    # Solve ODE ds/dx = u(x) with s(0) = 0 using RK45
    # This gives s(x) = integral from 0 to x of u(t)dt
    s_train = odeint(u_fn, 0.0, np.hstack((0.0, y_train)))[1:]

    # Tile input to match output dimensions (repeat u for each output point)
    u_train = np.tile(u, (P,1))

    return u_train, y_train, s_train
```

### MLP Architecture

```python
def MLP(layers, activation=relu):
    """
    Multi-layer perceptron with Xavier initialization
    Args:
        layers: list of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
        activation: activation function (default: ReLU)
    Returns:
        init: function to initialize parameters
        apply: function to apply the network
    """
    def init(rng_key):
        """Initialize network parameters with Glorot/Xavier initialization"""
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            # Xavier initialization: stddev = sqrt(2/(d_in + d_out))
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params

    def apply(params, inputs):
        """Forward pass through the network"""
        # Hidden layers with activation
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = activation(outputs)
        # Output layer (no activation)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs

    return init, apply
```

### DeepONet Model

```python
class DeepONet:
    def __init__(self, branch_layers, trunk_layers):
        """
        Initialize DeepONet with branch and trunk networks
        Args:
            branch_layers: architecture for branch net, e.g., [100, 100, 100]
            trunk_layers: architecture for trunk net, e.g., [1, 100, 100]
        """
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=relu)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=relu)

        # Initialize parameters for both networks
        branch_params = self.branch_init(rng_key = random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
        params = (branch_params, trunk_params)

        # Adam optimizer with exponential learning rate decay
        # Initial lr=1e-3, decays by 0.9 every 1000 steps
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3,
                                                                      decay_steps=1000,
                                                                      decay_rate=0.9))
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        self.loss_log = []

    def operator_net(self, params, u, y):
        """
        Evaluate the DeepONet operator: G(u)(y) = sum(B_k * T_k)
        Args:
            params: (branch_params, trunk_params)
            u: input function values at sensors, shape (m,)
            y: query location, shape (1,) or scalar
        Returns:
            Output value G(u)(y)
        """
        branch_params, trunk_params = params
        # Branch network encodes input function
        B = self.branch_apply(branch_params, u)
        # Trunk network encodes query location
        T = self.trunk_apply(trunk_params, y)
        # Combine via dot product
        outputs = np.sum(B * T)
        return outputs

    def residual_net(self, params, u, y):
        """
        Compute derivative ds/dy = u using automatic differentiation
        This allows learning the PDE residual ds/dy - u = 0
        """
        s_y = grad(self.operator_net, argnums=2)(params, u, y)
        return s_y

    def loss(self, params, batch):
        """
        Mean squared error loss between predicted and true outputs
        Args:
            params: network parameters
            batch: (inputs, outputs) where inputs = (u, y)
        Returns:
            MSE loss
        """
        inputs, outputs = batch
        u, y = inputs  # u: (N, m), y: (N, 1)

        # Vectorized forward pass over batch
        pred = vmap(self.operator_net, (None, 0, 0))(params, u, y)

        # Compute MSE
        loss = np.mean((outputs.flatten() - pred)**2)
        return loss

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        """
        Single optimization step (JIT compiled for speed)
        Args:
            i: iteration number
            opt_state: current optimizer state
            batch: training batch
        Returns:
            Updated optimizer state
        """
        params = self.get_params(opt_state)
        # Compute gradients via automatic differentiation
        g = grad(self.loss)(params, batch)
        # Update parameters using Adam
        return self.opt_update(i, g, opt_state)

    def train(self, dataset, nIter=10000):
        """
        Training loop
        Args:
            dataset: DataGenerator object
            nIter: number of training iterations
        """
        data = iter(dataset)
        pbar = trange(nIter)

        for it in pbar:
            batch = next(data)
            # Perform optimization step
            self.opt_state = self.step(next(self.itercount), self.opt_state, batch)

            # Log loss every 100 iterations
            if it % 100 == 0:
                params = self.get_params(self.opt_state)
                loss_value = self.loss(params, batch)
                self.loss_log.append(loss_value)
                pbar.set_postfix({'Loss': loss_value})

    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        """
        Predict output function values s at query points
        Args:
            params: network parameters
            U_star: input functions, shape (N, m)
            Y_star: query locations, shape (N, 1)
        Returns:
            Predicted s values, shape (N,)
        """
        s_pred = vmap(self.operator_net, (None, 0, 0))(params, U_star, Y_star)
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_s_y(self, params, U_star, Y_star):
        """
        Predict derivative ds/dy (which should equal u)
        Uses automatic differentiation to compute gradient
        """
        s_y_pred = vmap(self.residual_net, (None, 0, 0))(params, U_star, Y_star)
        return s_y_pred
```

### Training Script

```python
# Configuration
N_train = 10000  # Number of training functions
m = 100          # Number of input sensors per function
P_train = 1      # Number of output sensors per function
length_scale = 0.2  # Length scale for RBF kernel (controls smoothness)

# Generate training data
key_train = random.PRNGKey(0)
keys = random.split(key_train, N_train)
gen_fn = jit(lambda key: generate_one_training_data(key, m, P_train))
u_train, y_train, s_train = vmap(gen_fn)(keys)

# Reshape data
u_train = np.float32(u_train.reshape(N_train * P_train, -1))
y_train = np.float32(y_train.reshape(N_train * P_train, -1))
s_train = np.float32(s_train.reshape(N_train * P_train, -1))

# Initialize model
# Shallower networks work better for vanilla DeepONet
branch_layers = [100, 100, 100]  # Input: 100 sensors
trunk_layers = [1, 100, 100]     # Input: 1D location

model = DeepONet(branch_layers, trunk_layers)

# Create data loader
batch_size = 10000
dataset = DataGenerator(u_train, y_train, s_train, batch_size)

# Train the model
model.train(dataset, nIter=40000)

# Predict on test data
params = model.get_params(model.opt_state)
s_pred = model.predict_s(params, u_test, y_test)
s_y_pred = model.predict_s_y(params, u_test, y_test)  # ds/dy = u

# Compute relative L2 error
error_s = np.linalg.norm(s_test - s_pred) / np.linalg.norm(s_test)
error_u = np.linalg.norm(u_test[::P_test].flatten()[:,None] - s_y_pred) / \
          np.linalg.norm(u_test[::P_test].flatten()[:,None])
```

## Critical Parameters

1. **Number of input sensors (m)**
   - Value: 100
   - Purpose: Number of points where input function u(x) is evaluated
   - Effect: More sensors capture input function better but increase computational cost
   - For GRF with length scale l: m ∝ 1/l

2. **Number of training functions (N)**
   - Value: 10,000
   - Purpose: Number of different input functions in training set
   - Effect: More functions improve generalization; DeepONet shows exponential convergence with small datasets

3. **Number of output sensors per function (P)**
   - Value: 1 (training), 100 (testing)
   - Purpose: Number of points where output s(x) is evaluated per input function
   - Training strategy: Use P=1 during training for efficiency, evaluate densely during testing

4. **Branch network architecture**
   - Layers: [100, 100, 100]
   - Input dimension: m = 100 (sensor values)
   - Output dimension: 100 (basis functions)
   - Shallower networks (2-3 layers) work better than deep networks for vanilla DeepONet

5. **Trunk network architecture**
   - Layers: [1, 100, 100]
   - Input dimension: 1 (query location y)
   - Output dimension: 100 (basis functions matching branch output)

6. **Activation function**
   - Function: ReLU
   - Applied to all hidden layers except output layer

7. **Optimizer**
   - Type: Adam with exponential learning rate decay
   - Initial learning rate: 1e-3
   - Decay rate: 0.9 every 1000 steps
   - Prevents oscillations and improves convergence

8. **Batch size**
   - Value: 10,000
   - Can use full batch since N*P = 10,000 total training points

9. **Training iterations**
   - Value: 40,000
   - Typical convergence achieved before this limit
   - Final loss: ~5e-7 (MSE)

10. **Gaussian Random Field parameters**
    - Kernel: RBF (exponential quadratic)
    - Length scale (l): 0.2
    - Controls smoothness of sampled functions
    - Smaller l → less smooth functions → requires more sensors
    - Output scale: 1.0

11. **Weight initialization**
    - Method: Xavier/Glorot initialization
    - Standard deviation: 1/√((d_in + d_out)/2)
    - Ensures stable gradient flow during training

12. **Relative L2 errors (typical)**
    - s(x) prediction: ~0.002 (0.2%)
    - u(x) = ds/dx prediction: ~0.09 (9%)
    - Derivative prediction is inherently more difficult due to amplification of errors
