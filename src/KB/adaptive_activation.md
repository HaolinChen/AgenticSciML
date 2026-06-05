# Adaptive Activation Functions for Physics-Informed Neural Networks

**Keywords**: [PDE, nonlinear, forward-problem, burgers, 1D, PINN, MLP, self-adaptive, strong-form, adam, lbfgs, mse, tensorflow]

**Problem:** Physics-informed neural networks (PINNs) often suffer from slow convergence rates, especially during early training stages when solving nonlinear PDEs. The fixed activation functions (like tanh with slope=1) limit the network's ability to efficiently explore the loss landscape and capture different frequency components in the solution.

**Issues addressed:**
- Slow convergence in neural network training
- Difficulty capturing high-frequency components in PDE solutions
- Inefficient learning in early training stages
- Suboptimal performance when solving PDEs with steep gradients or discontinuities

## Key Method

The method introduces a **scalable adaptive hyperparameter** `a` into the activation function that is optimized alongside network weights and biases. The activation function is modified from σ(Lk(x)) to:

**σ(n·a·Lk(x))**

where:
- `a` is a trainable scalar parameter (adaptive slope)
- `n` is a fixed scaling factor (typically n ≥ 1)
- `Lk(x)` is the affine transformation at layer k

**Key advantages:**
- Dynamically changes the topology of the loss function during optimization
- Accelerates convergence, especially in early training
- Improves solution accuracy for both smooth and discontinuous functions
- Single additional parameter to train regardless of network depth

**Initialization:** `n·a = 1` to ensure stable initial conditions

## Implementation

### Core Network Class with Adaptive Activation

```python
class PhysicsInformedNN:
    def __init__(self, X_u, u, X_f, layers, lb, ub, beta, nu):
        """
        X_u: boundary/initial condition training points
        u: boundary/initial condition values
        X_f: collocation points for PDE residual
        layers: network architecture [input_dim, hidden1, hidden2, ..., output_dim]
        lb, ub: lower and upper bounds for input normalization
        beta: regularization parameter
        nu: viscosity coefficient for Burgers equation
        """
        self.lb = lb
        self.ub = ub
        self.x_u = X_u[:,0:1]
        self.t_u = X_u[:,1:2]
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        self.u = u
        self.layers = layers
        self.beta = beta
        self.nu = nu

        # Initialize network parameters including adaptive activation parameter
        self.weights, self.biases, self.a = self.initialize_NN(layers)

        # Setup TensorFlow session and placeholders
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        # Network predictions
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        # Loss function: data + PDE residual
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))

        # L-BFGS-B optimizer for fine-tuning
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 2000,
                                                                           'maxfun': 2000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        # Adam optimizer for initial training
        self.optimizer_Adam = tf.train.AdamOptimizer(0.0008)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)
```

### Initialize Network with Adaptive Parameter

```python
def initialize_NN(self, layers):
    """
    Initialize weights, biases using Xavier initialization
    and adaptive activation parameter 'a'
    """
    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(0, num_layers-1):
        W = self.xavier_init(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)

    # Initialize adaptive activation slope parameter
    # Starting value of 0.1 (will be scaled by n=10 to get na=1)
    a = tf.Variable(0.1, dtype=tf.float32)

    return weights, biases, a
```

### Neural Network Forward Pass with Adaptive Activation

```python
def neural_net(self, X, weights, biases, a):
    """
    Forward pass through network with adaptive activation function
    X: input features
    weights, biases: network parameters
    a: adaptive activation parameter
    """
    num_layers = len(weights) + 1

    # Normalize input to [-1, 1]
    H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0

    # Forward pass through hidden layers with adaptive tanh activation
    for l in range(0, num_layers-2):
        W = weights[l]
        b = biases[l]

        # Adaptive activation: tanh(n*a*(W*H + b))
        # Scaling factor n=10 ensures n*a=1 condition initially
        # As 'a' is optimized, the slope of tanh changes dynamically
        H = tf.tanh(10*a*tf.add(tf.matmul(H, W), b))

    # Output layer (no activation)
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y
```

### Physics-Informed Network for Burgers Equation

```python
def net_u(self, x, t):
    """Predict solution u(x,t)"""
    u = self.neural_net(tf.concat([x,t], 1), self.weights, self.biases, self.a)
    return u

def net_f(self, x, t):
    """
    Compute PDE residual for Burgers equation:
    u_t + u*u_x - nu*u_xx = 0
    """
    u = self.net_u(x, t)

    # Automatic differentiation for derivatives
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]

    # Burgers equation residual
    f = u_t + u*u_x - self.nu*u_xx

    return f
```

### Training Loop

```python
def train(self, nIter):
    """
    Two-stage training:
    1. Adam optimizer for nIter iterations
    2. L-BFGS-B for fine-tuning
    """
    tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
               self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

    MSE_history = []
    a_history = []

    # Stage 1: Adam optimization
    for it in range(nIter):
        self.sess.run(self.train_op_Adam, tf_dict)

        if it % 50 == 0:
            loss_value = self.sess.run(self.loss, tf_dict)
            a_value = self.sess.run(self.a, tf_dict)
            print('It: %d, Loss: %.3e, a_value: %.3e' %
                  (it, loss_value, a_value))
            MSE_history.append(loss_value)
            a_history.append(a_value)

    # Stage 2: L-BFGS-B fine-tuning
    self.optimizer.minimize(self.sess,
                            feed_dict = tf_dict,
                            fetches = [self.loss],
                            loss_callback = self.callback)

    return MSE_history, a_history
```

### Main Training Script

```python
if __name__ == "__main__":
    # Problem parameters
    beta = 1e-7  # Regularization
    noise = 0.0  # Noise level in data
    nu = 0.01/np.pi  # Viscosity coefficient

    # Training data sizes
    N_u = 200  # Boundary/initial points
    N_f = 10000  # Collocation points

    # Network architecture: 2 inputs (x,t) -> 6 hidden layers (20 neurons each) -> 1 output (u)
    layers = [2, 20, 20, 20, 20, 20, 20, 1]

    # Load Burgers equation data
    data = scipy.io.loadmat('burgers_shock.mat')
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]

    # Domain bounds for normalization
    lb = X_star.min(0)
    ub = X_star.max(0)

    # Prepare training data (initial + boundary conditions)
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))  # Initial condition
    uu1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))  # Left boundary
    uu2 = Exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))  # Right boundary
    uu3 = Exact[:,-1:]

    X_u_train = np.vstack([xx1, xx2, xx3])

    # Generate collocation points using Latin Hypercube Sampling
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))

    u_train = np.vstack([uu1, uu2, uu3])

    # Randomly select N_u training points
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]

    # Create and train model
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, beta, nu)

    Max_iter = 2000
    start_time = time.time()
    MSE_hist, a_hist = model.train(Max_iter)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    # Prediction and error calculation
    u_pred, f_pred = model.predict(X_star)
    error_u = np.linalg.norm(u_star-u_pred, 2) / np.linalg.norm(u_star, 2)
    print('Error u: %e' % (error_u))
```

## Critical Parameters

1. **Adaptive activation parameter (a)**
   - Initial value: 0.1 (combined with n=10 gives na=1 initially)
   - Trainable: Yes, optimized along with weights and biases
   - Typical final values: 0.15-0.2 (na ≈ 1.5-2.0)
   - Effect: Controls the slope of the activation function dynamically

2. **Scaling factor (n)**
   - Value: 10 (fixed)
   - Purpose: Accelerates the tuning of 'a' parameter
   - Condition: Should satisfy n·a ≈ 1 initially for stable training
   - Note: Large n (>10) can cause oscillations; n=5-10 recommended

3. **Learning rate**
   - Adam optimizer: 0.0008
   - Critical for stable optimization of 'a' parameter
   - Lower than typical values (0.001) to prevent divergence

4. **Network architecture**
   - Layers: [2, 20, 20, 20, 20, 20, 20, 1]
   - 6 hidden layers with 20 neurons each
   - Input: (x, t) coordinates
   - Output: u(x, t) solution

5. **Training data**
   - Boundary/initial points (N_u): 200
   - Collocation points (N_f): 10000
   - Sampling: Latin Hypercube Sampling for collocation points

6. **Optimizer sequence**
   - Stage 1: Adam for 2000 iterations (exploration)
   - Stage 2: L-BFGS-B for refinement (exploitation)
   - Two-stage approach combines fast convergence with high accuracy

7. **PDE-specific parameters** (Burgers equation)
   - Viscosity (nu): 0.01/π ≈ 0.00318
   - Domain: x ∈ [-1, 1], t ∈ [0, 1]
   - Boundary conditions: Dirichlet on all boundaries
