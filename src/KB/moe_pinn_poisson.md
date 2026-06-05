# Mixture-of-Experts Physics-Informed Neural Networks (MoE-PINNs) for Poisson Equation on L-shaped Domain

**Keywords**: [PDE, elliptic, linear, forward-problem, poisson, 2D, irregular, dirichlet, PINN, MLP, ensemble-training, strong-form, adam, mse, tensorflow]

**Problem:** Physics-Informed Neural Networks (PINNs) face significant challenges when solving PDEs on irregular geometries like L-shaped domains. Single PINNs struggle to accurately capture solutions in regions with re-entrant corners and varying solution characteristics across the domain. The L-shaped domain presents a particular challenge due to its non-convex geometry where the solution behavior differs significantly between different quadrants.

**Issues addressed:**
- Poor accuracy on irregular geometries with re-entrant corners
- Difficulty capturing spatially varying solution characteristics in single network
- Manual domain decomposition requiring careful boundary definition at partition interfaces
- Hyperparameter sensitivity in selecting optimal number of subdomains
- Training instability on complex geometric domains

## Key Method

**Mixture-of-Experts Physics-Informed Neural Networks (MoE-PINNs)** applies ensemble learning to PINNs by training multiple expert networks with an automated gating mechanism that learns to partition the irregular domain. Unlike manual domain decomposition methods (cPINNs, XPINNs), MoE-PINNs automatically discover optimal domain partitioning during training.

**Core components:**

1. **Multiple PINN Experts**: Each expert network independently solves the PDE across the entire domain but specializes on specific regions through the gating mechanism.

2. **Gating Network**: Learns to assign importance weights λᵢ(x,y) to each expert based on spatial location:
   ```
   λᵢ(x,y) = exp(P(i|x,y,θᵢ)) / Σⱼ exp(P(j|x,y,θⱼ))
   ```

3. **Weighted Ensemble Solution**:
   ```
   u(x,y) = Σᵢ λᵢ(x,y) · uᵢ(x,y)
   ```

4. **Automated Architecture Search**: By initializing diverse expert architectures (varying depth, width, activation functions) with sparsity regularization, the gating network automatically selects the most suitable subset during training.

5. **ReLoBRaLo Loss Balancing**: Adaptively weights PDE residual and boundary condition losses to ensure balanced training.

**Key findings for L-shaped Poisson:**
- 3 experts optimal: Gating network divides L-domain into three quadrants naturally
- Sine activation outperforms tanh in ensemble (despite tanh being most common)
- Sparsity regularization p=0.5 works better than p=2 (aggressive pruning preferred)
- Replacing ReLU with tanh in gate provides largest performance improvement

## Implementation

### MoE-PINN Model Architecture

```python
class MoEPINN(tf.keras.Model):
    """
    MoEPINN for Poisson equation on irregular L-shaped domain
    Automatically learns domain decomposition through gating network
    """
    def __init__(self,
                 n_models: int,          # Number of expert PINNs
                 n_layers: list,         # Layer count for each expert
                 n_nodes: list,          # Node count for each expert
                 activations: list,      # Activation function for each expert
                 gate_n_layers: int,     # Gating network depth
                 gate_n_nodes: int,      # Gating network width
                 gate_activation: Union[str, Callable],  # Gate activation
                 softmax_temp: float = 1,  # Temperature for softmax
                 name: str = 'moepinn',
                 **kwargs):
        super().__init__(name, **kwargs)
        self.n_models = n_models
        self.softmax_temp = softmax_temp

        # Create ensemble of PINN experts
        # Each expert can have different architecture
        self.pinns = [
            FCNN(1, n_layers[i], n_nodes[i], self.activations[i], name='pinn_'+str(i))
            for i in range(n_models)
        ]

        # Create gating network for learning importance weights
        if self.n_models > 1:
            self.gate = FCNN(
                n_models, gate_n_layers, gate_n_nodes, gate_activation,
                out_activation=lambda x, y, u: u,  # No derivatives for gate output
                name='gating_network'
            )

    def call(self, inputs):
        """
        Forward pass through MoE-PINN
        inputs: (x, y) coordinates, shape (batch_size, 2)
        Returns: weighted ensemble prediction
        """
        # Obtain predictions from all experts
        # Each u[i] has shape (batch_size, 3): [u, dudxx, dudyy]
        u = [pinn(inputs) for pinn in self.pinns]

        if self.n_models > 1:
            # Compute importance weights from gating network
            logits = self.gate(inputs)  # (batch_size, n_models)
            weights = tf.nn.softmax(logits / self.softmax_temp, axis=-1)

            # Weighted combination of expert predictions
            weighted_u = tf.reduce_sum(
                [u[i] * weights[:, i:i+1] for i in range(self.n_models)],
                axis=0
            )
            return weighted_u
        else:
            return u[0]

    def get_importances(self, inputs):
        """
        Get importance distribution across experts for visualization
        Returns: (batch_size, n_models) importance weights
        """
        if self.n_models == 1:
            return tf.ones((tf.shape(inputs)[0], 1))

        logits = self.gate(inputs)
        return tf.nn.softmax(logits / self.softmax_temp, axis=-1)
```

### PINN Expert Network with Residual Connections

```python
class FCNN(tf.keras.Model):
    """
    Fully-connected neural network for PINN experts
    Includes residual connections for better gradient flow
    """
    def __init__(self,
                 n_out_vars: int = 1,
                 n_layers: int = 4,
                 n_nodes: int = 256,
                 activation: Union[str, Callable] = 'tanh',
                 out_activation: Callable = lambda x, y, u: tf.concat(
                     [u] + list(compute_derivatives(x, y, u)), axis=-1
                 ),
                 **kwargs):
        super().__init__(**kwargs)

        # Hidden layers (all same width for residual connections)
        self.layer_sequence = [
            tf.keras.layers.Dense(
                n_nodes,
                activation=activation,
                kernel_initializer='glorot_normal'  # Xavier initialization
            )
            for _ in range(n_layers)
        ]

        # Output layer (linear)
        self.dense_out = tf.keras.layers.Dense(
            n_out_vars,
            use_bias=False,
            kernel_initializer='glorot_normal'
        )

        self.out_activation = out_activation

    def call(self, xy):
        """
        Forward pass with residual connections
        xy: concatenated (x, y) coordinates
        """
        x, y = xy[:, :1], xy[:, 1:]

        # Input layer
        u = tf.concat([x, y], axis=-1)

        # First hidden layer
        u = self.layer_sequence[0](u)

        # Subsequent layers with residual connections
        # h_{l+1} = σ(W_l h_l + b_l) + h_l
        for layer in self.layer_sequence[1:]:
            u = layer(u) + u

        # Output layer
        u = self.dense_out(u)

        # Compute derivatives via automatic differentiation
        return self.out_activation(x, y, u)
```

### Computing Second-Order Derivatives for Poisson Equation

```python
def compute_derivatives(x, y, u):
    """
    Computes second-order derivatives for Poisson equation via autodiff
    Poisson equation: -Δu = -(u_xx + u_yy) = 1

    Parameters
    ----------
    x : tf.Tensor, shape (batch_size, 1)
        x-coordinate
    y : tf.Tensor, shape (batch_size, 1)
        y-coordinate
    u : tf.Tensor, shape (batch_size, 1)
        Network prediction u(x,y)

    Returns
    -------
    tuple of (dudxx, dudyy)
        Second-order partial derivatives
    """
    # First order derivatives
    dudx, dudy = tf.gradients(u, [x, y])

    # Second order derivatives (Laplacian components)
    dudxx = tf.gradients(dudx, x)[0]  # ∂²u/∂x²
    dudyy = tf.gradients(dudy, y)[0]  # ∂²u/∂y²

    return dudxx, dudyy
```

### Poisson PDE on L-shaped Domain

```python
class PoissonPDE:
    """
    Poisson equation: -Δu(x,y) = 1
    Domain: Ω = [-1,1]² \ [0,1]²  (L-shaped)
    Boundary conditions: u(x,y) = 0 on Γ (Dirichlet)
    """
    def __init__(self):
        self.num_terms = 2  # PDE residual + boundary conditions

        # Load validation data computed with Spectral Element Method (SEM)
        data = np.load('Poisson_Lshape.npz', allow_pickle=True)

        # Filter out NaN values (points outside L-shaped domain)
        self.val_u = data['y_ref'][::4]
        valid_mask = np.isnan(self.val_u[:, 0]) != True

        self.val_x = tf.cast(
            data['X_test'][::4, :1][valid_mask],
            dtype=tf.float32
        )
        self.val_y = tf.cast(
            data['X_test'][::4, 1:][valid_mask],
            dtype=tf.float32
        )
        self.val_u = tf.cast(
            self.val_u[valid_mask],
            dtype=tf.float32
        )

    def training_batch(self, batch_size_domain: int = 4000,
                      batch_size_boundary: int = 1000):
        """
        Generate collocation points for L-shaped domain

        The L-shape consists of two rectangles:
        - Vertical rectangle: x ∈ [-1, 0], y ∈ [-1, 1]
        - Horizontal rectangle: x ∈ [-1, 1], y ∈ [-1, 0]

        Parameters
        ----------
        batch_size_domain : int
            Number of interior collocation points
        batch_size_boundary : int
            Number of boundary points per boundary segment
        """
        self.full_batch_size = batch_size_domain + 4 * batch_size_boundary

        # Sample interior points for L-shaped domain
        # Small square (removed part): [0,1] × [0,1]
        internal1 = np.random.uniform(
            low=[-0.1, -1.1],
            high=[1.1, 0.1],
            size=(batch_size_domain//3, 2)
        )

        # Large rectangle (main part)
        internal2 = np.random.uniform(
            low=[-1.1, -1.1],
            high=[0.1, 1.1],
            size=(batch_size_domain - batch_size_domain//3, 2)
        )

        # Sample boundary points
        # Boundary 1: Left edge (x = -1)
        BCx1 = -np.ones((batch_size_boundary, 1))
        BCy1 = np.random.uniform(-1, 1, (batch_size_boundary, 1))

        # Boundary 2: Bottom edge (y = -1)
        BCx2 = np.random.uniform(-1, 1, (batch_size_boundary, 1))
        BCy2 = -np.ones((batch_size_boundary, 1))

        # Boundary 3: Right edges (x = 1 for y < 0, x = 0 for y > 0)
        BCx3 = np.concatenate([
            np.ones((batch_size_boundary//2, 1)),      # x = 1
            np.zeros((batch_size_boundary//2, 1))      # x = 0
        ], axis=0)
        BCy3 = np.concatenate([
            np.random.uniform(-1, 0, (batch_size_boundary//2, 1)),  # y ∈ [-1, 0]
            np.random.uniform(0, 1, (batch_size_boundary//2, 1))    # y ∈ [0, 1]
        ], axis=0)

        # Boundary 4: Top edges (y = 1 for x < 0, y = 0 for x > 0)
        BCx4 = np.concatenate([
            np.random.uniform(-1, 0, (batch_size_boundary//2, 1)),  # x ∈ [-1, 0]
            np.random.uniform(0, 1, (batch_size_boundary//2, 1))    # x ∈ [0, 1]
        ], axis=0)
        BCy4 = np.concatenate([
            np.ones((batch_size_boundary//2, 1)),      # y = 1
            np.zeros((batch_size_boundary//2, 1))      # y = 0
        ], axis=0)

        # Concatenate all points
        x = tf.constant(
            np.concatenate([internal1[:,:1], internal2[:,:1],
                          BCx1, BCx2, BCx3, BCx4], axis=0),
            dtype=tf.float32
        )
        y = tf.constant(
            np.concatenate([internal1[:,1:], internal2[:,1:],
                          BCy1, BCy2, BCy3, BCy4], axis=0),
            dtype=tf.float32
        )

        return x, y

    def compute_loss(self, x, y, preds, eval=False):
        """
        Compute physics-informed loss for Poisson equation

        Parameters
        ----------
        x : tf.Tensor, shape (batch_size, 1)
            x-coordinates
        y : tf.Tensor, shape (batch_size, 1)
            y-coordinates
        preds : tf.Tensor, shape (batch_size, 3)
            Predictions: [u, dudxx, dudyy]
        eval : bool
            If True, also compute validation error

        Returns
        -------
        tuple of (L_f, L_bc) or (L_f, L_bc, L_u)
        """
        # Extract predictions
        u = preds[:, 0:1]
        dudxx = preds[:, 1:2]
        dudyy = preds[:, 2:3]

        # Governing equation residual: -Δu - 1 = -(u_xx + u_yy) - 1 = 0
        # Equivalently: u_xx + u_yy + 1 = 0
        L_f = (dudxx + dudyy + 1)**2

        # Identify boundary points for L-shaped domain
        EPS = 1e-5
        xl = tf.cast(isclose(x, -1, rtol=0., atol=EPS), dtype=tf.float32)      # x = -1

        # Right boundary (two segments)
        xu0 = tf.cast(
            tf.math.logical_and(isclose(x, 0, rtol=0., atol=EPS), y >= 0),
            dtype=tf.float32
        )  # x = 0, y ≥ 0
        xu1 = tf.cast(
            tf.math.logical_and(isclose(x, 1, rtol=0., atol=EPS), y <= 0),
            dtype=tf.float32
        )  # x = 1, y ≤ 0

        yl = tf.cast(isclose(y, -1, rtol=0., atol=EPS), dtype=tf.float32)      # y = -1

        # Top boundary (two segments)
        yu0 = tf.cast(
            tf.math.logical_and(isclose(y, 1, rtol=0., atol=EPS), x <= 0),
            dtype=tf.float32
        )  # y = 1, x ≤ 0
        yu1 = tf.cast(
            tf.math.logical_and(isclose(y, 0, rtol=0., atol=EPS), x >= 0),
            dtype=tf.float32
        )  # y = 0, x ≥ 0

        # Boundary condition loss: u = 0 on all boundaries
        boundary_indicator = xl + xu0 + xu1 + yl + yu0 + yu1
        L_bc = (boundary_indicator * u)**2

        if eval:
            # Validation error against SEM reference
            L_u = (self.val_u - u)**2
            return L_f, L_bc, L_u

        return L_f, L_bc
```

### ReLoBRaLo Adaptive Loss Balancing

```python
class ReLoBRaLoLoss(tf.keras.losses.Loss):
    """
    ReLoBRaLo: Adaptive multi-objective loss balancing for PINNs
    Dynamically adjusts weights between PDE residual and boundary condition losses
    """
    def __init__(self,
                 pde: PoissonPDE,
                 alpha: float = 0.999,       # EMA decay rate
                 temperature: float = 0.5,   # Softmax temperature
                 rho: float = 0.999,         # Random lookback probability
                 name='ReLoBRaLoLoss'):
        super().__init__(name=name)
        self.pde = pde
        self.alpha = alpha
        self.temperature = temperature
        self.rho = rho
        self.call_count = tf.Variable(0, trainable=False, dtype=tf.int16)

        # Initialize adaptive weights
        self.lambdas = [
            tf.Variable(1., trainable=False)
            for _ in range(pde.num_terms)
        ]
        self.last_losses = [
            tf.Variable(1., trainable=False)
            for _ in range(pde.num_terms)
        ]
        self.init_losses = [
            tf.Variable(1., trainable=False)
            for _ in range(pde.num_terms)
        ]

    def call(self, xy, preds):
        """
        Compute adaptively weighted loss
        """
        x, y = xy[:, :1], xy[:, 1:]
        losses = [
            tf.reduce_mean(loss)
            for loss in self.pde.compute_loss(x, y, preds)
        ]

        # Special cases for first two iterations
        alpha = tf.cond(
            tf.equal(self.call_count, 0),
            lambda: 1.,  # Iteration 0: use initial lambdas
            lambda: tf.cond(
                tf.equal(self.call_count, 1),
                lambda: 0.,  # Iteration 1: use only lambda_hat
                lambda: self.alpha  # Otherwise: default behavior
            )
        )

        rho = tf.cond(
            tf.equal(self.call_count, 0),
            lambda: 1.,
            lambda: tf.cond(
                tf.equal(self.call_count, 1),
                lambda: 1.,
                lambda: tf.cast(
                    tf.random.uniform(shape=()) < self.rho,
                    dtype=tf.float32
                )
            )
        )

        # Compute lambdas based on relative losses (current vs previous)
        EPS = 1e-5
        lambdas_hat = [
            losses[i] / (self.last_losses[i] * self.temperature + EPS)
            for i in range(len(losses))
        ]
        lambdas_hat = tf.nn.softmax(
            lambdas_hat - tf.reduce_max(lambdas_hat)
        ) * tf.cast(len(losses), dtype=tf.float32)

        # Compute lambdas based on relative losses (current vs initial)
        init_lambdas_hat = [
            losses[i] / (self.init_losses[i] * self.temperature + EPS)
            for i in range(len(losses))
        ]
        init_lambdas_hat = tf.nn.softmax(
            init_lambdas_hat - tf.reduce_max(init_lambdas_hat)
        ) * tf.cast(len(losses), dtype=tf.float32)

        # Combine with random lookback
        new_lambdas = [
            (rho * alpha * self.lambdas[i] +
             (1 - rho) * alpha * init_lambdas_hat[i] +
             (1 - alpha) * lambdas_hat[i])
            for i in range(len(losses))
        ]

        # Update lambdas (stop gradients)
        self.lambdas = [
            var.assign(tf.stop_gradient(lam))
            for var, lam in zip(self.lambdas, new_lambdas)
        ]

        # Compute weighted loss
        loss = tf.reduce_sum([
            lam * loss for lam, loss in zip(self.lambdas, losses)
        ])

        # Update loss history
        self.last_losses = [
            var.assign(tf.stop_gradient(loss))
            for var, loss in zip(self.last_losses, losses)
        ]

        # Store initial losses on first call
        first_iteration = tf.cast(self.call_count < 1, dtype=tf.float32)
        self.init_losses = [
            var.assign(tf.stop_gradient(
                loss * first_iteration + var * (1 - first_iteration)
            ))
            for var, loss in zip(self.init_losses, losses)
        ]

        self.call_count.assign_add(1)
        return loss
```

### Training Configuration

```python
# Initialize Poisson PDE on L-shaped domain
pde = PoissonPDE()

# Create ensemble of 3 experts (optimal for L-shaped domain)
moepinn = MoEPINN(
    n_models=3,
    n_layers=[2, 2, 2],                  # All experts: 2 hidden layers
    n_nodes=[128, 128, 128],             # All experts: 128 nodes per layer
    activations=['tanh', 'tanh', 'tanh'],  # All use tanh activation
    gate_n_layers=2,                      # Gating network: 2 layers
    gate_n_nodes=64,                      # Gating network: 64 nodes per layer
    gate_activation='tanh'                # Critical: tanh instead of ReLU
)

# Setup adaptive loss balancing
loss = ReLoBRaLoLoss(
    pde,
    temperature=0.1,   # Lower temperature for sharper balancing
    rho=0.99,          # High probability of random lookback
    alpha=0.999        # Strong EMA decay
)

# Compile model
moepinn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=loss,
    metrics=[PoissonMetric(pde), ReLoBRaLoLambdaMetric(loss)]
)

# Train with callbacks
h = moepinn.fit(
    pde.get_train_dataset(200, 100),  # 200 domain pts, 100 boundary pts per segment
    epochs=1000,
    steps_per_epoch=100,
    validation_data=pde.get_validation_dataset(),
    validation_steps=1,
    callbacks=[
        ReduceLROnPlateau(
            monitor='unscaled_loss',
            factor=0.1,        # Reduce LR by 10x
            patience=10,       # Wait 10 epochs
            min_delta=0,
            verbose=True
        ),
        EarlyStopping(
            monitor='unscaled_loss',
            patience=32,                    # Early stop after 32 epochs
            restore_best_weights=True,
            verbose=True
        )
    ]
)
```

## Critical Parameters

1. **Number of experts (n_models)**
   - Optimal: 3 experts for L-shaped Poisson
   - Natural decomposition: one expert per quadrant of L-domain
   - Paper finding: 4-5 experts increase error (sparsity regularization insufficient)
   - Recommendation: Start with 3 experts for L-shaped domains

2. **Expert network architecture**
   - Depth: 2-3 hidden layers (2 sufficient for Poisson)
   - Width: 128 nodes per layer
   - All experts can have same architecture for this problem
   - Simpler than Burgers (elliptic vs parabolic PDE)

3. **Activation functions**
   - tanh: Standard choice, works well for smooth Poisson solutions
   - sine: Paper found it performs excellently in ensemble
   - Diverse activations help but not critical for this problem
   - Gate activation: tanh is crucial (replacing ReLU gives largest improvement)

4. **Gating network configuration**
   - Architecture: 2 layers × 64 nodes (fixed)
   - Activation: tanh (infinitely differentiable)
   - Critical finding: Largest improvement from ReLU → tanh in gate
   - Smaller than experts (only learns partitioning, not PDE solution)

5. **Softmax temperature**
   - Default: 1.0
   - Controls sharpness of domain partitioning
   - Lower → sharper boundaries between expert regions
   - Higher → smoother blending at boundaries

6. **Sparsity regularization**
   - Order p: 0.25 or 0.5 (aggressive sparsity)
   - p = 0.5 > p = 2 (paper finding)
   - Drives weak experts to zero importance
   - Optional but helps with expert selection

7. **ReLoBRaLo parameters**
   - alpha: 0.999 (strong exponential moving average)
   - temperature: 0.1 (sharper loss balancing than Burgers)
   - rho: 0.99 (high random lookback probability)
   - Balances PDE residual vs boundary condition losses
   - More critical for problems with disparate loss scales

8. **Optimizer configuration**
   - Adam with initial LR: 0.01
   - Higher than Burgers ensemble (simpler problem)
   - Learning rate schedule: ×0.1 every 10 epochs without improvement
   - Early stopping: patience 32 epochs

9. **Training data sampling**
   - Interior domain points: 200-4000 per batch
   - Boundary points: 100-1000 per boundary segment
   - Total: 4 boundary segments for L-shaped domain
   - L-shape requires careful sampling to avoid missing corners

10. **L-shaped domain specifics**
    - Domain: Ω = [-1,1]² \ [0,1]² (non-convex)
    - Critical feature: Re-entrant corner at origin
    - Boundary segments: 6 edges (complex topology)
    - Solution characteristics vary significantly across three quadrants
    - Validation data: Spectral Element Method (SEM) reference solution
    - Grid resolution: 64×64 (interpolated for visualization)

11. **Comparison to single PINN**
    - Single PINN (3 layers, 128 nodes): MSE ≈ 8.32×10⁻⁵
    - MoE-PINN (3 experts, p=0.25): MSE ≈ 5.93×10⁻⁵
    - Improvement: ~30% error reduction
    - Single PINN (3 layers, 256 nodes): MSE ≈ 8.73×10⁻⁵ (more parameters worse)
    - MoE more effective than simply increasing single network size
