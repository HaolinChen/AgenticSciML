# Mixture-of-Experts Physics-Informed Neural Networks (MoE-PINNs) for Burgers Equation

**Keywords**: [PDE, parabolic, nonlinear, forward-problem, burgers, 1D, dirichlet, PINN, MLP, ensemble-training, strong-form, adam, mse, tensorflow]

**Problem:** Physics-Informed Neural Networks (PINNs) struggle with complex PDEs as computational domains or nonlinearity increase. Single PINNs have limited representation capacity and may suffer from training difficulties. The challenge is to improve PINN performance on nonlinear PDEs like Burgers equation by leveraging ensemble methods that allow different networks to specialize on different regions of the domain.

**Issues addressed:**
- Limited representation capacity of single PINNs for complex domains
- Training difficulties with nonlinear PDEs requiring multiple loss term balancing
- Poor performance on problems where different regions have different characteristics
- Suboptimal expert selection in ensemble methods (manual tuning of number of experts)

## Key Method

**Mixture-of-Experts Physics-Informed Neural Networks (MoE-PINNs)** combines multiple PINN experts with a learned gating network that automatically assigns importance weights to each expert based on the input location. The method allows experts to specialize on different subregions of the domain.

**Core components:**

1. **Multiple PINN Experts**: Each expert is a standard PINN (fully-connected neural network) that predicts the solution u(x,t) and its derivatives via automatic differentiation.

2. **Gating Network**: A separate neural network that takes (x,t) as input and outputs importance weights λᵢ for each expert via softmax:
   ```
   λᵢ(x,t) = exp(P(i|x,t,θᵢ)) / Σⱼ exp(P(j|x,t,θⱼ))
   ```

3. **Weighted Ensemble Prediction**:
   ```
   u(x,t) = Σᵢ λᵢ(x,t) · uᵢ(x,t)
   ```

4. **Sparsity Regularization** (optional): Encourages using fewer experts by penalizing small importance values more than large ones:
   ```
   L_sp = |1/|B| Σᵢ Σ_{x∈B} λᵢ(x)|^p    where 0 < p ≤ 1
   ```

5. **ReLoBRaLo Loss Balancing**: Dynamically balances multiple loss terms (PDE residual, boundary conditions, initial conditions) using adaptive weights that evolve during training.

**Key advantages:**
- Automatic domain decomposition learned during training
- Each expert specializes on specific regions
- Gating network enables smooth transitions between experts
- Sparsity regularization acts as automated architecture search
- Better performance than single PINNs and manual domain decomposition methods

## Implementation

### MoE-PINN Model Architecture

```python
class MoEPINN(tf.keras.Model):
    """
    MoEPINN - Mixture-of-Expert Physics Informed Neural Network
    Ensemble of PINN models with gating mechanism to determine contribution of each expert
    """
    def __init__(self,
                 n_models: int,          # Number of PINN experts in ensemble
                 n_layers: list,         # List of layer counts for each expert
                 n_nodes: list,          # List of node counts for each expert
                 activations: list,      # List of activation functions for each expert
                 gate_n_layers: int,     # Number of layers in gating network
                 gate_n_nodes: int,      # Number of nodes per layer in gating network
                 gate_activation: Union[str, Callable],  # Activation for gating network
                 softmax_temp: float = 1,  # Temperature for softmax in gating
                 name: str = 'moepinn',
                 **kwargs):
        super().__init__(name=name)
        self.n_models = n_models
        self.softmax_temp = softmax_temp

        # Create ensemble of PINN experts with potentially different architectures
        self.pinns = [
            FCNN(1, n_layers[i], n_nodes[i], self.activations[i], name='pinn_'+str(i))
            for i in range(n_models)
        ]

        # Create gating network (only needed for ensembles with multiple experts)
        if self.n_models > 1:
            self.gate = FCNN(
                n_models, gate_n_layers, gate_n_nodes, gate_activation,
                out_activation=lambda x, t, u: u,  # No derivative computation for gate
                name='gating_network'
            )

    def call(self, inputs):
        """
        Forward pass: compute weighted ensemble prediction
        inputs: (x, t) coordinates, shape (batch_size, 2)
        """
        # Get predictions from all expert PINNs
        # Each u[i] has shape (batch_size, 4): [u, dudx, dudt, dudxx]
        u = [pinn(inputs) for pinn in self.pinns]

        if self.n_models > 1:
            # Compute importance weights via gating network
            logits = self.gate(inputs)  # (batch_size, n_models)
            weights = tf.nn.softmax(logits / self.softmax_temp, axis=-1)

            # Compute weighted sum across all experts
            weighted_u = tf.reduce_sum(
                [u[i] * weights[:, i:i+1] for i in range(self.n_models)],
                axis=0
            )
            return weighted_u
        else:
            # Single expert mode (standard PINN)
            return u[0]

    def get_importances(self, inputs):
        """
        Computes the importance scores of the models in the ensemble
        Returns: (batch_size, n_models) tensor of importance weights
        """
        if self.n_models == 1:
            return tf.ones((tf.shape(inputs)[0], 1))

        logits = self.gate(inputs)
        return tf.nn.softmax(logits / self.softmax_temp, axis=-1)
```

### Individual PINN Expert with Residual Connections

```python
class FCNN(tf.keras.Model):
    """
    Fully-connected neural network with residual connections for PINN experts
    """
    def __init__(self,
                 n_out_vars: int = 1,           # Number of output variables
                 n_layers: int = 4,             # Number of hidden layers
                 n_nodes: int = 256,            # Nodes per hidden layer
                 activation: Union[str, Callable] = 'tanh',
                 out_activation: Callable = lambda x, t, u: tf.concat(
                     [u] + list(compute_derivatives(x, t, u)), axis=-1
                 ),  # Append derivatives to output
                 **kwargs):
        super().__init__(**kwargs)

        # Create sequence of hidden layers with same width
        self.layer_sequence = [
            tf.keras.layers.Dense(
                n_nodes, activation=activation,
                kernel_initializer='glorot_normal'
            )
            for _ in range(n_layers)
        ]

        # Output layer (no bias, linear activation)
        self.dense_out = tf.keras.layers.Dense(
            n_out_vars, use_bias=False,
            kernel_initializer='glorot_normal'
        )

        self.out_activation = out_activation

    def call(self, xt):
        """
        Forward pass with residual connections
        xt: input (x, t) coordinates, shape (batch_size, 2)
        """
        x, t = xt[:, :1], xt[:, 1:]

        # Concatenate inputs
        u = tf.concat([x, t], axis=-1)

        # First layer
        u = self.layer_sequence[0](u)

        # Subsequent layers with residual connections: h_{l+1} = layer(h_l) + h_l
        for layer in self.layer_sequence[1:]:
            u = layer(u) + u

        # Output layer
        u = self.dense_out(u)

        # Apply output activation (computes derivatives via autodiff)
        return self.out_activation(x, t, u)
```

### Computing Derivatives via Automatic Differentiation

```python
def compute_derivatives(x, t, u):
    """
    Computes derivatives of u with respect to x and t using automatic differentiation
    Required for Burgers equation: u_t + u*u_x - ν*u_xx = 0

    Parameters
    ----------
    x : tf.Tensor, shape (batch_size, 1)
        Spatial coordinate
    t : tf.Tensor, shape (batch_size, 1)
        Temporal coordinate
    u : tf.Tensor, shape (batch_size, 1)
        Network prediction u(x,t)

    Returns
    -------
    tuple of (dudx, dudt, dudxx)
        First and second order derivatives
    """
    # First order derivatives
    dudx, dudt = tf.gradients(u, [x, t])

    # Second order derivative with respect to x
    dudxx = tf.gradients(dudx, x)[0]

    return dudx, dudt, dudxx
```

### Burgers PDE Loss Computation

```python
class BurgersPDE:
    """
    Burgers equation: u_t + u*u_x - ν*u_xx = 0
    Domain: x ∈ [-1, 1], t ∈ [0, 1]
    IC: u(0, x) = -sin(πx)
    BC: u(t, -1) = u(t, 1) = 0
    """
    def __init__(self):
        self.nue = 0.01 / np.pi  # Viscosity coefficient
        self.num_terms = 3       # Number of loss terms (PDE, BC, IC)

    def compute_loss(self, x, t, preds, eval=False):
        """
        Computes physics-informed loss for Burgers equation

        Parameters
        ----------
        x : tf.Tensor, shape (batch_size, 1)
            Spatial coordinates
        t : tf.Tensor, shape (batch_size, 1)
            Temporal coordinates
        preds : tf.Tensor, shape (batch_size, 4)
            Network predictions: [u, dudx, dudt, dudxx]
        eval : bool
            If True, also compute error against reference solution

        Returns
        -------
        tuple of loss tensors (L_f, L_bc, L_ic) or (L_f, L_bc, L_ic, L_u)
        """
        # Extract predictions and derivatives
        u = preds[:, 0:1]
        dudx = preds[:, 1:2]
        dudt = preds[:, 2:3]
        dudxx = preds[:, 3:4]

        # Governing equation residual: u_t + u*u_x - ν*u_xx = 0
        # Using power 4 instead of square for stronger penalty
        L_f = (dudt + u * dudx - self.nue * dudxx)**4

        # Determine boundary points using tolerance
        EPS = 1e-5
        x_lower = tf.cast(isclose(x, -1, rtol=0., atol=EPS), dtype=tf.float32)  # x = -1
        x_upper = tf.cast(isclose(x,  1, rtol=0., atol=EPS), dtype=tf.float32)  # x = 1
        t_lower = tf.cast(isclose(t,  0, rtol=0., atol=EPS), dtype=tf.float32)  # t = 0

        # Boundary condition loss: u = 0 at x = ±1
        L_bc = ((x_lower + x_upper) * u)**2

        # Initial condition loss: u(0, x) = -sin(πx)
        L_ic = (t_lower * (u + tf.math.sin(np.pi * x)))**2

        if eval:
            # Also compute error against reference solution
            L_u = (self.u - u)**2
            return L_f, L_bc, L_ic, L_u

        return L_f, L_bc, L_ic
```

### ReLoBRaLo Adaptive Loss Balancing

```python
class ReLoBRaLoLoss(tf.keras.losses.Loss):
    """
    ReLoBRaLo (Relative Loss Balancing with Random Lookback) for dynamic loss weighting
    Automatically balances multiple loss terms during training
    """
    def __init__(self,
                 pde: BurgersPDE,
                 alpha: float = 0.999,      # Exponential moving average decay rate
                 temperature: float = 0.5,   # Softmax temperature for sharpness
                 rho: float = 0.999,        # Probability of random lookback
                 name='ReLoBRaLoLoss'):
        super().__init__(name=name)
        self.pde = pde
        self.alpha = alpha
        self.temperature = temperature
        self.rho = rho
        self.call_count = tf.Variable(0, trainable=False, dtype=tf.int16)

        # Initialize trainable lambda weights (one per loss term)
        self.lambdas = [tf.Variable(1., trainable=False) for _ in range(pde.num_terms)]
        self.last_losses = [tf.Variable(1., trainable=False) for _ in range(pde.num_terms)]
        self.init_losses = [tf.Variable(1., trainable=False) for _ in range(pde.num_terms)]

    def call(self, xt, preds):
        """
        Compute weighted loss with dynamic balancing
        """
        x, t = xt[:, :1], xt[:, 1:]
        losses = [tf.reduce_mean(loss) for loss in self.pde.compute_loss(x, t, preds)]

        # Special handling for first two iterations
        alpha = tf.cond(tf.equal(self.call_count, 0),
                lambda: 1.,  # First iteration: use only initial lambdas
                lambda: tf.cond(tf.equal(self.call_count, 1),
                                lambda: 0.,  # Second iteration: use only lambda_hat
                                lambda: self.alpha))  # Default thereafter

        rho = tf.cond(tf.equal(self.call_count, 0),
              lambda: 1.,
              lambda: tf.cond(tf.equal(self.call_count, 1),
                              lambda: 1.,
                              lambda: tf.cast(
                                  tf.random.uniform(shape=()) < self.rho,
                                  dtype=tf.float32
                              )))

        # Compute new lambdas based on loss ratio w.r.t. previous iteration
        EPS = 1e-5
        lambdas_hat = [
            losses[i] / (self.last_losses[i] * self.temperature + EPS)
            for i in range(len(losses))
        ]
        lambdas_hat = tf.nn.softmax(lambdas_hat - tf.reduce_max(lambdas_hat)) * \
                      tf.cast(len(losses), dtype=tf.float32)

        # Compute lambdas based on loss ratio w.r.t. first iteration
        init_lambdas_hat = [
            losses[i] / (self.init_losses[i] * self.temperature + EPS)
            for i in range(len(losses))
        ]
        init_lambdas_hat = tf.nn.softmax(init_lambdas_hat - tf.reduce_max(init_lambdas_hat)) * \
                           tf.cast(len(losses), dtype=tf.float32)

        # Combine current, previous, and initial lambdas with random lookback
        new_lambdas = [
            (rho * alpha * self.lambdas[i] +
             (1 - rho) * alpha * init_lambdas_hat[i] +
             (1 - alpha) * lambdas_hat[i])
            for i in range(len(losses))
        ]
        self.lambdas = [
            var.assign(tf.stop_gradient(lam))
            for var, lam in zip(self.lambdas, new_lambdas)
        ]

        # Compute final weighted loss
        loss = tf.reduce_sum([lam * loss for lam, loss in zip(self.lambdas, losses)])

        # Update loss history
        self.last_losses = [
            var.assign(tf.stop_gradient(loss))
            for var, loss in zip(self.last_losses, losses)
        ]

        # Store initial losses on first iteration
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

### Training Setup

```python
# Example: Train ensemble of 5 PINNs with diverse architectures
moepinn = MoEPINN(
    n_models=5,
    n_layers=[2, 2, 2, 3, 3],                    # Varying depths
    n_nodes=[64, 64, 128, 128, 256],             # Varying widths
    activations=[
        'tanh',
        lambda x: tf.math.sin(np.pi * x),        # Sine activation
        'tanh',
        'swish',
        'swish'
    ],
    gate_n_layers=2,                              # Gating network: 2 layers
    gate_n_nodes=64,                              # Gating network: 64 nodes per layer
    gate_activation='tanh',
)

# Create loss with adaptive balancing
pde = BurgersPDE()
loss = ReLoBRaLoLoss(pde, temperature=0.1, rho=0.99, alpha=0.999)

# Compile model
moepinn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=loss,
    metrics=[BurgersMetric(pde), ReLoBRaLoLambdaMetric(loss)]
)

# Train with learning rate reduction and early stopping
h = moepinn.fit(
    pde.get_train_dataset(),
    epochs=1000,
    steps_per_epoch=100,
    validation_data=pde.get_validation_dataset(),
    validation_steps=1,
    callbacks=[
        ReduceLROnPlateau(
            monitor='unscaled_loss',
            factor=0.1,        # Reduce LR by 10x
            patience=10,       # After 10 epochs without improvement
            min_delta=0,
            verbose=True
        ),
        EarlyStopping(
            monitor='unscaled_loss',
            patience=32,                    # Stop after 32 epochs without improvement
            restore_best_weights=True,      # Restore best model
            verbose=True
        )
    ]
)
```

## Critical Parameters

1. **Number of experts (n_models)**
   - Typical range: 3-5 experts
   - Paper findings: 3 experts optimal for Burgers equation
   - More experts increase capacity but may overfit
   - Too many experts difficult to train effectively

2. **Expert architectures**
   - Depths: 2-4 hidden layers
   - Widths: 64-256 nodes per layer
   - Diversity helps: mix different depths, widths, and activations
   - Paper finding: sine activation in ensemble outperforms single PINN with sine

3. **Activation functions**
   - tanh: Most common for PINNs, but paper found gating network often discards it
   - sine: Performed exceptionally well in ensemble (better than single PINN)
   - swish: Good performance
   - Recommendation: Use diverse activations across experts

4. **Gating network architecture**
   - Fixed configuration: 2 layers, 64 nodes
   - Activation: tanh (infinitely differentiable, crucial for replacing ReLU)
   - Smaller than experts (doesn't need to solve PDE)

5. **Softmax temperature**
   - Controls sharpness of expert selection
   - Default: 1.0
   - Lower values → sharper transitions between experts
   - Higher values → smoother blending

6. **Sparsity regularization (optional)**
   - Order p: 0 < p ≤ 1 (typically 0.25 or 0.5)
   - p = 0.5 found to work better than p = 2
   - Lower p → more aggressive sparsity (drives weak experts to zero)
   - Enables automated expert selection

7. **ReLoBRaLo hyperparameters**
   - alpha: 0.999 (exponential moving average decay)
   - temperature: 0.01-0.1 (lower for sharper loss balancing)
   - rho: 0.99 (probability of random lookback)
   - Critical for balancing PDE, BC, and IC loss terms

8. **Optimizer settings**
   - Adam optimizer with initial learning rate: 0.0001-0.01
   - Learning rate reduction: factor 0.1 after plateau (patience 10 epochs)
   - Early stopping: patience 32 epochs
   - Lower learning rate for ensembles (0.0001) vs single PINN (0.01)

9. **Training data**
   - Domain points: 3000 per batch (randomly sampled)
   - Boundary points: 1000 per boundary (3 boundaries for Burgers)
   - Total batch size: ~6000 points
   - Data regenerated each epoch (collocation points)

10. **PDE-specific parameters (Burgers equation)**
    - Viscosity (ν): 0.01/π ≈ 0.00318
    - Domain: x ∈ [-1, 1], t ∈ [0, 1]
    - Governing equation penalty: power 4 (stronger than MSE)
    - Boundary conditions: Dirichlet (u = 0 at x = ±1)
    - Initial condition: u(0, x) = -sin(πx)
