# Physics-Informed Neural Networks for Burgers Equation (Forward Problem)

**Keywords**: [PDE, nonlinear, parabolic, forward-problem, burgers, 1D, periodic, PINN, MLP, strong-form, adam, lbfgs, mse, pytorch]

**Problem:** Physics-informed neural networks (PINNs) solve nonlinear partial differential equations by embedding the governing equations directly into the loss function of a neural network. This approach addresses the challenge of solving the 1D Burgers equation, a fundamental nonlinear PDE in fluid dynamics that models viscous flow and can develop shock-like solutions. The method eliminates the need for mesh generation and can handle complex boundary conditions naturally.

**Issues addressed:**
- Mesh-free solution of nonlinear PDEs
- Handling periodic boundary conditions
- Automatic differentiation for computing derivatives required in PDEs
- Simultaneous satisfaction of initial/boundary conditions and PDE constraints

## Key Method

The PINN approach represents the solution u(x,t) as a neural network and trains it to satisfy:

1. **Data loss**: Match initial and boundary conditions
2. **Physics loss**: Satisfy the PDE residual at collocation points

For the 1D Burgers equation:
**∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²**

where:
- u(x,t) is the velocity field
- ν is the viscosity coefficient (ν = 0.01/π)
- Domain: x ∈ [-1, 1], t ∈ [0, 1]
- Periodic boundary conditions: u(-1,t) = u(1,t)

**Key advantages:**
- Mesh-free: no spatial discretization required
- Automatic differentiation computes exact derivatives
- Natural handling of complex boundary conditions
- Single neural network approximates entire solution field

## Implementation

### Core Network Class

```python
class FCN(nn.Module):
    """
    Fully Connected Network for Physics-Informed Neural Networks
    """
    def __init__(self, layers):
        """
        layers: list defining network architecture
                e.g., [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
                2 inputs (x, t), 8 hidden layers with 20 neurons, 1 output (u)
        """
        super().__init__()
        self.activation = nn.Tanh()  # Tanh activation for smooth solutions
        self.loss_function = nn.MSELoss(reduction='mean')

        # Create list of linear layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1])
                                      for i in range(len(layers)-1)])
        self.iter = 0

        # Xavier Normal Initialization for stable training
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):
        """Forward pass through network"""
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)

        a = x.float()
        # Pass through hidden layers with tanh activation
        for i in range(len(self.linears)-1):
            z = self.linears[i](a)
            a = self.activation(z)

        # Output layer (no activation)
        a = self.linears[-1](a)
        return a
```

### Loss Functions

```python
def loss_BC(self, x_BC):
    """
    Boundary condition loss
    x_BC: boundary condition points
    """
    loss_BC = self.loss_function(self.forward(x_BC), self.f_hat())
    return loss_BC

def loss_PDE(self, X_train_Nf):
    """
    PDE residual loss for Burgers equation
    Computes: f = u_t + u*u_x - nu*u_xx

    X_train_Nf: collocation points [x, t] for evaluating PDE residual
    """
    g = X_train_Nf.clone()
    g.requires_grad = True  # Enable automatic differentiation

    u = self.forward(g)

    # First-order derivatives using automatic differentiation
    u_x_t = autograd.grad(u, g, torch.ones([X_train_Nf.shape[0], 1]).to(device),
                          retain_graph=True, create_graph=True)[0]

    # Second-order derivatives
    u_xx_tt = autograd.grad(u_x_t, g, torch.ones(X_train_Nf.shape).to(device),
                            create_graph=True)[0]

    # Extract individual derivative components
    u_x = u_x_t[:, [0]]   # ∂u/∂x
    u_t = u_x_t[:, [1]]   # ∂u/∂t
    u_xx = u_xx_tt[:, [0]] # ∂²u/∂x²

    # Burgers equation residual: u_t + u*u_x - nu*u_xx = 0
    f = u_t + (self.forward(g)) * u_x - (0.01/np.pi) * u_xx

    return self.loss_function(f, f_hat)

def loss(self, x_BC, X_train_Nf):
    """
    Total loss = boundary condition loss + PDE residual loss
    """
    loss_bc = self.loss_BC(x_BC)
    loss_pde = self.loss_PDE(X_train_Nf)
    return loss_bc + loss_pde
```

### Data Preparation

```python
# Domain and parameters
min_x, max_x = -1, 1
min_t, max_t = 0, 1
total_points_x = 200
total_points_t = 100
nu = 0.01/np.pi  # Viscosity coefficient

# Nu: Number of boundary training points
# Nf: Number of collocation points for PDE residual
Nu = 100
Nf = 10000

# Create spatial and temporal grids
x = torch.linspace(min_x, max_x, total_points_x).view(-1, 1)
t = torch.linspace(min_t, max_t, total_points_t).view(-1, 1)

# Create mesh grid [X, T]
X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1), indexing='ij')

# Load exact solution (for training initial/boundary conditions)
usol = torch.load('burgers_solution.pt')

# Boundary conditions at t=0 (initial condition)
x_BC = torch.hstack((X[:, 0][:, None], T[:, 0][:, None]))
u_BC = usol[:, 0][:, None]

# Collocation points using Latin Hypercube Sampling
X_train_Nf = torch.vstack((
    torch.hstack((X.transpose(1, 0).flatten()[:, None],
                  T.transpose(1, 0).flatten()[:, None]))
))

# Randomly select Nu boundary points and Nf collocation points
idx = np.random.choice(x_BC.shape[0], Nu, replace=False)
x_BC_sampled = x_BC[idx, :]
u_BC_sampled = u_BC[idx, :]

idx_collocation = np.random.choice(X_train_Nf.shape[0], Nf, replace=False)
X_train_Nf_sampled = X_train_Nf[idx_collocation, :]
```

### Training Loop

```python
# Network architecture: 2 inputs -> 8 hidden layers (20 neurons) -> 1 output
layers = np.array([2, 20, 20, 20, 20, 20, 20, 20, 20, 1])

# Create model and optimizer
torch.manual_seed(123)
model = FCN(layers)
model.to(device)

# Two-stage optimization
steps_adam = 10000
lr = 1e-1

# Stage 1: Adam optimizer for initial training
optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)

for i in range(steps_adam):
    loss = model.loss(x_BC_sampled.to(device), X_train_Nf_sampled.to(device))
    optimizer_adam.zero_grad()
    loss.backward()
    optimizer_adam.step()

    if i % (steps_adam/10) == 0:
        print(f'Adam Step {i}, Loss: {loss.item():.6e}')

# Stage 2: L-BFGS optimizer for fine-tuning
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=500,
    max_eval=500,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    history_size=100,
    line_search_fn='strong_wolfe'
)

def closure():
    optimizer_lbfgs.zero_grad()
    loss = model.loss(x_BC_sampled.to(device), X_train_Nf_sampled.to(device))
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)
```

### Prediction and Evaluation

```python
# Predict solution over entire domain
u_pred = model(X_train_Nf.to(device))
u_pred = u_pred.reshape(total_points_x, total_points_t)

# Compute relative L2 error
u_exact = usol.flatten()
u_pred_flat = u_pred.cpu().detach().numpy().flatten()
error = torch.linalg.norm(u_exact - torch.from_numpy(u_pred_flat), 2) / torch.linalg.norm(u_exact, 2)
print(f'Relative L2 Error: {error.item():.6e}')
```

## Critical Parameters

1. **Network architecture**
   - Layers: [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
   - 8 hidden layers with 20 neurons each
   - Input: (x, t) coordinates
   - Output: u(x, t) velocity field
   - Activation: Tanh (smooth, bounded, suitable for PDE solutions)

2. **Training data**
   - Boundary/initial points (Nu): 100
   - Collocation points (Nf): 10,000
   - Sampling: Random selection from full domain

3. **Optimizer sequence**
   - Stage 1: Adam optimizer
     - Steps: 10,000
     - Learning rate: 0.1
     - Purpose: Fast initial convergence
   - Stage 2: L-BFGS optimizer
     - Max iterations: 500
     - Line search: Strong Wolfe conditions
     - Purpose: Fine-tuning to high accuracy

4. **PDE-specific parameters**
   - Viscosity (ν): 0.01/π ≈ 0.00318
   - Domain: x ∈ [-1, 1], t ∈ [0, 1]
   - Boundary conditions: Periodic at x = -1 and x = 1
   - Initial condition: Specified at t = 0

5. **Initialization**
   - Xavier normal initialization for weights
   - Zero initialization for biases
   - Random seed: 123 for reproducibility

6. **Loss function**
   - MSE for both boundary condition loss and PDE residual loss
   - Total loss: L = L_BC + L_PDE
   - Equal weighting between data and physics constraints
