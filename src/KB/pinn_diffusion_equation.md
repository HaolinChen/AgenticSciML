# Physics-Informed Neural Networks for Diffusion Equation with Source Term

**Keywords**: [PDE, linear, parabolic, forward-problem, diffusion, 2D, dirichlet, periodic, PINN, MLP, strong-form, adam, mse, pytorch]

**Problem:** Physics-informed neural networks solve the diffusion equation with a time-dependent source term. This problem models heat conduction or mass diffusion with external forcing. The method addresses the challenge of solving parabolic PDEs with known analytical solutions, demonstrating PINN's ability to accurately approximate smooth solutions that satisfy both the PDE and boundary/initial conditions.

**Issues addressed:**
- Mesh-free solution of linear parabolic PDEs
- Handling Dirichlet boundary conditions at spatial boundaries
- Initial condition enforcement at t=0
- Automatic computation of spatial and temporal derivatives

## Key Method

The PINN represents the solution y(x,t) as a neural network and enforces:

1. **Initial condition**: y(x,0) = sin(πx)
2. **Boundary conditions**: y(0,t) = 0, y(1,t) = 0 (Dirichlet)
3. **PDE residual**: ∂y/∂t - ∂²y/∂x² + e^(-t)(sin(πx) - π²sin(πx)) = 0

The diffusion equation with source term:
**∂y/∂t = ∂²y/∂x² - e^(-t)(sin(πx) - π²sin(πx))**

where:
- y(x,t) is the temperature or concentration field
- Domain: x ∈ [0, 1], t ∈ [0, 1]
- Exact solution: y(x,t) = e^(-t)sin(πx)

**Key advantages:**
- Verifiable accuracy against exact analytical solution
- Natural enforcement of boundary and initial conditions
- Automatic differentiation eliminates discretization errors
- Single network captures full spatiotemporal solution

## Implementation

### Core Network Class

```python
class FCN(nn.Module):
    """
    Fully Connected Network for Diffusion Equation PINN
    """
    def __init__(self, layers):
        """
        layers: network architecture
                e.g., [2, 32, 32, 1] - 2 inputs (x,t), 2 hidden layers (32 neurons), 1 output
        """
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')

        # Create neural network layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1])
                                      for i in range(len(layers)-1)])
        self.iter = 0

        # Xavier initialization for stable training
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):
        """Forward pass through network"""
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)

        a = x.float()
        for i in range(len(self.linears)-1):
            z = self.linears[i](a)
            a = self.activation(z)

        a = self.linears[-1](a)
        return a
```

### Loss Functions

```python
def loss_BC(self, x_BC, y_BC):
    """
    Boundary and initial condition loss
    x_BC: points on boundaries and initial time [x, t]
    y_BC: prescribed values at boundary/initial points
    """
    loss_BC = self.loss_function(self.forward(x_BC), y_BC)
    return loss_BC

def loss_PDE(self, X_train_Nf):
    """
    PDE residual loss for diffusion equation
    Residual: f = y_t - y_xx + source(x,t)
    where source(x,t) = e^(-t)(sin(πx) - π²sin(πx))

    X_train_Nf: collocation points [x, t] for PDE evaluation
    """
    g = X_train_Nf.clone()
    g.requires_grad = True  # Enable automatic differentiation

    y = self.forward(g)

    # Compute first-order derivatives using autograd
    y_x_t = autograd.grad(y, g, torch.ones([X_train_Nf.shape[0], 1]).to(device),
                          retain_graph=True, create_graph=True)[0]

    # Compute second-order derivatives
    y_xx_tt = autograd.grad(y_x_t, g, torch.ones(X_train_Nf.shape).to(device),
                            create_graph=True)[0]

    # Extract individual derivatives
    y_x = y_x_t[:, [0]]   # ∂y/∂x
    y_t = y_x_t[:, [1]]   # ∂y/∂t
    y_xx = y_xx_tt[:, [0]] # ∂²y/∂x²

    # Extract x and t coordinates
    x = g[:, [0]]
    t = g[:, [1]]

    # Source term: e^(-t)(sin(πx) - π²sin(πx))
    source = torch.exp(-t) * (torch.sin(np.pi * x) - (np.pi**2) * torch.sin(np.pi * x))

    # Diffusion equation residual: y_t - y_xx + source = 0
    f = y_t - y_xx + source

    return self.loss_function(f, f_hat)

def loss(self, x_BC, y_BC, X_train_Nf):
    """
    Total loss = boundary/initial condition loss + PDE residual loss
    """
    loss_bc = self.loss_BC(x_BC, y_BC)
    loss_pde = self.loss_PDE(X_train_Nf)
    return loss_bc + loss_pde
```

### Exact Solution and Data Preparation

```python
def exact_solution(x, t):
    """
    Analytical solution: y(x,t) = e^(-t)sin(πx)
    Used for generating boundary/initial conditions and validation
    """
    return torch.exp(-t) * torch.sin(np.pi * x)

# Domain parameters
min_x, max_x = 0, 1
min_t, max_t = 0, 1
total_points_x = 100
total_points_t = 100

# Training data sizes
Nu = 100   # Boundary/initial condition points
Nf = 10000 # Collocation points for PDE residual

# Create spatial and temporal grids
x = torch.linspace(min_x, max_x, total_points_x).view(-1, 1)
t = torch.linspace(min_t, max_t, total_points_t).view(-1, 1)

# Create mesh
X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1), indexing='ij')

# Compute exact solution on grid
y_exact = exact_solution(X, T)

# Boundary conditions:
# 1. Initial condition: y(x, 0) = sin(πx)
x_ic = X[:, 0][:, None]  # All x at t=0
t_ic = T[:, 0][:, None]  # t=0
y_ic = y_exact[:, 0][:, None]

# 2. Left boundary: y(0, t) = 0
x_bc_left = X[0, :][:, None]  # x=0
t_bc_left = T[0, :][:, None]  # All t
y_bc_left = y_exact[0, :][:, None]

# 3. Right boundary: y(1, t) = 0
x_bc_right = X[-1, :][:, None]  # x=1
t_bc_right = T[-1, :][:, None]  # All t
y_bc_right = y_exact[-1, :][:, None]

# Combine all boundary/initial conditions
x_BC = torch.vstack([
    torch.hstack([x_ic, t_ic]),
    torch.hstack([x_bc_left, t_bc_left]),
    torch.hstack([x_bc_right, t_bc_right])
])
y_BC = torch.vstack([y_ic, y_bc_left, y_bc_right])

# Randomly sample Nu boundary points
idx_bc = np.random.choice(x_BC.shape[0], Nu, replace=False)
x_BC_sampled = x_BC[idx_bc, :]
y_BC_sampled = y_BC[idx_bc, :]

# Collocation points (full domain for PDE residual)
X_train_Nf = torch.hstack([X.flatten()[:, None], T.flatten()[:, None]])

# Randomly sample Nf collocation points
idx_pde = np.random.choice(X_train_Nf.shape[0], Nf, replace=False)
X_train_Nf_sampled = X_train_Nf[idx_pde, :]
```

### Training Loop

```python
# Network architecture: 2 inputs -> 2 hidden layers (32 neurons) -> 1 output
layers = np.array([2, 32, 32, 1])

# Create model
torch.manual_seed(123)
model = FCN(layers)
model.to(device)

# Move data to device
x_BC_sampled = x_BC_sampled.to(device)
y_BC_sampled = y_BC_sampled.to(device)
X_train_Nf_sampled = X_train_Nf_sampled.to(device)

# Training parameters
steps = 20000
lr = 1e-3

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)

print('Training started...')
for i in range(steps):
    loss = model.loss(x_BC_sampled, y_BC_sampled, X_train_Nf_sampled)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % (steps/10) == 0:
        print(f'Step {i}, Loss: {loss.item():.6e}')

print('Training completed.')
```

### Prediction and Error Analysis

```python
# Predict solution over entire domain
X_test = torch.hstack([X.flatten()[:, None], T.flatten()[:, None]])
y_pred = model(X_test.to(device))
y_pred = y_pred.reshape(total_points_x, total_points_t)

# Compute relative L2 error
y_exact_flat = y_exact.flatten()
y_pred_flat = y_pred.cpu().detach().flatten()

error_L2 = torch.linalg.norm(y_exact_flat - y_pred_flat, 2) / torch.linalg.norm(y_exact_flat, 2)
print(f'\nRelative L2 Error: {error_L2.item():.6e}')

# Point-wise maximum error
error_max = torch.max(torch.abs(y_exact_flat - y_pred_flat))
print(f'Maximum Absolute Error: {error_max.item():.6e}')
```

## Critical Parameters

1. **Network architecture**
   - Layers: [2, 32, 32, 1]
   - 2 hidden layers with 32 neurons each (smaller than Burgers equation due to linear PDE)
   - Input: (x, t) coordinates
   - Output: y(x, t) temperature/concentration field
   - Activation: Tanh

2. **Training data**
   - Boundary/initial points (Nu): 100
     - Initial condition at t=0 for all x
     - Left boundary at x=0 for all t
     - Right boundary at x=1 for all t
   - Collocation points (Nf): 10,000
   - Sampling: Random selection from boundary and domain

3. **Optimizer**
   - Single-stage Adam optimizer (no L-BFGS needed for this simpler problem)
   - Steps: 20,000
   - Learning rate: 0.001 (lower than Burgers due to smaller network)
   - Fast convergence due to linear nature of PDE

4. **PDE-specific parameters**
   - Domain: x ∈ [0, 1], t ∈ [0, 1]
   - Source term: e^(-t)(sin(πx) - π²sin(πx))
   - Exact solution available: y(x,t) = e^(-t)sin(πx)
   - Boundary conditions: Dirichlet (y=0 at x=0 and x=1)
   - Initial condition: y(x,0) = sin(πx)

5. **Initialization**
   - Xavier normal initialization for weights
   - Zero initialization for biases
   - Random seed: 123 for reproducibility

6. **Loss function**
   - MSE for both boundary/initial condition loss and PDE residual loss
   - Total loss: L = L_BC + L_PDE
   - Equal weighting between data and physics constraints

7. **Expected accuracy**
   - Relative L2 error: typically < 1e-3
   - Fast convergence due to smooth solution and linear PDE
   - No special techniques needed (unlike discontinuous or highly nonlinear problems)
