# Physics-Informed Neural Networks for 1D Poisson Equation with Dirichlet Boundary Conditions

**Keywords**: [PDE, linear, elliptic, second-order, forward-problem, poisson, 1D, dirichlet, PINN, MLP, strong-form, adam, mse, pytorch]

**Problem:** Physics-informed neural networks solve the 1D Poisson equation, a fundamental elliptic PDE that arises in electrostatics, heat conduction, and fluid flow. The Poisson equation relates the Laplacian of a function to a source term. This implementation demonstrates PINN's ability to solve elliptic PDEs with Dirichlet boundary conditions, where the solution must satisfy both the PDE and prescribed boundary values.

**Issues addressed:**
- Mesh-free solution of elliptic PDEs
- Enforcement of Dirichlet boundary conditions
- Automatic computation of second-order derivatives
- Solution of boundary value problems without discretization

## Key Method

The PINN represents the solution u(x) as a neural network and minimizes:

1. **Boundary condition loss**: Match prescribed values at boundaries
2. **PDE residual loss**: Satisfy -Δu = f(x)

The 1D Poisson equation:
**-∂²u/∂x² = π²sin(πx)**

where:
- u(x) is the potential or temperature field
- Domain: x ∈ [-1, 1]
- Boundary conditions: u(-1) = 0, u(1) = 0 (Dirichlet)
- Source term: f(x) = π²sin(πx)
- Exact solution: u(x) = sin(πx)

**Key advantages:**
- No mesh generation required for elliptic problems
- Natural boundary condition enforcement
- Continuous solution representation
- Exact derivatives through automatic differentiation

## Implementation

### Core Network Class

```python
class FCN(nn.Module):
    """
    Fully Connected Network for Poisson Equation PINN
    """
    def __init__(self, layers):
        """
        layers: network architecture
                e.g., [1, 50, 50, 20, 50, 50, 1]
                1 input (x), 5 hidden layers, 1 output (u)
        """
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')

        # Create neural network layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1])
                                      for i in range(len(layers)-1)])
        self.iter = 0

        # Xavier Normal Initialization
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
def f_BC(x):
    """
    Boundary condition helper function
    Returns: 1 - |x|
    Satisfies BC: f_BC(-1) = 0, f_BC(1) = 0
    Note: This is NOT the exact solution, just helps describe BC
    """
    return 1 - torch.abs(x)

def f_real(x):
    """
    Exact solution: u(x) = sin(πx)
    Used for validation and boundary condition values
    """
    return torch.sin(np.pi * x)

def PDE(x):
    """
    Right-hand side of Poisson equation
    -∂²u/∂x² = π²sin(πx)
    Returns: -π²sin(πx)
    """
    return -1 * (np.pi**2) * torch.sin(np.pi * x)

def lossBC(self, x_BC, y_BC):
    """
    Boundary condition loss
    x_BC: boundary points [x=-1, x=1]
    y_BC: prescribed values at boundaries (both 0)
    """
    loss_BC = self.loss_function(self.forward(x_BC), y_BC)
    return loss_BC

def lossPDE(self, x_PDE):
    """
    PDE residual loss for Poisson equation
    Residual: f = ∂²u/∂x² + π²sin(πx) = 0

    x_PDE: collocation points in domain [-1, 1]
    """
    g = x_PDE.clone()
    g.requires_grad = True  # Enable automatic differentiation

    # Forward pass to get u(x)
    f = self.forward(g)

    # Compute first derivative ∂u/∂x
    f_x = autograd.grad(f, g, torch.ones([x_PDE.shape[0], 1]).to(device),
                        retain_graph=True, create_graph=True)[0]

    # Compute second derivative ∂²u/∂x²
    f_xx = autograd.grad(f_x, g, torch.ones([x_PDE.shape[0], 1]).to(device),
                         create_graph=True)[0]

    # Poisson equation residual: ∂²u/∂x² - (-π²sin(πx))
    # Or equivalently: ∂²u/∂x² + π²sin(πx) = 0
    return self.loss_function(f_xx, PDE(g))

def loss(self, x_BC, y_BC, x_PDE):
    """
    Total loss = boundary condition loss + PDE residual loss
    """
    loss_bc = self.lossBC(x_BC, y_BC)
    loss_pde = self.lossPDE(x_PDE)
    return loss_bc + loss_pde
```

### Data Preparation

```python
# Domain parameters
min_x = -1
max_x = 1
total_points = 500

# Training data sizes
Nu = 2    # Number of boundary points (2 endpoints)
Nf = 250  # Number of collocation points for PDE residual

# Generate full domain for visualization
x = torch.linspace(min_x, max_x, total_points).view(-1, 1)
y = f_real(x)  # Exact solution: u(x) = sin(πx)

# Set boundary conditions at endpoints
BC_1 = x[0, :]   # x = -1
BC_2 = x[-1, :]  # x = 1

# Combine boundary points
all_train = torch.vstack([BC_1, BC_2])

# Select Nu boundary points (both endpoints in this case)
idx = np.random.choice(all_train.shape[0], Nu, replace=False)
x_BC = all_train[idx]

# Boundary values using helper function (both should be 0)
y_BC = f_BC(x_BC)

# Generate collocation points using Latin Hypercube Sampling
# LHS ensures good coverage of domain for PDE residual evaluation
x_PDE = BC_1 + (BC_2 - BC_1) * lhs(1, Nf)

# Combine collocation points with boundary points
x_PDE = torch.vstack((x_PDE, x_BC))
```

### Training Loop

```python
# Network architecture: 1 input -> 5 hidden layers -> 1 output
# Hidden layers: [50, 50, 20, 50, 50] neurons
layers = np.array([1, 50, 50, 20, 50, 50, 1])

# Training parameters
steps = 5000
lr = 1e-3

# Move data to device
torch.manual_seed(123)
x_PDE = x_PDE.float().to(device)
x_BC = x_BC.to(device)
y_BC = y_BC.to(device)

# Create model
model = FCN(layers)
model.to(device)

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)

# Training loop
print('Training started...')
start_time = time.time()

for i in range(steps):
    loss = model.loss(x_BC, y_BC, x_PDE)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % (steps/10) == 0:
        print(f'Step {i}, Loss: {loss.item():.6e}')

elapsed = time.time() - start_time
print(f'Training completed in {elapsed:.2f} seconds')
```

### Prediction and Analysis

```python
# Predict solution over full domain
yh = model(x.to(device))
y_exact = f_real(x)

# Compute boundary condition error
bc_error = model.lossBC(x.to(device), f_real(x).to(device))
print(f'\nBoundary Condition Loss: {bc_error.item():.6e}')

# Compute first derivative for validation
g = x.to(device).clone()
g.requires_grad = True

# Forward pass
f = model(g)

# Compute ∂u/∂x
f_x = autograd.grad(f, g, torch.ones([g.shape[0], 1]).to(device),
                    retain_graph=True, create_graph=True)[0]

# Move to CPU for analysis
y_pred = yh.detach().cpu().numpy()
y_exact_np = y_exact.detach().numpy()
derivative = f_x.detach().cpu().numpy()

# Compute relative L2 error
error_L2 = np.linalg.norm(y_exact_np - y_pred, 2) / np.linalg.norm(y_exact_np, 2)
print(f'Relative L2 Error: {error_L2:.6e}')

# Visualization
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(x, y_exact_np, color='blue', label='Exact: u(x) = sin(πx)')
ax.plot(x, y_pred, color='red', linestyle='--', label='PINN Prediction')
ax.plot(x, derivative, color='green', label="First Derivative: u'(x)")
ax.set_xlabel('x')
ax.set_ylabel('u(x)')
ax.legend(loc='upper left')
ax.set_title('1D Poisson Equation: -u_xx = π²sin(πx)')
plt.show()
```

## Critical Parameters

1. **Network architecture**
   - Layers: [1, 50, 50, 20, 50, 50, 1]
   - 5 hidden layers with varying widths (50-50-20-50-50)
   - Input: x coordinate (1D spatial)
   - Output: u(x) solution value
   - Activation: Tanh (smooth, suitable for elliptic problems)
   - Deep architecture helps capture smooth solution

2. **Training data**
   - Boundary points (Nu): 2 (endpoints x=-1 and x=1)
   - Collocation points (Nf): 250
   - Total training points: 252
   - Sampling: Latin Hypercube Sampling for uniform coverage
   - Boundary values: u(-1) = 0, u(1) = 0

3. **Optimizer**
   - Single-stage Adam optimizer
   - Steps: 5,000
   - Learning rate: 0.001
   - amsgrad: False
   - No L-BFGS needed for this linear elliptic problem

4. **PDE-specific parameters**
   - PDE type: Elliptic (Poisson equation)
   - Equation: -∂²u/∂x² = π²sin(πx)
   - Domain: x ∈ [-1, 1]
   - Boundary conditions: Dirichlet (u = 0 at both boundaries)
   - Source term: f(x) = π²sin(πx)
   - Exact solution: u(x) = sin(πx)

5. **Initialization**
   - Xavier normal initialization for weights (gain=1.0)
   - Zero initialization for biases
   - Random seed: 123 for reproducibility

6. **Loss function**
   - MSE for both boundary condition loss and PDE residual loss
   - Total loss: L = L_BC + L_PDE
   - Equal weighting between boundary and physics constraints
   - Both losses have similar magnitudes for balanced training

7. **Expected accuracy**
   - Relative L2 error: typically < 1e-4
   - Boundary condition loss: < 1e-5
   - Fast convergence due to smooth solution and linear PDE
   - Second derivatives are accurately computed via automatic differentiation

8. **Key implementation details**
   - Automatic differentiation computes second derivatives exactly
   - No finite difference approximation errors
   - Boundary condition helper function f_BC(x) = 1 - |x| is used for generating BC data, but NOT as the solution
   - The network learns the true solution u(x) = sin(πx) from scratch
   - Elliptic PDEs typically require more collocation points than time-dependent problems
