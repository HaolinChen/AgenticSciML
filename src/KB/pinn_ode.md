# Physics-Informed Neural Networks for Simple Ordinary Differential Equation

**Keywords**: [ODE, linear, first-order, forward-problem, 1D, dirichlet, PINN, MLP, strong-form, adam, mse, pytorch]

**Problem:** Physics-informed neural networks can solve ordinary differential equations by incorporating the ODE residual into the loss function. This implementation addresses a simple first-order ODE with known analytical solution, demonstrating the fundamental PINN approach for ODEs. The method learns the solution y(x) that satisfies both the differential equation and boundary conditions without requiring discretization of the domain.

**Issues addressed:**
- Mesh-free solution of ODEs
- Automatic differentiation for computing derivatives
- Simultaneous enforcement of differential equation and boundary conditions
- Foundation for more complex PDE problems

## Key Method

The PINN represents the solution y(x) as a neural network and minimizes:

1. **Boundary condition loss**: Match prescribed values at boundaries
2. **ODE residual loss**: Satisfy dy/dx = cos(x)

The first-order ODE:
**dy/dx = cos(x)**

where:
- y(x) is the solution function
- Domain: x ∈ [0, 2π]
- Boundary conditions: y(0) = 0, y(2π) = 0
- Exact solution: y(x) = sin(x)

**Key advantages:**
- No need for numerical discretization schemes
- Automatic differentiation provides exact derivatives
- Natural boundary condition enforcement
- Solution is a continuous function (neural network)

## Implementation

### Core Network Class

```python
class FCN(nn.Module):
    """
    Fully Connected Network for ODE PINN
    """
    def __init__(self, layers):
        """
        layers: network architecture
                e.g., [1, 50, 50, 20, 50, 50, 1]
                1 input (x), 5 hidden layers, 1 output (y)
        """
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')

        # Create neural network layers
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
def f_BC(x):
    """
    Exact solution (used only for boundary conditions)
    y(x) = sin(x)
    """
    return torch.sin(x)

def PDE(x):
    """
    Right-hand side of ODE: dy/dx = cos(x)
    Returns: cos(x)
    """
    return torch.cos(x)

def lossBC(self, x_BC):
    """
    Boundary condition loss
    x_BC: boundary points [x=0, x=2π]
    Enforces: y(0) = sin(0) = 0, y(2π) = sin(2π) = 0
    """
    loss_BC = self.loss_function(self.forward(x_BC), f_BC(x_BC))
    return loss_BC

def lossPDE(self, x_PDE):
    """
    ODE residual loss
    Residual: f = dy/dx - cos(x) = 0

    x_PDE: collocation points in domain [0, 2π]
    """
    g = x_PDE.clone()
    g.requires_grad = True  # Enable automatic differentiation

    # Forward pass to get y(x)
    f = self.forward(g)

    # Compute dy/dx using automatic differentiation
    f_x = autograd.grad(f, g, torch.ones([x_PDE.shape[0], 1]).to(device),
                        retain_graph=True, create_graph=True)[0]

    # ODE residual: dy/dx - cos(x)
    loss_PDE = self.loss_function(f_x, PDE(g))
    return loss_PDE

def loss(self, x_BC, x_PDE):
    """
    Total loss = boundary condition loss + ODE residual loss
    """
    loss_bc = self.lossBC(x_BC)
    loss_pde = self.lossPDE(x_PDE)
    return loss_bc + loss_pde
```

### Data Preparation

```python
# Domain parameters
min_x = 0
max_x = 2 * np.pi
total_points = 500

# Training data sizes
Nu = 2    # Number of boundary points (only 2 endpoints)
Nf = 250  # Number of collocation points for ODE residual

# Generate full domain
x = torch.linspace(min_x, max_x, total_points).view(-1, 1)
y = f_BC(x)  # Exact solution (for visualization, not training)

# Set boundary conditions at endpoints
BC_1 = x[0, :]   # x = 0
BC_2 = x[-1, :]  # x = 2π

# Combine boundary points
all_train = torch.vstack([BC_1, BC_2])

# Select Nu boundary points (in this case, both endpoints)
idx = np.random.choice(all_train.shape[0], Nu, replace=False)
x_BC = all_train[idx]

# Generate collocation points using Latin Hypercube Sampling
# Ensures good coverage of domain for ODE residual evaluation
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

# Create model
model = FCN(layers)
model.to(device)

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)

# Training loop
print('Training started...')
start_time = time.time()

for i in range(steps):
    loss = model.loss(x_BC, x_PDE)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % (steps/10) == 0:
        print(f'Step {i}, Loss: {loss.item():.6e}')

elapsed = time.time() - start_time
print(f'Training completed in {elapsed:.2f} seconds')
```

### Prediction and Derivative Analysis

```python
# Predict solution over full domain
yh = model(x.to(device))
y_true = f_BC(x)

# Compute boundary condition error
bc_error = model.lossBC(x.to(device))
print(f'\nBoundary Condition Loss: {bc_error.item():.6e}')

# Compute derivative dy/dx for validation
g = x.to(device).clone()
g.requires_grad = True

# Forward pass
f = model(g)

# Compute dy/dx
f_x = autograd.grad(f, g, torch.ones([g.shape[0], 1]).to(device),
                    retain_graph=True, create_graph=True)[0]

# Move to CPU for plotting
y_pred = yh.detach().cpu().numpy()
y_exact = y_true.detach().numpy()
derivative = f_x.detach().cpu().numpy()

# Compute relative L2 error
error_L2 = np.linalg.norm(y_exact - y_pred, 2) / np.linalg.norm(y_exact, 2)
print(f'Relative L2 Error: {error_L2:.6e}')

# Plot results
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(x, y_exact, color='blue', label='Exact: y = sin(x)')
ax.plot(x, y_pred, color='red', linestyle='--', label='PINN Prediction')
ax.plot(x, derivative, color='green', label='Derivative: dy/dx')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.set_title('ODE Solution: dy/dx = cos(x)')
plt.show()
```

## Critical Parameters

1. **Network architecture**
   - Layers: [1, 50, 50, 20, 50, 50, 1]
   - 5 hidden layers with varying widths (50-50-20-50-50)
   - Input: x coordinate (1D)
   - Output: y(x) solution value
   - Activation: Tanh (smooth, bounded)
   - Relatively deep for a simple ODE (demonstrates over-parameterization is acceptable)

2. **Training data**
   - Boundary points (Nu): 2 (only the two endpoints x=0 and x=2π)
   - Collocation points (Nf): 250
   - Total training points: 252
   - Sampling: Latin Hypercube Sampling for collocation points ensures uniform coverage

3. **Optimizer**
   - Single-stage Adam optimizer
   - Steps: 5,000
   - Learning rate: 0.001
   - No L-BFGS needed due to simplicity of ODE
   - amsgrad: False

4. **ODE-specific parameters**
   - Domain: x ∈ [0, 2π]
   - ODE: dy/dx = cos(x)
   - Boundary conditions: y(0) = 0, y(2π) = 0 (Dirichlet)
   - Exact solution: y(x) = sin(x)
   - Linear first-order ODE (analytically solvable)

5. **Initialization**
   - Xavier normal initialization for weights (gain=1.0)
   - Zero initialization for biases
   - Random seed: 123 for reproducibility

6. **Loss function**
   - MSE for both boundary condition loss and ODE residual loss
   - Total loss: L = L_BC + L_PDE
   - Equal weighting (both losses have similar scales)
   - No adaptive weights needed for this simple problem

7. **Expected accuracy**
   - Relative L2 error: typically < 1e-4
   - Very fast convergence due to linear nature of ODE
   - Smooth solution (sin(x)) is easy for neural network to approximate
   - Derivative dy/dx ≈ cos(x) is also well-approximated

8. **Key implementation detail**
   - Automatic differentiation computes exact derivatives (no finite difference approximation)
   - The derivative is computed as part of the loss function calculation
   - Both solution y(x) and its derivative dy/dx are learned implicitly
