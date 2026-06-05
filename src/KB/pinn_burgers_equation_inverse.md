# Physics-Informed Neural Networks for Burgers Equation (Inverse Problem)

**Keywords**: [PDE, nonlinear, parabolic, inverse-problem, parameter-estimation, burgers, 1D, periodic, PINN, MLP, strong-form, adam, lbfgs, mse, pytorch]

**Problem:** The inverse problem for PINNs involves discovering unknown parameters in the governing equations from observed data. For the Burgers equation, this means learning the coefficients λ₁ (for the nonlinear convection term) and λ₂ (for the diffusion term) from solution measurements. This addresses the challenge of system identification when the physical model structure is known but parameters are uncertain. The method simultaneously learns both the solution field u(x,t) and the unknown parameters λ₁, λ₂.

**Issues addressed:**
- Parameter estimation from sparse observational data
- System identification for nonlinear PDEs
- Simultaneous learning of solution and unknown coefficients
- Handling incomplete knowledge of physical parameters

## Key Method

The inverse PINN represents both the solution u(x,t) and unknown parameters λ as trainable quantities. The network minimizes:

1. **Data loss**: Match observed measurements of u(x,t)
2. **Physics loss**: Satisfy the PDE residual with learned parameters

For the Burgers equation with unknown parameters:
**∂u/∂t + λ₁·u·∂u/∂x = λ₂·∂²u/∂x²**

where:
- λ₁ and λ₂ are trainable parameters to be discovered
- True values: λ₁ = 1.0, λ₂ = 0.01/π ≈ 0.00318
- Initial guesses: λ₁ = 2.0, λ₂ = 0.2

**Key advantages:**
- Learns unknown PDE parameters from data
- Robust to noise in observations
- Regularizes parameter estimation through physics constraints
- Provides both solution and discovered parameters

## Implementation

### Core Network Class with Trainable Parameters

```python
class FCN(nn.Module):
    """
    Fully Connected Network for Inverse PINN
    Includes trainable parameters lambda_1 and lambda_2
    """
    def __init__(self, layers):
        """
        layers: network architecture [2, 20, 20, ..., 1]
        """
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')

        # Neural network layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1])
                                      for i in range(len(layers)-1)])

        # Trainable PDE parameters (registered as nn.Parameter)
        # Initial guesses for λ₁ and λ₂
        self.lambda_1 = nn.Parameter(torch.tensor([2.0]))  # Initial guess
        self.lambda_2 = nn.Parameter(torch.tensor([0.2]))  # Initial guess

        self.iter = 0

        # Xavier initialization
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

### Loss Functions for Inverse Problem

```python
def loss_BC(self, x_BC, y_BC):
    """
    Data loss: match observed measurements
    x_BC: observation points [x, t]
    y_BC: observed values of u(x,t)
    """
    loss_BC = self.loss_function(self.forward(x_BC), y_BC)
    return loss_BC

def loss_PDE(self, X_train_Nf):
    """
    Physics loss: PDE residual with learned parameters
    Burgers equation: u_t + λ₁*u*u_x - λ₂*u_xx = 0

    X_train_Nf: collocation points for PDE residual evaluation
    """
    g = X_train_Nf.clone()
    g.requires_grad = True

    u = self.forward(g)

    # First-order derivatives
    u_x_t = autograd.grad(u, g, torch.ones([X_train_Nf.shape[0], 1]).to(device),
                          retain_graph=True, create_graph=True)[0]

    # Second-order derivatives
    u_xx_tt = autograd.grad(u_x_t, g, torch.ones(X_train_Nf.shape).to(device),
                            create_graph=True)[0]

    u_x = u_x_t[:, [0]]   # ∂u/∂x
    u_t = u_x_t[:, [1]]   # ∂u/∂t
    u_xx = u_xx_tt[:, [0]] # ∂²u/∂x²

    # Burgers equation residual with trainable parameters λ₁ and λ₂
    # f = u_t + λ₁*u*u_x - λ₂*u_xx
    f = u_t + self.lambda_1 * (self.forward(g)) * u_x - self.lambda_2 * u_xx

    return self.loss_function(f, f_hat)

def loss(self, x_BC, y_BC, X_train_Nf):
    """
    Total loss = data loss + PDE residual loss
    Both neural network weights and λ parameters are optimized
    """
    loss_bc = self.loss_BC(x_BC, y_BC)
    loss_pde = self.loss_PDE(X_train_Nf)
    return loss_bc + loss_pde
```

### Data Preparation for Inverse Problem

```python
# Domain and true parameters
min_x, max_x = -1, 1
min_t, max_t = 0, 1
total_points_x = 200
total_points_t = 100

# True parameter values (unknown to the model)
lambda_1_true = 1.0
lambda_2_true = 0.01/np.pi

# Training data
Nu = 100    # Number of observation points
Nf = 10000  # Number of collocation points

# Create grid
x = torch.linspace(min_x, max_x, total_points_x).view(-1, 1)
t = torch.linspace(min_t, max_t, total_points_t).view(-1, 1)
X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1), indexing='ij')

# Load true solution (from simulation or experiment)
usol = torch.load('burgers_solution.pt')

# Observation points (sparse measurements)
X_train = torch.hstack((X.transpose(1, 0).flatten()[:, None],
                        T.transpose(1, 0).flatten()[:, None]))
u_train = usol.transpose(1, 0).flatten()[:, None]

# Randomly select Nu observation points
idx_obs = np.random.choice(X_train.shape[0], Nu, replace=False)
X_train_Nu = X_train[idx_obs, :]
u_train_Nu = u_train[idx_obs, :]

# Collocation points for PDE residual
idx_collocation = np.random.choice(X_train.shape[0], Nf, replace=False)
X_train_Nf = X_train[idx_collocation, :]
```

### Training Loop with Parameter Discovery

```python
# Network architecture
layers = np.array([2, 20, 20, 20, 20, 20, 20, 20, 20, 1])

# Create model (with initial parameter guesses λ₁=2.0, λ₂=0.2)
torch.manual_seed(123)
model = FCN(layers)
model.to(device)

# Move data to device
X_train_Nu = X_train_Nu.to(device)
u_train_Nu = u_train_Nu.to(device)
X_train_Nf = X_train_Nf.to(device)

# Training parameters
steps_adam = 10000
lr = 1e-1

# Stage 1: Adam optimizer (optimizes both network weights and λ parameters)
optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)

print(f'Initial guesses: λ₁={model.lambda_1.item():.6f}, λ₂={model.lambda_2.item():.6f}')

for i in range(steps_adam):
    loss = model.loss(X_train_Nu, u_train_Nu, X_train_Nf)
    optimizer_adam.zero_grad()
    loss.backward()
    optimizer_adam.step()

    if i % (steps_adam/10) == 0:
        print(f'Step {i}, Loss: {loss.item():.6e}, '
              f'λ₁={model.lambda_1.item():.6f}, λ₂={model.lambda_2.item():.6f}')

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
    loss = model.loss(X_train_Nu, u_train_Nu, X_train_Nf)
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)

# Final discovered parameters
print(f'\nDiscovered parameters:')
print(f'λ₁ = {model.lambda_1.item():.6f} (true: {lambda_1_true:.6f})')
print(f'λ₂ = {model.lambda_2.item():.6f} (true: {lambda_2_true:.6f})')
```

### Parameter Error Analysis

```python
# Compute parameter errors
lambda_1_error = abs(model.lambda_1.item() - lambda_1_true) / lambda_1_true * 100
lambda_2_error = abs(model.lambda_2.item() - lambda_2_true) / lambda_2_true * 100

print(f'\nParameter identification errors:')
print(f'λ₁ error: {lambda_1_error:.2f}%')
print(f'λ₂ error: {lambda_2_error:.2f}%')

# Solution error
u_pred = model(X_train.to(device))
u_exact = u_train.to(device)
solution_error = torch.linalg.norm(u_exact - u_pred, 2) / torch.linalg.norm(u_exact, 2)
print(f'Relative L2 solution error: {solution_error.item():.6e}')
```

## Critical Parameters

1. **Network architecture**
   - Layers: [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
   - 8 hidden layers with 20 neurons each
   - Input: (x, t) coordinates
   - Output: u(x, t) velocity field
   - Activation: Tanh

2. **Trainable PDE parameters**
   - λ₁: coefficient for convection term u·∂u/∂x
     - Initial guess: 2.0
     - True value: 1.0
     - Registered as nn.Parameter for joint optimization
   - λ₂: coefficient for diffusion term ∂²u/∂x²
     - Initial guess: 0.2
     - True value: 0.01/π ≈ 0.00318
     - Registered as nn.Parameter for joint optimization

3. **Training data**
   - Observation points (Nu): 100 (sparse measurements of u(x,t))
   - Collocation points (Nf): 10,000 (for PDE residual)
   - Both solution and parameters learned from this data

4. **Optimizer sequence**
   - Stage 1: Adam optimizer
     - Steps: 10,000
     - Learning rate: 0.1
     - Optimizes both network weights and λ parameters
   - Stage 2: L-BFGS optimizer
     - Max iterations: 500
     - Fine-tunes all parameters simultaneously

5. **Loss function components**
   - Data loss (L_BC): MSE between predictions and observations
   - Physics loss (L_PDE): MSE of PDE residual with learned parameters
   - Total loss: L = L_BC + L_PDE
   - Both losses depend on the learned λ parameters

6. **Initial conditions**
   - Parameter initial guesses are critical for convergence
   - Too far from true values may lead to local minima
   - Xavier initialization for network weights
   - Manual initialization for λ parameters based on physical intuition

7. **Convergence monitoring**
   - Track loss values during training
   - Monitor λ₁ and λ₂ evolution to ensure convergence to true values
   - Compare discovered parameters to true values for validation
