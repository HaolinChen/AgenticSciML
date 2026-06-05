# NSFnets (Navier-Stokes Flow nets): Physics-Informed Neural Networks for Incompressible Navier-Stokes Equations

**Keywords**: [PDE, elliptic, nonlinear, forward-problem, steady-navier-stokes, 2D, dirichlet, PINN, MLP, strong-form, adam, lbfgs, mse, pytorch]

**Problem:** NSFnets employ physics-informed neural networks (PINNs) to simulate incompressible flows ranging from laminar to turbulent regimes. The method addresses the challenge of solving the incompressible Navier-Stokes equations without mesh generation, using automatic differentiation to handle differential operators. The code specifically implements steady 2D cavity flow at Reynolds number 2000, where the key challenge is enforcing both the momentum equations and the divergence-free constraint simultaneously.

**Issues addressed:** Manual weight tuning for loss function components (boundary conditions vs equation residuals), convergence difficulties in PINN training for fluid mechanics problems, and obtaining pressure field as a hidden state without requiring data or solving a separate Poisson equation.

## Key Method

NSFnets use neural networks to approximate solutions of the Navier-Stokes equations in velocity-pressure (VP) formulation. The spatial coordinates (x, y) are inputs, and the outputs are velocity components (u, v) and pressure (p). The method leverages automatic differentiation to compute all derivatives needed in the governing equations.

**Velocity-Pressure Formulation:**
The incompressible Navier-Stokes equations in VP form are:
- Momentum: ∂u/∂t + (u·∇)u = -∇p + (1/Re)∇²u
- Continuity: ∇·u = 0

For steady flow (∂u/∂t = 0), the equations simplify to:
- eq1 = u·∂u/∂x + v·∂u/∂y + ∂p/∂x - (1/Re)(∂²u/∂x² + ∂²u/∂y²) = 0
- eq2 = u·∂v/∂x + v·∂v/∂y + ∂p/∂y - (1/Re)(∂²v/∂x² + ∂²v/∂y²) = 0
- eq3 = ∂u/∂x + ∂v/∂y = 0

**Loss Function:**
The total loss combines boundary conditions (Lb) and equation residuals (Le):
- L = α·Lb + Le
- Lb = (1/Nb)Σ[|u - u_bc|² + |v - v_bc|²]
- Le = (1/Ne)Σ[|eq1|² + |eq2|² + |eq3|²]

where α is a weighting coefficient (typically 10-100) to balance the two loss components.

**Key Innovation:**
1. Pressure is treated as a hidden state obtained via the incompressibility constraint, without requiring boundary/initial conditions for pressure
2. Multi-stage training with progressively decreasing learning rates
3. Hybrid optimization: Adam followed by L-BFGS-B for fine-tuning
4. Dynamic weight adjustment strategies to balance loss components

## Implementation

### Network Architecture

```python
# Fully connected neural network with tanh activation
class FCNet(torch.nn.Module):
    def __init__(self, num_ins=2,      # Input: (x, y)
                 num_outs=3,            # Output: (u, v, p)
                 num_layers=4,          # Number of hidden layers
                 hidden_size=120,       # Neurons per layer
                 activation=torch.nn.Tanh):
        super(FCNet, self).__init__()

        # Build layer structure: [2, 120, 120, 120, 120, 3]
        layers = [num_ins] + [hidden_size] * num_layers + [num_outs]
        self.depth = len(layers) - 1
        self.activation = activation

        # Create layers with activation functions
        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        # Final layer (no activation)
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out
```

### Physics-Informed Loss Computation

```python
def neural_net_equations(self, x, y):
    """
    Compute Navier-Stokes equation residuals using automatic differentiation
    Args:
        x, y: spatial coordinates (require gradients)
    Returns:
        eq1, eq2, eq3: residuals of momentum and continuity equations
    """
    X = torch.cat((x, y), dim=1)
    uvpe = self.net(X)

    # Extract predicted fields
    u = uvpe[:, 0:1]  # x-velocity
    v = uvpe[:, 1:2]  # y-velocity
    p = uvpe[:, 2:3]  # pressure

    # Compute first derivatives using autograd
    u_x, u_y = self.autograd(u, [x, y])
    v_x, v_y = self.autograd(v, [x, y])
    p_x, p_y = self.autograd(p, [x, y])

    # Compute second derivatives
    u_xx = self.autograd(u_x, [x])[0]
    u_yy = self.autograd(u_y, [y])[0]
    v_xx = self.autograd(v_x, [x])[0]
    v_yy = self.autograd(v_y, [y])[0]

    # Navier-Stokes equations (steady, 2D)
    # Momentum equation in x-direction
    eq1 = (u*u_x + v*u_y) + p_x - 1.0/self.Re*(u_xx + u_yy)
    # Momentum equation in y-direction
    eq2 = (u*v_x + v*v_y) + p_y - 1.0/self.Re*(v_xx + v_yy)
    # Continuity equation (incompressibility)
    eq3 = u_x + v_y

    return eq1, eq2, eq3

@torch.jit.script
def autograd(y: torch.Tensor, x: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    TorchScript function to compute gradient of tensor y w.r.t. multiple inputs x
    Uses automatic differentiation to compute derivatives
    """
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y, device=y.device)]
    grad = torch.autograd.grad(
        [y,], x,
        grad_outputs=grad_outputs,
        create_graph=True,  # Allow computing higher-order derivatives
        allow_unused=True,
    )

    if grad is None:
        grad = [torch.zeros_like(xx) for xx in x]
    grad = [g if g is not None else torch.zeros_like(x[i]) for i, g in enumerate(grad)]
    return grad
```

### Loss Function and Training

```python
def fwd_computing_loss_2d(self, loss_mode='MSE'):
    """
    Compute total loss: boundary condition loss + equation residual loss
    """
    # Predict velocity at boundary points
    (self.u_pred_b, self.v_pred_b, _) = self.neural_net_u(self.x_b, self.y_b)

    # Boundary condition loss (MSE between prediction and BC)
    self.loss_b = torch.mean(torch.square(self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1]))) + \
                  torch.mean(torch.square(self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1])))

    # Compute equation residuals at collocation points
    (self.eq1_pred, self.eq2_pred, self.eq3_pred) = self.neural_net_equations(self.x_f, self.y_f)

    # Equation residual loss (mean squared residuals)
    self.loss_eq1 = torch.mean(torch.square(self.eq1_pred.reshape([-1])))
    self.loss_eq2 = torch.mean(torch.square(self.eq2_pred.reshape([-1])))
    self.loss_eq3 = torch.mean(torch.square(self.eq3_pred.reshape([-1])))
    self.loss_e = self.loss_eq1 + self.loss_eq2 + self.loss_eq3

    # Total weighted loss
    self.loss = self.alpha_b * self.loss_b + self.alpha_e * self.loss_e

    return self.loss, [self.loss_e, self.loss_b]

def solve_Adam(self, loss_func, num_epoch=1000, batchsize=None, scheduler=None):
    """
    Train network using Adam optimizer
    """
    epoch_id = 0
    with tqdm(initial=epoch_id, total=num_epoch) as pbar:
        while epoch_id < num_epoch:
            # Forward pass: compute loss
            loss, losses = loss_func()

            # Backward pass: compute gradients
            loss.backward()

            # Update parameters
            self.opt.step()
            self.opt.zero_grad()

            # Update learning rate if scheduler provided
            if scheduler:
                scheduler.step()

            # Save checkpoint every 10000 iterations
            if epoch_id % 10000 == 0:
                saved_ckpt = 'model_cavity_loop_%d.pth'%(epoch_id)
                self.save(saved_ckpt, N_HLayer=self.layers,
                         N_neu=self.hidden_size, N_f=self.N_f)

            epoch_id = epoch_id + 1
```

### Training Procedure

```python
def train(net_params=None):
    """
    Multi-stage training for 2D cavity flow at Re=2000
    """
    Re = 2000              # Reynolds number
    N_neu = 120            # Neurons per hidden layer
    N_HLayer = 4           # Number of hidden layers
    N_f = 40000            # Number of collocation points for equations
    lam_bcs = 10           # Weight for boundary condition loss
    lam_equ = 1            # Weight for equation residual loss

    # Initialize PINN solver
    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=Re,
        layers=N_HLayer,
        hidden_size=N_neu,
        N_f=N_f,
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
        num_ins=2,   # Input: (x, y)
        num_outs=3,  # Output: (u, v, p)
        net_params=net_params,
        checkpoint_path='./checkpoint/')

    # Load boundary and collocation point data
    dataloader = cavity.DataLoader(path='./datasets/', N_f=N_f, N_b=1000)
    boundary_data = dataloader.loading_boundary_data()
    PINN.set_boundary_data(X=boundary_data)
    training_data = dataloader.loading_training_data()
    PINN.set_eq_training_data(X=training_data)

    # Load reference data for evaluation
    filename = './data/cavity_Re'+str(Re)+'_256.mat'
    x_star, y_star, u_star, v_star = dataloader.loading_evaluate_data(filename)

    # Multi-stage training with decreasing learning rates
    PINN.set_stage(1)
    PINN.train(num_epoch=200000, lr=1e-3)
    PINN.evaluate(x_star, y_star, u_star, v_star)

    PINN.set_stage(2)
    PINN.train(num_epoch=200000, lr=2e-4)
    PINN.evaluate(x_star, y_star, u_star, v_star)

    PINN.set_stage(3)
    PINN.train(num_epoch=200000, lr=5e-5)
    PINN.evaluate(x_star, y_star, u_star, v_star)

    PINN.set_stage(4)
    PINN.train(num_epoch=500000, lr=1e-5)
    PINN.evaluate(x_star, y_star, u_star, v_star)

    PINN.set_stage(5)
    PINN.train(num_epoch=500000, lr=2e-6)
    PINN.evaluate(x_star, y_star, u_star, v_star)
```

## Critical Parameters

1. **Network Architecture:**
   - Hidden layers: 4 (larger networks improve accuracy)
   - Neurons per layer: 120
   - Activation: tanh (smooth, differentiable)
   - Input dimension: 2 (x, y)
   - Output dimension: 3 (u, v, p)

2. **Training Data:**
   - Boundary points (N_b): 1000
   - Collocation points (N_f): 40000 (sampled using Latin Hypercube Sampling)
   - Reynolds number: 2000

3. **Loss Weights:**
   - Boundary weight (lam_bcs): 10
   - Equation weight (lam_equ): 1
   - Ratio controls relative importance of satisfying BCs vs PDEs

4. **Optimization:**
   - Optimizer: Adam (adaptive learning rate method)
   - Multi-stage learning rates: 1e-3 → 2e-4 → 5e-5 → 1e-5 → 2e-6
   - Total epochs: 1.6 million across 5 stages
   - Optional: L-BFGS-B for fine-tuning after Adam

5. **Boundary Conditions:**
   - Lid-driven cavity: top wall moves with velocity u = 1 - cosh(r(x-0.5))/cosh(0.5r) where r=10
   - All other walls: no-slip (u=v=0)
   - Domain: [0,1] × [0,1]

6. **Key Implementation Details:**
   - Automatic differentiation: create_graph=True enables higher-order derivatives
   - Checkpoint frequency: save model every 10000 iterations
   - Evaluation: compute relative L2 error against reference DNS/analytical solution
