# Entropy-Viscosity Regularized NSFnet (ev-NSFnet) for Stable PINN Solutions

**Keywords**: ["PDE", "navier-stokes", "parabolic", "hyperbolic", "nonlinear", "forward-problem", "2D", "dirichlet", "not-converging", "stability", "PINN", "MLP", "self-adaptive", "strong-form", "adam", "mse", "gpu", "pytorch"]

**Problem:** Physics-Informed Neural Networks (PINNs) applied to incompressible Navier-Stokes equations can converge to multiple solutions, including spurious ones that do not match DNS reference data. At moderate to high Reynolds numbers (Re=2000-5000), standard NSFnets (Navier-Stokes Flow nets using velocity-pressure formulation) suffer from non-uniqueness: different random initializations yield different flow solutions, some being physically unstable or incorrect. This presents a significant challenge when PINNs find solutions in classes that DNS does not capture, making the method unreliable for turbulent or high-Re flows.

**Issues addressed:** Non-convergence to correct solution, stability issues during training, spurious solutions from different initializations, non-uniqueness in solution space for high Reynolds number flows.

## Key Method

The ev-NSFnet introduces an **entropy-viscosity regularization** technique to stabilize PINN training for the incompressible Navier-Stokes equations. The method employs two neural networks:

1. **Main network (net)**: Predicts velocity (u, v) and pressure (p)
2. **Auxiliary network (net_1)**: Predicts an entropy-like residual field (e)

The key innovation is the introduction of an **eddy viscosity term** computed adaptively from the entropy residual:

- **Entropy residual**: e = (eq1·(u-0.5) + eq2·(v-0.5)) - e_predicted, where eq1 and eq2 are momentum equation residuals
- **Eddy viscosity**: ν_t = min(ν_t0, α_evm · |e|), where ν_t0 = 20/Re
- **Modified momentum equations**: Include (1/Re + ν_t) instead of just 1/Re in the diffusion terms

The eddy viscosity acts as a stabilizing regularization that suppresses spurious oscillations and guides the solution toward the DNS-matching stable state. The auxiliary network is trained intermittently (every 10,000 iterations) and frozen otherwise, allowing the main network to adapt to the regularized problem progressively.

By progressively reducing α_evm (from 0.05 → 0.002) during multi-stage training, the method gradually removes regularization, converging to the true Navier-Stokes solution while maintaining stability.

## Implementation

```python
# Dual-network architecture: main network for (u,v,p), auxiliary network for entropy residual (e)
class PysicsInformedNeuralNetwork:
    def __init__(self, Re=5000, layers=4, layers_1=4, hidden_size=120,
                 hidden_size_1=40, N_f=120000, alpha_evm=0.03):

        self.Re = Re
        self.vis_t0 = 20.0/self.Re  # Reference eddy viscosity threshold
        self.alpha_evm = alpha_evm  # Entropy-viscosity coefficient

        # Main network: (x,y) → (u, v, p)
        self.net = self.initialize_NN(
            num_ins=2, num_outs=3, num_layers=layers, hidden_size=hidden_size)

        # Auxiliary network: (x,y) → (e), predicts entropy-like residual
        self.net_1 = self.initialize_NN(
            num_ins=2, num_outs=1, num_layers=layers_1, hidden_size=hidden_size_1)

        # Adam optimizer for both networks
        self.opt = torch.optim.Adam(
            list(self.net.parameters())+list(self.net_1.parameters()),
            lr=learning_rate, weight_decay=0.0)
```

```python
# Neural network predictions
def neural_net_u(self, x, y):
    X = torch.cat((x, y), dim=1)
    uvp = self.net(X)       # Main network output
    ee = self.net_1(X)      # Auxiliary network output
    u = uvp[:, 0]
    v = uvp[:, 1]
    p = uvp[:, 2:3]
    e = ee[:, 0:1]          # Entropy residual
    return u, v, p, e
```

```python
# Modified Navier-Stokes equations with entropy-viscosity regularization
def neural_net_equations(self, x, y):
    X = torch.cat((x, y), dim=1)
    uvp = self.net(X)
    ee = self.net_1(X)

    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]
    e = ee[:, 0:1]
    self.evm = e

    # Automatic differentiation for spatial derivatives
    u_x, u_y = self.autograd(u, [x,y])
    u_xx = self.autograd(u_x, [x])[0]
    u_yy = self.autograd(u_y, [y])[0]

    v_x, v_y = self.autograd(v, [x,y])
    v_xx = self.autograd(v_x, [x])[0]
    v_yy = self.autograd(v_y, [y])[0]

    p_x, p_y = self.autograd(p, [x,y])

    # Compute adaptive eddy viscosity: min(vis_t0, alpha_evm * |e_previous|)
    self.vis_t = torch.tensor(
        np.minimum(self.vis_t0, self.vis_t_minus)).float().to(device)

    # Update vis_t_minus for next iteration using current entropy residual
    self.vis_t_minus = self.alpha_evm * torch.abs(e).detach().cpu().numpy()

    # Regularized Navier-Stokes momentum equations with eddy viscosity
    # Standard viscosity (1/Re) + adaptive eddy viscosity (vis_t)
    eq1 = (u*u_x + v*u_y) + p_x - (1.0/self.Re + self.vis_t)*(u_xx + u_yy)
    eq2 = (u*v_x + v*v_y) + p_y - (1.0/self.Re + self.vis_t)*(v_xx + v_yy)
    eq3 = u_x + v_y  # Continuity equation (incompressibility)

    # Entropy residual: measures energy dissipation vs. predicted residual
    residual = (eq1*(u-0.5) + eq2*(v-0.5)) - e
    return eq1, eq2, eq3, residual
```

```python
# Loss function: MSE of PDE residuals + entropy regularization
def fwd_computing_loss_2d(self, loss_mode='MSE'):
    # Boundary condition loss
    (self.u_pred_b, self.v_pred_b, _, _) = self.neural_net_u(self.x_b, self.y_b)
    self.loss_b = torch.mean(torch.square(self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1]))) + \
                  torch.mean(torch.square(self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1])))

    # PDE residual loss (momentum + continuity + entropy residual)
    (self.eq1_pred, self.eq2_pred, self.eq3_pred, self.eq4_pred) = \
        self.neural_net_equations(self.x_f, self.y_f)

    self.loss_eq1 = torch.mean(torch.square(self.eq1_pred.reshape([-1])))  # x-momentum
    self.loss_eq2 = torch.mean(torch.square(self.eq2_pred.reshape([-1])))  # y-momentum
    self.loss_eq3 = torch.mean(torch.square(self.eq3_pred.reshape([-1])))  # continuity
    self.loss_eq4 = torch.mean(torch.square(self.eq4_pred.reshape([-1])))  # entropy residual

    # Total equation loss with weighted entropy term
    self.loss_e = self.loss_eq1 + self.loss_eq2 + self.loss_eq3 + 0.1*self.loss_eq4

    # Total loss: boundary + weighted equation loss
    self.loss = self.alpha_b * self.loss_b + self.alpha_e * self.loss_e
    return self.loss, [self.loss_e, self.loss_b]
```

```python
# Alternating training strategy: freeze/unfreeze auxiliary network
def solve_Adam(self, loss_func, num_epoch=1000):
    self.freeze_evm_net(0)  # Start with auxiliary network frozen

    for epoch_id in range(num_epoch):
        # Unfreeze auxiliary network every 10,000 iterations
        if epoch_id != 0 and epoch_id % 10000 == 0:
            self.defreeze_evm_net(epoch_id)
        # Freeze auxiliary network after 1 iteration of joint training
        if (epoch_id - 1) % 10000 == 0:
            self.freeze_evm_net(epoch_id)

        loss, losses = loss_func()
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

def freeze_evm_net(self, epoch_id):
    # Freeze auxiliary network parameters, train only main network
    for para in self.net_1.parameters():
        para.requires_grad = False
    self.opt.param_groups[0]['params'] = list(self.net.parameters())

def defreeze_evm_net(self, epoch_id):
    # Unfreeze auxiliary network, train both networks jointly
    for para in self.net_1.parameters():
        para.requires_grad = True
    self.opt.param_groups[0]['params'] = list(self.net.parameters()) + \
                                          list(self.net_1.parameters())
```

```python
# Multi-stage training with progressively reduced entropy-viscosity coefficient
def train(net_params=None):
    Re = 5000
    N_neu = 120        # Neurons per layer in main network
    N_neu_1 = 40       # Neurons per layer in auxiliary network
    N_HLayer = 4       # Hidden layers in main network
    N_HLayer_1 = 4     # Hidden layers in auxiliary network
    N_f = 120000       # Number of collocation points
    lam_bcs = 10       # Boundary condition weight
    lam_equ = 1        # Equation residual weight

    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=Re, layers=N_HLayer, layers_1=N_HLayer_1,
        hidden_size=N_neu, hidden_size_1=N_neu_1, N_f=N_f,
        alpha_evm=0.05, bc_weight=lam_bcs, eq_weight=lam_equ)

    # Stage 1: alpha_evm = 0.05, lr = 1e-3
    PINN.set_alpha_evm(0.05)
    PINN.train(num_epoch=500000, lr=1e-3)

    # Stage 2: alpha_evm = 0.03, lr = 2e-4
    PINN.set_alpha_evm(0.03)
    PINN.train(num_epoch=500000, lr=2e-4)

    # Stage 3: alpha_evm = 0.02, lr = 5e-5
    PINN.set_alpha_evm(0.02)
    PINN.train(num_epoch=500000, lr=5e-5)

    # Stage 4: alpha_evm = 0.01, lr = 5e-5
    PINN.set_alpha_evm(0.01)
    PINN.train(num_epoch=500000, lr=5e-5)

    # Stage 5: alpha_evm = 0.005, lr = 1e-5
    PINN.set_alpha_evm(0.005)
    PINN.train(num_epoch=500000, lr=1e-5)

    # Stage 6: alpha_evm = 0.002, lr = 2e-6
    PINN.set_alpha_evm(0.002)
    PINN.train(num_epoch=500000, lr=2e-6)
```

## Critical Parameters

- **Reynolds number (Re)**: 2000-5000 for cavity flow test case
- **Entropy-viscosity coefficient (alpha_evm)**: Progressively reduced from 0.05 → 0.002 across training stages
- **Reference eddy viscosity (vis_t0)**: 20/Re, provides upper bound for stabilization
- **Network architecture**:
  - Main network: 4 hidden layers × 120 neurons
  - Auxiliary network: 4 hidden layers × 40 neurons
- **Training points**: 120,000-200,000 collocation points, 1000 boundary points
- **Loss weights**:
  - Boundary condition weight (lam_bcs): 10
  - Equation residual weight (lam_equ): 1
  - Entropy residual weight: 0.1 (in equation loss)
- **Training schedule**: 500,000 epochs per stage, 6 stages total (3 million epochs)
- **Learning rates**: Progressively reduced from 1e-3 → 2e-6 across stages
- **Auxiliary network update frequency**: Every 10,000 iterations (alternating freeze/unfreeze)
- **Optimizer**: Adam
- **Activation function**: Tanh
