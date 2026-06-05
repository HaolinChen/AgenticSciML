# Delta-PINNs: Physics-Informed Neural Networks on Complex Geometries

**Keywords**: ["PDE", "elliptic", "parabolic", "forward-problem", "inverse-problem", "eikonal", "poisson", "heat", "2D", "3D", "dirichlet", "neumann", "robin", "irregular", "complex-geometry", "PINN", "finite_element", "MLP", "strong-form", "adam", "mse", "jax"]

**Problem:** Traditional PINNs struggle on complex geometric domains because they use Cartesian coordinates as input, which fails to capture the intrinsic topology of the domain. When points are close in Euclidean space but far apart in geodesic distance on the manifold (e.g., on a coil or bunny surface), standard PINNs cannot distinguish them and produce poor approximations. This limits PINNs to simple rectangular domains and hinders their applicability to real-world problems with complex shapes.

**Issues addressed:**
- Failure of traditional PINNs on complex geometries where Euclidean and geodesic distances differ significantly
- Inability to represent solutions on manifolds or surfaces embedded in 3D
- Poor performance on domains with intricate topologies (coils, heat sinks, curved surfaces)
- Difficulty in ensuring gradients remain tangent to manifolds

## Key Method

Delta-PINNs replaces the Cartesian coordinate input to the neural network with eigenfunctions of the Laplace-Beltrami operator. The key innovations are:

1. **Positional Encoding via Eigenfunctions**: Instead of using coordinates x as input, the method uses v(x) = [v₁(x), v₂(x), ..., vₙ(x)], where vᵢ are the N eigenfunctions associated with the N lowest eigenvalues of the Laplace-Beltrami operator: -Δₛvᵢ(x) = λᵢvᵢ(x).

2. **Geometry-Aware Representation**: The eigenfunctions encode both topological and geometric information about the domain. Points close in geodesic distance have similar eigenfunction values, ensuring the neural network respects the intrinsic geometry of the manifold.

3. **Finite Element Operators**: Since eigenfunctions are computed numerically via finite elements, differential operators (gradient, Laplacian) in the PDE loss are also evaluated using finite element approximations rather than automatic differentiation. For a piecewise-linear interpolant of the neural network predictions, the gradient is computed as: ∇ₓu|ₑ ≈ Bᵉ · [u₁, u₂, u₃]ᵀ, where Bᵉ is the gradient operator matrix for triangular element e.

4. **Physics-Informed Loss**: The loss function combines data fitting, PDE residual, and boundary conditions:
   - MSEdata: Fit sparse observations
   - MSEPDE: Enforce PDE using finite element operators on random element batches
   - MSEb: Satisfy boundary conditions (when applicable)

The method works for both 2D domains and 3D surfaces/manifolds, handling Dirichlet, Neumann, and Robin boundary conditions.

## Implementation

### Computing Laplace-Beltrami Eigenfunctions

```python
from scipy.linalg import eigh

# Given mesh with vertices (verts) and connectivity
m = Mesh(verts=verts_new, connectivity=connectivity)

# Compute stiffness and mass matrices via finite elements
K, M = m.computeLaplacian()  # K = ∫∇Nᵢ·∇Nⱼ dB, M = ∫NᵢNⱼ dB

# Solve generalized eigenvalue problem: K v = λ M v
eigvals, eigvecs = eigh(K, M)

# Select first n_eigs eigenfunctions as neural network input
n_eigs = 50
eigfuncs = eigvecs[:, :n_eigs]
```

### Delta-PINN Model for Eikonal Equation

```python
import jax.numpy as np
from jax import grad, jit, vmap
from jax.tree_util import Partial as partial

class LaplacePINN(PINN):
    def __init__(self, eigenfuncs, connectivity, mesh_operator, norm_const, mu_X=0.0, sigma_X=1.0):
        super().__init__(mu_X, sigma_X)
        self.eigenfuncs = np.array(eigenfuncs)  # Shape: (n_nodes, n_eigs)
        self.connectivity = connectivity         # Triangular mesh connectivity
        self.mesh_operator = mesh_operator       # Finite element operators (B matrices)
        self.mu_y = norm_const['mu_y']
        self.sigma_y = norm_const['sigma_y']
        self.num_loss_terms = 2

    # Neural network takes eigenfunctions as input, outputs scalar u
    def net_u(self, params, x):
        inputs = np.stack([x])  # x is eigenfunction vector at a point
        u = self.net_apply(params, inputs)
        return u[0]

    # Compute Eikonal norm: ||∇u|| using finite element gradient operator
    @partial(jit, static_argnums=(0))
    def eiknorm(self, BB, u):
        # BB is B^T B matrix for an element, u is nodal values
        return np.sqrt(np.dot(u, np.dot(BB, u)))

    # Data loss: fit sparse observations
    @partial(jit, static_argnums=(0,))
    def loss_u(self, params, batch):
        inputs, targets = batch
        X, _, _ = inputs
        Y, _ = targets
        u_fn = lambda x: self.net_u(params, x)
        u_pred = vmap(u_fn)(X)
        loss_u = np.mean((Y - u_pred)**2)
        return loss_u

    # PDE residual loss: enforce Eikonal equation ||∇u|| = 1
    @partial(jit, static_argnums=(0,))
    def loss_r(self, params, batch):
        inputs, targets = batch
        _, X_res, Bs = inputs  # X_res: eigenfunction values, Bs: FE operators
        _, Y_res = targets

        u_fn = lambda x: self.net_u(params, x)
        yc = vmap(u_fn)(X_res).reshape((-1, 3))  # Predictions at element nodes
        # Compute ||∇u|| for each element using finite element operator
        res = vmap(self.eiknorm)(Bs, yc * self.sigma_y + self.mu_y)
        loss_r = np.mean((Y_res - res[:, None])**2)
        return loss_r

    # Combined loss with weights
    def loss(self, params, batch, weights=(1.0, 1.0)):
        w_u, w_r = weights
        loss_u = self.loss_u(params, batch)
        loss_r = self.loss_r(params, batch)
        loss = w_u * loss_u + w_r * loss_r
        return loss

    # Predict at test points
    @partial(jit, static_argnums=(0,))
    def predict(self, params, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        u_fn = lambda x: self.net_u(params, x)
        u_star = vmap(u_fn)(X_star)
        return u_star
```

### Finite Element Gradient Operator

```python
# Compute gradient operator matrix B^e for triangular element
# For element e with nodes at positions (x1,y1), (x2,y2), (x3,y3):
def Bmatrix(element_index):
    # Get node coordinates for this element
    x1, y1 = verts[connectivity[element_index, 0]]
    x2, y2 = verts[connectivity[element_index, 1]]
    x3, y3 = verts[connectivity[element_index, 2]]

    # Element area
    A_e = (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2

    # Gradient operator matrix
    B_e = (1/(2*A_e)) * np.array([
        [y2-y3, y3-y1, y1-y2],
        [x3-x2, x1-x3, x2-x1]
    ])

    # For Eikonal equation, we need B^T B to compute ||∇u||²
    return B_e.T @ B_e / (2*A_e)**2
```

### Data Generator for Mini-Batch Training

```python
class LaplacePINNGenerator(data.Dataset):
    def __init__(self, X, Y, eigenfuncs, connectivity, mesh_operator,
                 batch_size=1, rng_key=random.PRNGKey(1234)):
        self.X = X  # Eigenfunction values at data points
        self.Y = Y  # Observed values
        self.eigenfuncs = eigenfuncs  # All eigenfunction values (n_nodes, n_eigs)
        self.connectivity = connectivity  # Mesh connectivity
        self.mesh_operator = mesh_operator  # B matrices for all elements
        self.batch_size = batch_size
        self.key = rng_key

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        # Sample random data points and random elements
        idx = random.choice(key, self.X.shape[0], (self.batch_size,), replace=False)
        idx_e = random.choice(key, self.mesh_operator.shape[0], (self.batch_size,), replace=False)

        # Get eigenfunction values at element nodes
        Xc = self.eigenfuncs[np.ravel(self.connectivity[idx_e]), :]

        inputs = (self.X[idx], Xc, self.mesh_operator[idx_e])
        outputs = (self.Y[idx], np.ones((self.batch_size, 1)))
        return inputs, outputs
```

### Training Setup

```python
# Setup neural network architecture
from jaxpinns.architectures import MLP
from jaxpinns.optimizers import adam
from jax.example_libraries import optimizers

n_eigs = 50
layers = [n_eigs, 100, 1]  # Input: eigenfunctions, Hidden: 100 neurons, Output: scalar
model.architecture(MLP, layers, init_key=random.PRNGKey(0))

# Setup optimizer with exponential learning rate decay
learning_rate = optimizers.exponential_decay(1e-3, decay_steps=100, decay_rate=1.0)
model.optimizer(adam, learning_rate, model.loss)

# Train for 40,000 iterations with batch size 10
dataset = LaplacePINNGenerator(eigfuncs[idx_train, :], Y, eigfuncs,
                                m.connectivity, Bs, batch_size=10)
model.train(dataset, nIter=40000, ntk_weights=False)
```

### Heat Transfer Example with Laplacian Operator

```python
# For Laplace equation: -Δu = 0
# The Laplacian at node i is computed using the discrete operator L = M^(-1)K

@partial(jit, static_argnums=(0,))
def loss_r(self, params, batch):
    inputs, targets = batch
    _, V_res, vals, segments = inputs
    _, Y_res = targets

    # Predict at collocation nodes
    u_res = vmap(self.net_u, in_axes=(None, 0))(params, V_res) * self.sigma_y + self.mu_y

    # Compute Laplacian using sparse matrix multiplication (segment_sum for JAX)
    # This implements: Δu_i ≈ Σⱼ Lᵢⱼ uⱼ where L = M^(-1)K
    res = segment_sum(u_res * vals, segments, num_segments)

    loss_r = np.sum(res**2) / (segments[-1] + 1)
    return loss_r
```

## Critical Parameters

1. **Number of eigenfunctions (n_eigs)**: Controls the representation capacity
   - Lower frequencies (small n): Smoother solutions, less prone to overfitting with sparse data
   - Higher frequencies (large n): Can represent more complex solutions, requires more data
   - Typical range: 25-100 for complex geometries
   - Sensitivity: Performance degrades with too few (<10) or too many (>400)

2. **Mesh discretization size**: Affects accuracy of finite element operators
   - Element size: Optimal range 0.066-0.1 for normalized domains
   - Too coarse: Operators not accurately approximated
   - Too fine: MSEPDE term becomes less important during training
   - Trade-off between computational cost and accuracy

3. **Neural network architecture**:
   - Hidden layers: 1-3 layers typically sufficient
   - Neurons per layer: 100-200 neurons
   - Activation: Hyperbolic tangent (tanh) commonly used

4. **Training hyperparameters**:
   - Batch size: 10-30 for data term, 10-1000 for residual term
   - Iterations: 40,000-50,000 for convergence
   - Learning rate: 1e-3 with exponential decay (decay_rate 0.99-1.0)
   - Loss weights: Balance MSEdata and MSEPDE (typically equal weights 1:1)

5. **Preprocessing**:
   - Eigenfunction computation time: ~0.9s for 1,546 nodes, ~35s for 134,345 nodes
   - Normalize coordinates and eigenfunction values for stable training
   - Use homogeneous Neumann boundary conditions for eigenfunction computation
