# DeepONet for 2D Darcy Flow with Rectangular Piecewise Constant Permeability

**Keywords**: [PDE, elliptic, linear, forward-problem, darcy, 2D, regular, dirichlet, DeepONet, MLP, adam, mse, relative-l2, deepxde, tensorflow]

**Problem:** Learning the solution operator for the 2D Darcy flow equation:
```
-∇·(a(x,y)∇u(x,y)) = f,  (x,y) ∈ [0,1]², u = 0 on ∂Ω
```
where a(x,y) is piecewise constant permeability from GRF, f is source term. Operator G maps a → u. Resolution: 421×421 downsampled to 29×29 (841 DOF). Homogeneous Dirichlet BC enforced via output transform. 1000 training, 200 test samples.

**Issues addressed:**
- Expensive repeated solves of elliptic PDEs with varying coefficients
- Need for operator learning on 2D structured grids
- Enforcement of Dirichlet boundary conditions in neural operator learning
- Generalization across different permeability realizations from Gaussian Random Fields

## Key Method

DeepONet learns G: a(x,y) → u(x,y) using:

1. **Branch Network**: Encodes flattened 2D permeability field a (29×29 grid)
2. **Trunk Network**: Encodes 2D spatial query locations (x,y)
3. **Dirichlet BC Enforcement**: Output transform that automatically satisfies u=0 on boundaries

**Architecture**: Branch [841, 128, 128, 128] flattens 29×29 permeability field. Trunk [2, 128, 128, 128] for (x,y) coordinates. Output dimension: 128 basis functions.

**Output Transform for BC**:
```
u_pred(x,y) = 20·x·(1-x)·y·(1-y)·(network_output + 1)
```
This vanishes at x=0, x=1, y=0, y=1 automatically, eliminating need for boundary loss terms.

**Data**:
- Permeability: Gaussian Random Field with piecewise constant structure
- Domain: [0,1] × [0,1]
- Grid: 421×421 downsampled to 29×29
- BC: Homogeneous Dirichlet (u=0 on boundary)

## Implementation

```python
import deepxde as dde
import numpy as np
from scipy import io
from deepxde.backend import tf

def get_data(filename, ndata):
    """
    Load and prepare Darcy flow data.

    Args:
        filename: MATLAB file with permeability and solution fields
        ndata: number of samples to use

    Returns:
        x: tuple of (x_branch, x_trunk) where:
            - x_branch: permeability fields, shape (ndata, s*s)
            - x_trunk: spatial grid points, shape (s*s, 2)
        y: pressure solutions, shape (ndata, s*s)
    """
    # Downsampling: 421×421 → 29×29 (r=15 gives s=29)
    r = 15
    s = int(((421 - 1) / r) + 1)  # s = 29

    # Load data: permeability 'coeff' and solution 'sol'
    data = io.loadmat(filename)
    x_branch = data["coeff"][:ndata, ::r, ::r].astype(np.float32) * 0.1 - 0.75
    y = data["sol"][:ndata, ::r, ::r].astype(np.float32) * 100

    # Fix boundary values (dataset has mistake, BC should be 0)
    y[:, 0, :] = 0    # Top boundary
    y[:, -1, :] = 0   # Bottom boundary
    y[:, :, 0] = 0    # Left boundary
    y[:, :, -1] = 0   # Right boundary

    # Create spatial grid [0,1] × [0,1]
    grids = []
    grids.append(np.linspace(0, 1, s, dtype=np.float32))  # x coordinates
    grids.append(np.linspace(0, 1, s, dtype=np.float32))  # y coordinates
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T  # (s*s, 2)

    # Flatten spatial dimensions
    x_branch = x_branch.reshape(ndata, s * s)  # (ndata, 841)
    x = (x_branch, grid)
    y = y.reshape(ndata, s * s)
    return x, y


def dirichlet(inputs, output):
    """
    Output transformation enforcing homogeneous Dirichlet BC.

    Multiplies network output by a function that vanishes on boundaries:
    u(x,y) = 20·x·(1-x)·y·(1-y)·(NN_output + 1)

    This guarantees u(0,y) = u(1,y) = u(x,0) = u(x,1) = 0
    """
    x_trunk = inputs[1]  # Spatial coordinates (x,y)
    x, y = x_trunk[:, 0], x_trunk[:, 1]
    return 20 * x * (1 - x) * y * (1 - y) * (output + 1)


# Load training and test data
x_train, y_train = get_data("piececonst_r421_N1024_smooth1.mat", 1000)
x_test, y_test = get_data("piececonst_r421_N1024_smooth2.mat", 200)
data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

# DeepONet architecture
m = 29 ** 2  # 841 input points for branch (29×29 grid flattened)
net = dde.maps.DeepONetCartesianProd(
    [m, 128, 128, 128],      # Branch: 841 → 128 → 128 → 128
    [2, 128, 128, 128],      # Trunk: 2 (for x,y) → 128 → 128 → 128
    "relu",                   # Activation
    "Glorot normal"           # Initialization
)

# Apply Dirichlet BC enforcement
net.apply_output_transform(dirichlet)

# Compile model
model = dde.Model(data, net)
model.compile(
    "adam",
    lr=0.001,
    metrics=["mean l2 relative error"]
)

# Train
losshistory, train_state = model.train(epochs=100000, batch_size=None)
```

## Critical Parameters

1. **Network Architecture**:
   - Branch: [841, 128, 128, 128]
     - Input: 29×29 = 841 permeability values
     - 3 hidden layers with 128 neurons each
   - Trunk: [2, 128, 128, 128]
     - Input: (x,y) coordinates
     - 3 hidden layers with 128 neurons each
   - Output dimension: 128 (implicit, must match)

2. **Spatial Resolution**:
   - Original grid: 421×421
   - Downsampling rate r: 15
   - Working resolution: 29×29 points
   - Flattened dimension: 841

3. **Data Preprocessing**:
   - Permeability scaling: coeff * 0.1 - 0.75
   - Solution scaling: sol * 100
   - BC correction: manually set boundaries to 0

4. **Output Transform**:
   - Function: 20·x·(1-x)·y·(1-y)·(output + 1)
   - Purpose: enforce homogeneous Dirichlet BC
   - Scaling factor: 20 (improves numerical range)

5. **Training Configuration**:
   - Optimizer: Adam
   - Learning rate: 0.001
   - Epochs: 100,000
   - Batch size: None (full batch)
   - Training samples: 1000
   - Test samples: 200

6. **Activation Function**: ReLU

7. **Weight Initialization**: Glorot normal

8. **Metric**: Mean L2 relative error
