# DeepONet for 2D Darcy Flow on Triangular Domain with Notch

**Keywords**: [PDE, elliptic, linear, forward-problem, darcy, 2D, irregular, DeepONet, MLP, adam, mse, tensorflow]

**Problem:** Learning the solution operator for the 2D Darcy flow equation:
```
-∇·(a∇u) = f on triangular domain with notch
```
Irregular geometry with 2295 mesh points (non-uniform). Operator G maps permeability a (sampled at 101 points) to pressure u (at 2295 points). Custom DeepONet handles irregular meshes via coordinate normalization to [Xmin, Xmax].

**Issues addressed:**
- Operator learning on irregular/non-rectangular domains
- Handling non-uniform mesh distributions
- Branch network processing of lower-resolution inputs (101 points)
- Trunk network querying at irregular spatial locations (2295 points)

## Key Method

Custom DeepONet implementation for irregular domains:

**Architecture**:
- Branch [101, 128, 128, 100]: Processes 101-point permeability samples
- Trunk [2, 128, 128, 128, 100]: Processes irregular (x,y) coordinates with normalization to [Xmin, Xmax]
- Output: u = Σᵢ branch_iᵢ · trunk_iᵢ (element-wise product then sum), dimension p=100

**Key Features**:
- Coordinate normalization handles irregular mesh points
- Element-wise multiplication u_B * u_T followed by reduction sum
- Batch size 100, trained for 20000 epochs with adaptive LR schedule

The network uses element-wise multiplication followed by summation instead of explicit dot product.

## Implementation

```python
import tensorflow as tf
import numpy as np
from fnn import FNN

# Architecture configuration
p = 100  # Output dimension (latent space)
num = 101  # Branch input dimension
layer_B = [num, 128, 128, p]  # Branch network
layer_T = [2, 128, 128, 128, p]  # Trunk network
bs = 100  # Batch size
nx = 2295  # Number of query points (irregular mesh)
epochs = 20000

# Build network
fnn_model = FNN()

# Branch network: permeability field -> latent representation
W_B, b_B = fnn_model.hyper_initial(layer_B)
u_B = fnn_model.fnn_B(W_B, b_B, f_ph)  # f_ph: [bs, 1, 101]
u_B = tf.tile(u_B, [1, nx, 1])  # Replicate for all query points

# Trunk network: spatial coordinates -> latent representation
W_T, b_T = fnn_model.hyper_initial(layer_T)
u_T = fnn_model.fnn_T(W_T, b_T, x, Xmin, Xmax)  # x: [bs, nx, 2]

# Output: element-wise product and sum
u_nn = u_B * u_T  # [bs, nx, p]
u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True)  # [bs, nx, 1]

# Loss and training
loss = tf.reduce_mean(tf.square(u_ph - u_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

# Learning rate schedule
if n < 930:
    lr = 0.001
elif n < 3000:
    lr = 0.0005
else:
    lr = 0.0001
```

## Critical Parameters

1. **Architecture**: p=100, Branch=[101,128,128,100], Trunk=[2,128,128,128,100]
2. **Irregular Mesh**: 2295 query points
3. **Training**: batch_size=100, epochs=20000, adaptive LR
4. **Framework**: TensorFlow 1.x/2.x
