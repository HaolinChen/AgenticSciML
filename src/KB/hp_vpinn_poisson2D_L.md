# hp-VPINN for 2D Poisson Equation on L-Shaped Domain

**Keywords**: [PDE, elliptic, linear, forward-problem, poisson, 2D, irregular, dirichlet, singular, corner-singularity, PINN, weak-form, domain-decomposition, adaptive-refinement, MLP, adam, mse, tensorflow]

**Problem:** Solving the homogeneous Poisson equation (∇²u = -1) on an L-shaped domain with a re-entrant corner at the origin. This problem features a **corner singularity** at (0,0) where the solution exhibits reduced regularity, making it challenging for standard numerical methods. The solution has infinite gradient at the corner, requiring adaptive mesh refinement to capture the singular behavior accurately.

**Issues addressed:**
- Corner singularities in non-convex domains with reduced solution regularity
- Adaptive mesh refinement to concentrate resolution near singular points
- Balancing accuracy between smooth regions and singular regions
- Irregular domain geometry requiring flexible discretization

## Key Method

hp-VPINN addresses corner singularities through **adaptive domain decomposition** with varying element sizes and polynomial orders. The L-shaped domain is decomposed into rectangular elements, with finer elements near the re-entrant corner.

**Domain Geometry**:
```
L-shaped domain: Ω = [-1,1] × [-1,1] \ [0,1] × [0,1]
(Remove upper-right quadrant)

Re-entrant corner at (0,0) with interior angle 3π/2
```

**Mathematical Formulation**:

Homogeneous Poisson: ∇²u = -1 on Ω, u = 0 on ∂Ω

The variational residual in element e is:
```
R^(e)_{k1,k2} = ∫∫_{Ω_e} (∂²u_NN/∂x² + ∂²u_NN/∂y²) v_{k1}(x)v_{k2}(y) dx dy
                + ∫∫_{Ω_e} v_{k1}(x)v_{k2}(y) dx dy
```

**Adaptive Refinement Strategy**:
1. **Coarse mesh (3 elements)**: Divide L-shape into 3 equal rectangular elements
2. **Fine mesh (8-35 elements)**: Concentrate small elements near corner at (0,0)
3. **Local polynomial enrichment**: Use high-order test functions in all elements

The singularity at the re-entrant corner requires local h-refinement (small elements) to capture the rapid variation, while p-refinement (high polynomial order) maintains accuracy in smooth regions.

**Key Innovation**:
- Flexible domain decomposition handles irregular geometries
- Adaptive h-refinement targets singular regions without uniform refinement
- Variational formulation naturally handles weak solutions with reduced regularity

## Implementation

### Core VPINN Class for L-Shaped Domain

```python
class VPINN:
    def __init__(self, X_u_train, u_train, X_f_train, f_train, X_quad, W_quad,
                 F_exact_total, elements, N_testfcn_total, X_test, u_test, layers):

        # Training data
        self.x = X_u_train[:,0:1]
        self.y = X_u_train[:,1:2]
        self.utrain = u_train

        # Quadrature
        self.xquad = X_quad[:,0:1]
        self.yquad = X_quad[:,1:2]
        self.wquad = W_quad

        # Element information for irregular domain
        self.NE_total = np.shape(N_testfcn_total)[0]  # Total number of elements
        self.Ntestx = N_testfcn_total[0][0]  # Test functions in x per element
        self.Ntesty = N_testfcn_total[0][1]  # Test functions in y per element
        self.F_ext_total = F_exact_total  # RHS integrals for each element

        # Initialize neural network
        self.weights, self.biases, self.a = self.initialize_NN(layers)

        # Variational loss over all elements in L-shaped domain
        self.varloss_total = 0
        for e in range(self.NE_total):
            F_ext_element = self.F_ext_total[e]
            Ntest_elementx = N_testfcn_total[e][0]
            Ntest_elementy = N_testfcn_total[e][1]

            # Extract element corners from elements array
            grid_x_element = np.array([elements[e,0,0], elements[e,1,0]])
            grid_y_element = np.array([elements[e,0,1], elements[e,2,1]])

            # Map reference element to physical element
            x_quad_element = tf.constant(
                grid_x_element[0] + (grid_x_element[1]-grid_x_element[0])/2*(self.xquad+1))
            y_quad_element = tf.constant(
                grid_y_element[0] + (grid_y_element[1]-grid_y_element[0])/2*(self.yquad+1))

            # Jacobians
            jacobian_x = ((grid_x_element[1]-grid_x_element[0])/2)
            jacobian_y = ((grid_y_element[1]-grid_y_element[0])/2)
            jacobian = jacobian_x * jacobian_y

            # Neural network derivatives
            d1xu_NN_quad_element, d2xu_NN_quad_element = self.net_dxu(x_quad_element, y_quad_element)
            d1yu_NN_quad_element, d2yu_NN_quad_element = self.net_dyu(x_quad_element, y_quad_element)

            # Test functions
            testx_quad_element = self.Test_fcnx(Ntest_elementx, self.xquad)
            d1testx_quad_element, d2testx_quad_element = self.dTest_fcn(Ntest_elementx, self.xquad)
            testy_quad_element = self.Test_fcny(Ntest_elementy, self.yquad)
            d1testy_quad_element, d2testy_quad_element = self.dTest_fcn(Ntest_elementy, self.yquad)

            # Strong form variational residual
            if var_form == 0:
                integrand_1 = d2xu_NN_quad_element + d2yu_NN_quad_element
                U_NN_element = tf.convert_to_tensor([[
                    jacobian*tf.reduce_sum(
                        self.wquad[:,0:1]*testx_quad_element[r]*
                        self.wquad[:,1:2]*testy_quad_element[k]*integrand_1)
                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)],
                    dtype=tf.float64)

            # Weak form variational residual
            if var_form == 1:
                U_NN_element_1 = tf.convert_to_tensor([[
                    jacobian/jacobian_x*tf.reduce_sum(
                        self.wquad[:,0:1]*d1testx_quad_element[r]*
                        self.wquad[:,1:2]*testy_quad_element[k]*d1xu_NN_quad_element)
                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)],
                    dtype=tf.float64)

                U_NN_element_2 = tf.convert_to_tensor([[
                    jacobian/jacobian_y*tf.reduce_sum(
                        self.wquad[:,0:1]*testx_quad_element[r]*
                        self.wquad[:,1:2]*d1testy_quad_element[k]*d1yu_NN_quad_element)
                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)],
                    dtype=tf.float64)

                U_NN_element = -U_NN_element_1 - U_NN_element_2

            # Compute element residual and loss
            Res_NN_element = tf.reshape(U_NN_element - F_ext_element, [1,-1])
            loss_element = tf.reduce_mean(tf.square(Res_NN_element))
            self.varloss_total = self.varloss_total + loss_element

        # Total loss
        self.lossb = tf.reduce_mean(tf.square(self.u_tf - self.u_pred_boundary))
        self.lossv = self.varloss_total
        self.loss = 10*self.lossb + self.lossv
```

### Domain Decomposition for L-Shaped Domain

```python
# Adaptive domain decomposition strategies

# Option 1: Coarse mesh (3 elements)
if NE_total == 3:
    vertices = [[-1,-1], [0,-1], [1,-1],  # Bottom row
                [-1,0],  [0,0],  [1,0],   # Middle row
                [-1,1],  [0,1]]            # Top row (partial)

    # Three rectangular elements forming L-shape
    elements = np.asarray([
        [vertices[0], vertices[1], vertices[3], vertices[4]],  # Bottom-left
        [vertices[1], vertices[2], vertices[4], vertices[5]],  # Bottom-right
        [vertices[3], vertices[4], vertices[6], vertices[7]]   # Top-left
    ])

    N_testfcn_total = [[N_testfcn_x, N_testfcn_y],
                      [N_testfcn_x, N_testfcn_y],
                      [N_testfcn_x, N_testfcn_y]]

# Option 2: Fine mesh near corner (8 elements)
if NE_total == 8:
    delta = 0.05  # Small element size near corner
    vertices = [[-1,-1], [-delta,-1], [1,-1],
                [-1,-delta], [-delta,-delta], [0,-delta], [delta,-delta], [1,-delta],
                [-delta,0], [0,0], [delta,0], [1,0],
                [-delta,delta], [0,delta],
                [-1,1], [-delta,1], [0,1]]

    # Eight elements with refinement near (0,0)
    elements = np.asarray([
        [vertices[0], vertices[1], vertices[3], vertices[4]],    # Far bottom-left
        [vertices[1], vertices[2], vertices[4], vertices[7]],    # Far bottom-right
        [vertices[3], vertices[4], vertices[14], vertices[15]],  # Far top-left
        [vertices[4], vertices[5], vertices[8], vertices[9]],    # Near corner (bottom-left)
        [vertices[5], vertices[6], vertices[9], vertices[10]],   # Near corner (bottom-mid)
        [vertices[6], vertices[7], vertices[10], vertices[11]],  # Near corner (bottom-right)
        [vertices[8], vertices[9], vertices[12], vertices[13]],  # Near corner (mid-left)
        [vertices[12], vertices[13], vertices[15], vertices[16]] # Near corner (top-left)
    ])

    N_testfcn_total = [[N_testfcn_x, N_testfcn_y] for _ in range(8)]

# Option 3: Very fine mesh (35 elements) - highest resolution near corner
if NE_total == 35:
    # Create graded mesh with smallest elements at corner
    vertices = [
        # Fine grid near corner with spacing 0.125
        # Coarser grid away from corner with spacing 0.25-0.5
        # (Full vertex list omitted for brevity - see code for details)
    ]

    elements = np.asarray([
        # 35 rectangular elements covering L-shaped domain
        # Concentrated refinement in region [-0.25, 0.25] × [-0.25, 0.25]
        # (Full element list omitted for brevity - see code for details)
    ])

    N_testfcn_total = [[N_testfcn_x, N_testfcn_y] for _ in range(35)]
```

### Pre-computing RHS for Homogeneous Poisson

```python
def f_ext(x, y):
    """Force term for homogeneous Poisson equation"""
    return -1  # Constant forcing

# Pre-compute RHS integrals for all elements
F_ext_total = []
for e in range(NE_total):
    Ntest_elementx = N_testfcn_total[e][0]
    Ntest_elementy = N_testfcn_total[e][1]

    # Extract element geometry
    grid_x_element = np.array([elements[e,0,0], elements[e,1,0]])
    grid_y_element = np.array([elements[e,0,1], elements[e,2,1]])

    # Map quadrature points to physical element
    x_quad_element = grid_x_element[0] + (grid_x_element[1]-grid_x_element[0])/2*(x_quad+1)
    y_quad_element = grid_y_element[0] + (grid_y_element[1]-grid_y_element[0])/2*(y_quad+1)
    jacobian = ((grid_x_element[1]-grid_x_element[0])/2) * \
               ((grid_y_element[1]-grid_y_element[0])/2)

    # Test functions
    testx_quad_element = np.asarray([Test_fcn_x(n, x_quad) for n in range(1, Ntest_elementx+1)])
    testy_quad_element = np.asarray([Test_fcn_y(n, y_quad) for n in range(1, Ntest_elementy+1)])

    # Force term (constant)
    f_quad_element = f_ext(x_quad_element, y_quad_element)

    # Compute RHS integral: ∫∫_{Ω_e} (-1)·v_{k1}·v_{k2} dx dy
    F_ext_element = np.asarray([[
        jacobian*np.sum(w_quad[:,0:1]*testx_quad_element[r]*
                       w_quad[:,1:2]*testy_quad_element[k]*f_quad_element)
        for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])

    F_ext_total.append(F_ext_element)
```

### Boundary Conditions for L-Shaped Domain

```python
# Top boundary (partial): y = 1, x ∈ [-1, 0]
x_up = lhs(1, N_bound) - 1  # Sample from [-1, 0]
y_up = np.ones_like(x_up)
u_up_train = np.zeros_like(x_up)  # Homogeneous Dirichlet

# Horizontal interior boundary: y = 0, x ∈ [0, 1]
x_up2 = lhs(1, N_bound)  # Sample from [0, 1]
y_up2 = np.zeros_like(x_up2)
u_up2_train = np.zeros_like(x_up2)

# Bottom boundary: y = -1, x ∈ [-1, 1]
x_lo = 2*lhs(1, 2*N_bound) - 1
y_lo = -np.ones_like(x_lo)
u_lo_train = np.zeros_like(x_lo)

# Right boundary (partial): x = 1, y ∈ [-1, 0]
y_ri = lhs(1, N_bound) - 1
x_ri = np.ones_like(y_ri)
u_ri_train = np.zeros_like(y_ri)

# Vertical interior boundary: x = 0, y ∈ [0, 1]
y_ri2 = lhs(1, N_bound)
x_ri2 = np.zeros_like(y_ri2)
u_ri2_train = np.zeros_like(y_ri2)

# Left boundary: x = -1, y ∈ [-1, 1]
y_le = 2*lhs(1, 2*N_bound) - 1
x_le = -np.ones_like(y_le)
u_le_train = np.zeros_like(y_le)

# Concatenate all boundary segments
X_u_train = np.concatenate((X_up_train, X_up2_train, X_lo_train,
                           X_ri_train, X_ri2_train, X_le_train))
u_train = np.concatenate((u_up_train, u_up2_train, u_lo_train,
                         u_ri_train, u_ri2_train, u_le_train))
```

### Training with Reference Solution

```python
# Load reference solution (from spectral element method)
X_test_data = np.load('Data/X_test.npy')   # Test point coordinates
u_test_data = np.load('Data/y_ref.npy')    # Reference solution values

# Initialize and train model
model = VPINN(X_u_train, u_train, X_f_train, f_train, XY_quad_train, WXY_quad_train,
              F_ext_total, elements, N_testfcn_total, X_test_data, u_test_data, Net_layer)

loss_his = []
model.train(25000 + 1)
u_pred = model.predict()
```

## Critical Parameters

1. **Network architecture**
   - Layers: [2, 5, 5, 5, 1] (2 inputs → 3 hidden layers with 5 neurons → 1 output)
   - Activation: tanh
   - Input: (x, y) coordinates in L-shaped domain
   - Output: u(x, y) solution

2. **Adaptive domain decomposition**
   - NE_total = 3: Coarse mesh (3 equal elements)
   - NE_total = 8: Medium refinement (δ = 0.05 near corner)
   - NE_total = 35: Fine refinement (δ = 0.125 near corner)
   - Refinement concentrates resolution in [-0.25, 0.25] × [-0.25, 0.25]

3. **Test functions**
   - N_testfcn_x: 5 per element in x-direction
   - N_testfcn_y: 5 per element in y-direction
   - Type: Modified Legendre polynomials
   - Total per element: 5 × 5 = 25

4. **Quadrature**
   - N_quad: 10 Gauss-Lobatto points per direction
   - Total per element: 10 × 10 = 100 points
   - Tensor product quadrature

5. **Variational formulation**
   - var_form = 0: Strong form (recommended for this problem)
   - var_form = 1: Weak form
   - Strong form used in paper for L-shaped domain

6. **Training parameters**
   - Optimizer: Adam with learning rate 0.001
   - Iterations: 25000
   - Boundary loss weight: 10
   - Longer training needed for singular solution

7. **Boundary data**
   - N_bound: 20 points per boundary segment
   - All boundaries: homogeneous Dirichlet (u = 0)
   - Interior boundaries at x = 0 (y > 0) and y = 0 (x > 0)

8. **Corner singularity characteristics**
   - Location: Re-entrant corner at (0, 0)
   - Interior angle: 3π/2 (270°)
   - Singularity strength: r^(2/3) behavior near corner
   - Largest errors typically concentrated near (0, 0)
   - Requires h-refinement (small elements) for accuracy

9. **Reference solution**
   - Computed using Spectral Element Method (SEM)
   - 12 elements with polynomial degree 10×10
   - Used as benchmark for error evaluation
