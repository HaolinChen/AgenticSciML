# hp-VPINN for 2D Poisson Equation

**Keywords**: [PDE, elliptic, linear, forward-problem, poisson, 2D, regular, dirichlet, PINN, weak-form, domain-decomposition, MLP, adam, mse, tensorflow]

**Problem:** Solving two-dimensional Poisson equations with steep gradients and directional features. The challenge includes capturing solutions with steep changes in one direction while maintaining smooth variations in another, requiring adaptive spatial resolution and efficient training over 2D domains.

**Issues addressed:**
- Directional steep gradients (e.g., sharp transition along x-axis with smooth variation along y-axis)
- Training efficiency for 2D PDEs through structured domain decomposition
- Balancing accuracy across regions with varying solution complexity
- Modal oscillations characteristic of spectral methods

## Key Method

hp-VPINN extends variational formulation with domain decomposition to **two-dimensional elliptic PDEs** using tensor product construction of test functions.

**Mathematical Formulation**:

For the 2D Poisson equation: ∇²u = ∂²u/∂x² + ∂²u/∂y² = f(x,y) on Ω = [-1,1] × [-1,1]

The domain is decomposed into N_x × N_y rectangular elements. In each element (e_x, e_y), the variational residual is:
```
R^(ex,ey)_{k1,k2} = (∇²u_NN - f, v_{k1}(x)·v_{k2}(y))_{Ω_{ex,ey}}
```

where v_{k1}(x) and v_{k2}(y) are 1D test functions, forming 2D test functions via tensor product.

**Three Variational Forms**:

**(1) Strong Form** (no integration by parts):
```
R^(ex,ey)_{k1,k2} = ∫∫_{Ω_{ex,ey}} (∂²u_NN/∂x² + ∂²u_NN/∂y²)·v_{k1}(x)·v_{k2}(y) dx dy
                   - ∫∫_{Ω_{ex,ey}} f(x,y)·v_{k1}(x)·v_{k2}(y) dx dy
```

**(2) Weak Form** (once integration by parts in each direction):
```
R^(ex,ey)_{k1,k2} = -∫∫_{Ω_{ex,ey}} (∂u_NN/∂x)·(∂v_{k1}/∂x)·v_{k2} dx dy
                    -∫∫_{Ω_{ex,ey}} (∂u_NN/∂y)·v_{k1}·(∂v_{k2}/∂y) dx dy
                    - [RHS integral]
```

Total loss: L = τ_b·L_boundary + Σ_{ex,ey} (1/(K_x·K_y)) Σ_{k1,k2} |R^(ex,ey)_{k1,k2}|²

**Advantages**:
- Tensor product structure enables efficient 2D integration via 1D quadrature rules
- Structured mesh decomposition balances computational cost across elements
- Independent control of resolution in x and y directions
- Weak formulation reduces smoothness requirements on neural network

## Implementation

### Core VPINN Class for 2D Poisson

```python
class VPINN:
    def __init__(self, X_u_train, u_train, X_f_train, f_train, X_quad, W_quad,
                 U_exact_total, F_exact_total, gridx, gridy, N_testfcn, X_test, u_test, layers):

        # Training data
        self.x = X_u_train[:,0:1]  # Boundary x coordinates
        self.y = X_u_train[:,1:2]  # Boundary y coordinates
        self.utrain = u_train      # Boundary values

        # Quadrature for tensor product integration
        self.xquad = X_quad[:,0:1]  # x quadrature points
        self.yquad = X_quad[:,1:2]  # y quadrature points
        self.wquad = W_quad         # Tensor product weights

        # Element information
        self.Nelementx = np.size(N_testfcn[0])  # Number of elements in x
        self.Nelementy = np.size(N_testfcn[1])  # Number of elements in y
        self.Ntestx = N_testfcn[0][0]  # Test functions per element in x
        self.Ntesty = N_testfcn[1][0]  # Test functions per element in y
        self.F_ext_total = F_exact_total  # Pre-computed RHS integrals

        # Initialize neural network
        self.weights, self.biases, self.a = self.initialize_NN(layers)

        # Variational loss over all elements
        self.varloss_total = 0
        for ex in range(self.Nelementx):
            for ey in range(self.Nelementy):
                F_ext_element = self.F_ext_total[ex, ey]  # RHS for this element
                Ntest_elementx = N_testfcn[0][ex]
                Ntest_elementy = N_testfcn[1][ey]

                # Map reference element to physical element
                x_quad_element = tf.constant(gridx[ex] + (gridx[ex+1]-gridx[ex])/2*(self.xquad+1))
                y_quad_element = tf.constant(gridy[ey] + (gridy[ey+1]-gridy[ey])/2*(self.yquad+1))

                # Jacobians for coordinate transformation
                jacobian_x = ((gridx[ex+1]-gridx[ex])/2)
                jacobian_y = ((gridy[ey+1]-gridy[ey])/2)
                jacobian = jacobian_x * jacobian_y

                # Neural network derivatives at quadrature points
                d1xu_NN_quad_element, d2xu_NN_quad_element = self.net_dxu(x_quad_element, y_quad_element)
                d1yu_NN_quad_element, d2yu_NN_quad_element = self.net_dyu(x_quad_element, y_quad_element)

                # Test functions at quadrature points
                testx_quad_element = self.Test_fcnx(Ntest_elementx, self.xquad)
                d1testx_quad_element, d2testx_quad_element = self.dTest_fcn(Ntest_elementx, self.xquad)
                testy_quad_element = self.Test_fcny(Ntest_elementy, self.yquad)
                d1testy_quad_element, d2testy_quad_element = self.dTest_fcn(Ntest_elementy, self.yquad)

                # Strong form: ∫∫ (∂²u/∂x² + ∂²u/∂y²)·v_{k1}·v_{k2} dx dy
                if var_form == 0:
                    integrand_1 = d2xu_NN_quad_element + d2yu_NN_quad_element
                    U_NN_element = tf.convert_to_tensor([[
                        jacobian*tf.reduce_sum(
                            self.wquad[:,0:1]*testx_quad_element[r]*
                            self.wquad[:,1:2]*testy_quad_element[k]*integrand_1)
                        for r in range(Ntest_elementx)] for k in range(Ntest_elementy)],
                        dtype=tf.float64)

                # Weak form: -∫∫ (∂u/∂x·∂v_{k1}/∂x·v_{k2} + u/∂y·v_{k1}·∂v_{k2}/∂y) dx dy
                if var_form == 1:
                    # x-direction contribution
                    U_NN_element_1 = tf.convert_to_tensor([[
                        jacobian/jacobian_x*tf.reduce_sum(
                            self.wquad[:,0:1]*d1testx_quad_element[r]*
                            self.wquad[:,1:2]*testy_quad_element[k]*d1xu_NN_quad_element)
                        for r in range(Ntest_elementx)] for k in range(Ntest_elementy)],
                        dtype=tf.float64)

                    # y-direction contribution
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

        # Total loss: boundary + variational
        self.lossb = tf.reduce_mean(tf.square(self.u_tf - self.u_pred_boundary))
        self.lossv = self.varloss_total
        self.loss = 10*self.lossb + self.lossv
```

### Neural Network with 2D Input

```python
def neural_net(self, X, weights, biases, a):
    """
    Fully connected network for 2D problems with tanh activation.
    X: concatenated input [x, y]
    """
    num_layers = len(weights) + 1
    H = X
    # Hidden layers with tanh activation
    for l in range(0, num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    # Output layer (linear)
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

def net_u(self, x, y):
    """Network prediction for 2D input"""
    u = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases, self.a)
    return u

def net_dxu(self, x, y):
    """Compute x-derivatives via automatic differentiation"""
    u = self.net_u(x, y)
    d1xu = tf.gradients(u, x)[0]   # ∂u/∂x
    d2xu = tf.gradients(d1xu, x)[0] # ∂²u/∂x²
    return d1xu, d2xu

def net_dyu(self, x, y):
    """Compute y-derivatives via automatic differentiation"""
    u = self.net_u(x, y)
    d1yu = tf.gradients(u, y)[0]   # ∂u/∂y
    d2yu = tf.gradients(d1yu, y)[0] # ∂²u/∂y²
    return d1yu, d2yu
```

### Test Functions for 2D (Tensor Product)

```python
def Test_fcnx(self, N_test, x):
    """
    1D test functions in x-direction.
    Modified Legendre polynomials: v_k(x) = P_{k+1}(x) - P_{k-1}(x)
    """
    test_total = []
    for n in range(1, N_test+1):
        test = Jacobi(n+1, 0, 0, x) - Jacobi(n-1, 0, 0, x)
        test_total.append(test)
    return np.asarray(test_total)

def Test_fcny(self, N_test, y):
    """
    1D test functions in y-direction.
    Same construction as x-direction.
    """
    test_total = []
    for n in range(1, N_test+1):
        test = Jacobi(n+1, 0, 0, y) - Jacobi(n-1, 0, 0, y)
        test_total.append(test)
    return np.asarray(test_total)

def dTest_fcn(self, N_test, x):
    """
    Derivatives of 1D test functions.
    Used in both x and y directions.
    """
    d1test_total = []
    d2test_total = []
    for n in range(1, N_test+1):
        if n == 1:
            d1test = ((n+2)/2)*Jacobi(n, 1, 1, x)
            d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1, 2, 2, x)
        elif n == 2:
            d1test = ((n+2)/2)*Jacobi(n, 1, 1, x) - ((n)/2)*Jacobi(n-2, 1, 1, x)
            d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1, 2, 2, x)
        else:
            d1test = ((n+2)/2)*Jacobi(n, 1, 1, x) - ((n)/2)*Jacobi(n-2, 1, 1, x)
            d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1, 2, 2, x) - \
                    ((n)*(n+1)/(2*2))*Jacobi(n-3, 2, 2, x)
        d1test_total.append(d1test)
        d2test_total.append(d2test)
    return np.asarray(d1test_total), np.asarray(d2test_total)
```

### Pre-computing RHS Integrals for 2D

```python
# Exact solution with directional steep gradient
def u_ext(x, y):
    """Steep gradient in x, sinusoidal in y"""
    omegax = 2*np.pi
    omegay = 2*np.pi
    r1 = 10
    return (0.1*np.sin(omegax*x) + np.tanh(r1*x)) * np.sin(omegay*y)

def f_ext(x, y):
    """Force term from Laplacian of exact solution"""
    omegax = 2*np.pi
    omegay = 2*np.pi
    r1 = 10
    # ∇²u = ∂²u/∂x² + ∂²u/∂y²
    term1 = (-0.1*(omegax**2)*np.sin(omegax*x) -
            (2*r1**2)*(np.tanh(r1*x))/((np.cosh(r1*x))**2)) * np.sin(omegay*y)
    term2 = (0.1*np.sin(omegax*x) + np.tanh(r1*x)) * (-omegay**2 * np.sin(omegay*y))
    return term1 + term2

# Domain decomposition: structured rectangular mesh
NE_x, NE_y = N_el_x, N_el_y
delta_x = 2.0/NE_x  # Element size in x
delta_y = 2.0/NE_y  # Element size in y
grid_x = np.asarray([(-1) + i*delta_x for i in range(NE_x+1)])
grid_y = np.asarray([(-1) + i*delta_y for i in range(NE_y+1)])

# Tensor product quadrature
[X_quad, WX_quad] = GaussLobattoJacobiWeights(N_quad, 0, 0)
Y_quad, WY_quad = (X_quad, WX_quad)
xx, yy = np.meshgrid(X_quad, Y_quad)
wxx, wyy = np.meshgrid(WX_quad, WY_quad)
XY_quad_train = np.hstack((xx.flatten()[:,None], yy.flatten()[:,None]))
WXY_quad_train = np.hstack((wxx.flatten()[:,None], wyy.flatten()[:,None]))

# Pre-compute RHS integrals for all elements
F_ext_total = []
for ex in range(NE_x):
    for ey in range(NE_y):
        # Map to physical element
        x_quad_element = grid_x[ex] + (grid_x[ex+1]-grid_x[ex])/2*(x_quad+1)
        y_quad_element = grid_y[ey] + (grid_y[ey+1]-gridy[ey])/2*(y_quad+1)
        jacobian = ((grid_x[ex+1]-grid_x[ex])/2) * ((grid_y[ey+1]-grid_y[ey])/2)

        # Test functions
        testx_quad_element = np.asarray([Test_fcn_x(n, x_quad) for n in range(1, N_testfcn_x+1)])
        testy_quad_element = np.asarray([Test_fcn_y(n, y_quad) for n in range(1, N_testfcn_y+1)])

        # Force term at quadrature points
        f_quad_element = f_ext(x_quad_element, y_quad_element)

        # Compute integrals: ∫∫ f·v_{k1}·v_{k2} dx dy
        F_ext_element = np.asarray([[
            jacobian*np.sum(w_quad[:,0:1]*testx_quad_element[r]*
                          w_quad[:,1:2]*testy_quad_element[k]*f_quad_element)
            for r in range(N_testfcn_x)] for k in range(N_testfcn_y)])

        F_ext_total.append(F_ext_element)

F_ext_total = np.reshape(F_ext_total, [NE_x, NE_y, N_testfcn_y, N_testfcn_x])
```

### Boundary Conditions Setup

```python
# Top boundary: y = 1
x_up = 2*lhs(1, N_bound) - 1
y_up = np.ones_like(x_up)
u_up_train = u_ext(x_up, y_up)
X_up_train = np.hstack((x_up, y_up))

# Bottom boundary: y = -1
x_lo = 2*lhs(1, N_bound) - 1
y_lo = -np.ones_like(x_lo)
u_lo_train = u_ext(x_lo, y_lo)
X_lo_train = np.hstack((x_lo, y_lo))

# Right boundary: x = 1
y_ri = 2*lhs(1, N_bound) - 1
x_ri = np.ones_like(y_ri)
u_ri_train = u_ext(x_ri, y_ri)
X_ri_train = np.hstack((x_ri, y_ri))

# Left boundary: x = -1
y_le = 2*lhs(1, N_bound) - 1
x_le = -np.ones_like(y_le)
u_le_train = u_ext(x_le, y_le)
X_le_train = np.hstack((x_le, y_le))

# Concatenate all boundaries
X_u_train = np.concatenate((X_up_train, X_lo_train, X_ri_train, X_le_train))
u_train = np.concatenate((u_up_train, u_lo_train, u_ri_train, u_le_train))
```

## Critical Parameters

1. **Network architecture**
   - Layers: [2, 5, 5, 5, 1] (2 inputs → 3 hidden layers with 5 neurons → 1 output)
   - Activation: tanh
   - Input: (x, y) coordinates
   - Output: u(x, y) solution

2. **Domain decomposition (h-refinement)**
   - N_el_x: 4 elements in x-direction
   - N_el_y: 4 elements in y-direction
   - Total elements: 4 × 4 = 16
   - Can use non-uniform grids to concentrate resolution near steep gradients

3. **Test functions (p-refinement)**
   - N_test_x: 5 test functions per element in x
   - N_test_y: 5 test functions per element in y
   - Total test functions per element: 5 × 5 = 25
   - Type: Tensor product of 1D modified Legendre polynomials

4. **Quadrature**
   - N_quad: 10 points per direction per element
   - Total integration points per element: 10 × 10 = 100
   - Tensor product Gauss-Lobatto quadrature
   - Rule: Use Q ≥ 2·max(N_test_x, N_test_y)

5. **Variational formulation**
   - var_form = 0: Strong form (∫∫ ∇²u·v dx dy)
   - var_form = 1: Weak form (-∫∫ ∇u·∇v dx dy) - recommended
   - Weak form typically provides better conditioning

6. **Training parameters**
   - Optimizer: Adam with learning rate 0.001
   - Iterations: 10000
   - Boundary loss weight: 10
   - Ensures accurate satisfaction of Dirichlet boundary conditions

7. **Boundary data**
   - N_bound: 80 points per boundary edge
   - Total boundary points: 4 × 80 = 320
   - Sampled using Latin Hypercube Sampling (LHS)

8. **Problem-specific parameters**
   - Steep solution: u(x,y) = (0.1·sin(2πx) + tanh(10x)) · sin(2πy)
   - Steepness in x: controlled by r1 = 10
   - Oscillation frequencies: ω_x = 2π, ω_y = 2π
   - Domain: (x, y) ∈ [-1, 1] × [-1, 1]
