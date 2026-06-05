# hp-VPINN for 1D Poisson Equation

**Keywords**: [PDE, elliptic, linear, forward-problem, poisson, 1D, dirichlet, PINN, weak-form, domain-decomposition, MLP, adam, mse, sine, tensorflow]

**Problem:** Solving one-dimensional Poisson equations with steep gradients and sharp transitions. Standard PINNs struggle with solutions containing high-frequency oscillations or localized steep regions, requiring dense collocation points. The challenge is to achieve high accuracy with efficient training for non-smooth solutions.

**Issues addressed:**
- Steep gradients and sharp transitions in PDE solutions
- Localized features requiring adaptive resolution
- Training efficiency for non-smooth solutions through domain decomposition
- Modal oscillations in approximation errors (Gibbs phenomenon)

## Key Method

hp-VPINN applies **variational formulation with domain decomposition** to 1D elliptic PDEs. The key distinction from standard PINNs:

1. **Variational Residual**: Instead of point-wise residual evaluation, the PDE residual is projected onto polynomial test functions via integration
2. **Domain Decomposition**: The domain is divided into non-overlapping elements where test functions have local support
3. **Multiple Integration-by-Parts Options**: Three variational forms available by performing 0, 1, or 2 integrations by parts

**Mathematical Formulation**:

For the 1D Poisson equation: -d²u/dx² = f(x) with u(-1) = g, u(1) = h

The variational residual in element e is:
```
(1)R^(e)_k = -∫_{Ω_e} (d²u_NN/dx²) v_k(x) dx - ∫_{Ω_e} f(x)v_k(x) dx

(2)R^(e)_k = ∫_{Ω_e} (du_NN/dx)(dv_k/dx) dx - ∫_{Ω_e} f(x)v_k(x) dx

(3)R^(e)_k = -∫_{Ω_e} u_NN(d²v_k/dx²) dx + [boundary terms] - ∫_{Ω_e} f(x)v_k(x) dx
```

The total loss is: L = τ_b·L_boundary + Σ_e (1/K^(e)) Σ_k |R^(e)_k|²

**Advantages**:
- Localized learning focuses optimization on challenging regions
- High-order polynomial test functions capture smooth variations accurately
- Integration by parts reduces regularity requirements on neural network
- Domain decomposition enables h-refinement near steep gradients

## Implementation

### Core VPINN Class for 1D Poisson

```python
class VPINN:
    def __init__(self, X_u_train, u_train, X_quad, W_quad, F_exact_total,
                 grid, X_test, u_test, layers, X_f_train, f_train):

        # Training data
        self.x = X_u_train        # Boundary points
        self.u = u_train          # Boundary values

        # Quadrature for variational integrals
        self.xquad = X_quad       # Quadrature points in reference element [-1,1]
        self.wquad = W_quad       # Quadrature weights

        # Element information
        self.F_ext_total = F_exact_total  # RHS integrals: ∫ f(x)v_k(x) dx for each element
        self.Nelement = np.shape(self.F_ext_total)[0]  # Number of elements
        self.N_test = np.shape(self.F_ext_total[0])[0]  # Number of test functions per element

        # Initialize neural network
        self.weights, self.biases, self.a = self.initialize_NN(layers)

        # Compute variational loss over all elements
        self.varloss_total = 0
        for e in range(self.Nelement):
            F_ext_element = self.F_ext_total[e]  # RHS for this element
            Ntest_element = np.shape(F_ext_element)[0]

            # Map reference element [-1,1] to physical element [grid[e], grid[e+1]]
            x_quad_element = tf.constant(grid[e] + (grid[e+1]-grid[e])/2*(self.xquad+1))
            jacobian = (grid[e+1]-grid[e])/2  # Coordinate transformation Jacobian

            # Neural network and derivatives at quadrature points
            u_NN_quad_element = self.net_u(x_quad_element)
            d1u_NN_quad_element, d2u_NN_quad_element = self.net_du(x_quad_element)

            # Test functions and derivatives at quadrature points
            test_quad_element = self.Test_fcn(Ntest_element, self.xquad)
            d1test_quad_element, d2test_quad_element = self.dTest_fcn(Ntest_element, self.xquad)

            # Three variational formulations via integration by parts
            if var_form == 1:
                # Strong form: -∫ u''·v dx
                U_NN_element = tf.reshape(tf.stack([
                    -jacobian*tf.reduce_sum(self.wquad*d2u_NN_quad_element*test_quad_element[i])
                    for i in range(Ntest_element)
                ]), (-1,1))

            if var_form == 2:
                # Weak form: ∫ u'·v' dx
                U_NN_element = tf.reshape(tf.stack([
                    tf.reduce_sum(self.wquad*d1u_NN_quad_element*d1test_quad_element[i])
                    for i in range(Ntest_element)
                ]), (-1,1))

            if var_form == 3:
                # Ultra-weak form: -∫ u·v'' dx + [boundary terms]
                u_NN_bound_element = self.net_u(x_b_element)
                d1test_bound_element, _ = self.dTest_fcn(Ntest_element, np.array([[-1],[1]]))
                U_NN_element = tf.reshape(tf.stack([
                    -1/jacobian*tf.reduce_sum(self.wquad*u_NN_quad_element*d2test_quad_element[i])
                    + 1/jacobian*tf.reduce_sum(u_NN_bound_element*
                                              np.array([-d1test_bound_element[i][0],
                                                       d1test_bound_element[i][-1]]))
                    for i in range(Ntest_element)
                ]), (-1,1))

            # Compute element residual and loss
            Res_NN_element = U_NN_element - F_ext_element
            loss_element = tf.reduce_mean(tf.square(Res_NN_element))
            self.varloss_total = self.varloss_total + loss_element

        # Total loss: boundary + variational
        self.lossb = tf.reduce_mean(tf.square(self.u_tf - self.u_NN_pred))
        self.lossv = self.varloss_total
        self.loss = lossb_weight*self.lossb + self.lossv
```

### Neural Network with Sine Activation

```python
def neural_net(self, X, weights, biases, a):
    """
    Fully connected network with sine activation functions.
    Sine activations are particularly effective for smooth periodic solutions.
    """
    num_layers = len(weights) + 1
    H = X
    # Hidden layers with sine activation
    for l in range(0, num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.sin(tf.add(tf.matmul(H, W), b))  # sin activation
    # Output layer (linear)
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

def net_u(self, x):
    """Network prediction"""
    u = self.neural_net(tf.concat([x], 1), self.weights, self.biases, self.a)
    return u

def net_du(self, x):
    """Compute first and second derivatives via automatic differentiation"""
    u = self.net_u(x)
    d1u = tf.gradients(u, x)[0]   # First derivative
    d2u = tf.gradients(d1u, x)[0]  # Second derivative
    return d1u, d2u
```

### Test Functions (Modified Legendre Polynomials)

```python
def Test_fcn(self, N_test, x):
    """
    Construct test functions that vanish at element boundaries.
    Uses modified Legendre polynomials: v_k(x) = P_{k+1}(x) - P_{k-1}(x)
    This ensures v_k(-1) = v_k(1) = 0
    """
    test_total = []
    for n in range(1, N_test+1):
        test = Jacobi(n+1, 0, 0, x) - Jacobi(n-1, 0, 0, x)
        test_total.append(test)
    return np.asarray(test_total)

def dTest_fcn(self, N_test, x):
    """
    Compute derivatives of test functions using recurrence relations.
    First and second derivatives needed for different variational forms.
    """
    d1test_total = []
    d2test_total = []
    for n in range(1, N_test+1):
        if n == 1:
            # Special case for n=1
            d1test = ((n+2)/2)*Jacobi(n, 1, 1, x)
            d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1, 2, 2, x)
        elif n == 2:
            # Special case for n=2
            d1test = ((n+2)/2)*Jacobi(n, 1, 1, x) - ((n)/2)*Jacobi(n-2, 1, 1, x)
            d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1, 2, 2, x)
        else:
            # General recurrence relation
            d1test = ((n+2)/2)*Jacobi(n, 1, 1, x) - ((n)/2)*Jacobi(n-2, 1, 1, x)
            d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1, 2, 2, x) - \
                    ((n)*(n+1)/(2*2))*Jacobi(n-3, 2, 2, x)
        d1test_total.append(d1test)
        d2test_total.append(d2test)
    return np.asarray(d1test_total), np.asarray(d2test_total)
```

### Pre-computing RHS Integrals

```python
# Exact solution with steep gradient
def u_ext(x):
    """Target solution: 0.1·sin(8πx) + tanh(80x)"""
    return 0.1*np.sin(8*np.pi*x) + np.tanh(80*x)

def f_ext(x):
    """Force term obtained by substituting exact solution into PDE"""
    return -0.1*(8*np.pi)**2*np.sin(8*np.pi*x) - \
           (2*80**2)*(np.tanh(80*x))/((np.cosh(80*x))**2)

# Domain decomposition
NE = N_Element  # Number of elements
delta_x = 2.0/NE  # Element size (domain is [-1,1])
grid = np.asarray([(-1) + i*delta_x for i in range(NE+1)])

# Pre-compute RHS integrals: F^(e)_k = ∫_{Ω_e} f(x)v_k(x) dx
F_ext_total = []
for e in range(NE):
    # Map reference quadrature points to physical element
    x_quad_element = grid[e] + (grid[e+1]-grid[e])/2*(x_quad+1)
    jacobian = (grid[e+1]-grid[e])/2

    # Evaluate force term at quadrature points
    f_quad_element = f_ext(x_quad_element)

    # Compute integrals for all test functions in this element
    testfcn_element = np.asarray([Test_fcn(n, x_quad) for n in range(1, N_testfcn+1)])
    F_ext_element = jacobian*np.asarray([
        sum(w_quad*f_quad_element*testfcn_element[i])
        for i in range(N_testfcn)
    ])
    F_ext_element = F_ext_element[:,None]
    F_ext_total.append(F_ext_element)

F_ext_total = np.asarray(F_ext_total)
```

### Training and Prediction

```python
# Training loop
def train(self, nIter, tresh):
    tf_dict = {self.x_tf: self.x, self.u_tf: self.u,
               self.x_quad: self.xquad, self.x_test: self.xtest,
               self.xf_tf: self.xf, self.f_tf: self.f}

    for it in range(nIter):
        self.sess.run(self.train_op_Adam, tf_dict)

        if it % 10 == 0:
            loss_value = self.sess.run(self.loss, tf_dict)
            if loss_value < tresh:
                print('It: %d, Loss: %.3e' % (it, loss_value))
                break

        if it % 100 == 0:
            loss_valueb = self.sess.run(self.lossb, tf_dict)
            loss_valuev = self.sess.run(self.lossv, tf_dict)
            print('It: %d, Lossb: %.3e, Lossv: %.3e' % (it, loss_valueb, loss_valuev))

# Main execution
model = VPINN(X_u_train, u_train, X_quad_train, W_quad_train, F_ext_total,
              grid, X_test, u_test, Net_layer, X_f_train, f_train)
model.train(1000 + 1, 2e-32)
u_pred = model.predict(X_test)
```

## Critical Parameters

1. **Network architecture**
   - Layers: [1, 20, 20, 20, 20, 1] (1 input → 4 hidden layers with 20 neurons → 1 output)
   - Activation: sine (sin)
   - Input: x coordinate
   - Output: u(x) solution
   - Sine activation crucial for smooth periodic solutions

2. **Domain decomposition (h-refinement)**
   - N_Element: 1 (single element) or 3 (three elements)
   - For steep solutions, use 3 elements with refined grid near steep region
   - Example: grid = [-1, -0.1, 0.1, 1] concentrates resolution near x=0

3. **Test functions (p-refinement)**
   - N_testfcn: 60 test functions per element
   - Type: Modified Legendre polynomials
   - High polynomial order captures smooth variations accurately
   - Vanishing boundary conditions: v_k(±1) = 0

4. **Quadrature**
   - N_Quad: 80 Gauss-Lobatto quadrature points per element
   - High quadrature order essential for accurate integration
   - Rule: Q ≥ 2N_testfcn to avoid under-integration errors

5. **Variational formulation**
   - var_form = 1: Strong form (-∫ u''·v dx)
   - var_form = 2: Weak form (∫ u'·v' dx) - recommended
   - var_form = 3: Ultra-weak form (-∫ u·v'' dx + boundary terms)
   - Weak form reduces regularity requirements and often converges better

6. **Training parameters**
   - Optimizer: Adam with learning rate 0.001
   - Iterations: 1000
   - Convergence threshold: 2e-32
   - lossb_weight: 1 (boundary loss weight)

7. **Boundary conditions**
   - Type: Dirichlet at x = -1 and x = 1
   - X_u_train: np.array([[-1.0], [1.0]])
   - u_train: u_ext(X_u_train) evaluated from exact solution

8. **Problem-specific parameters**
   - Steep solution: u(x) = 0.1·sin(8πx) + tanh(80x)
   - Steepness parameter: r1 = 80 (controls sharpness of tanh)
   - Oscillation frequency: ω = 8π (high-frequency component)
   - Domain: x ∈ [-1, 1]
