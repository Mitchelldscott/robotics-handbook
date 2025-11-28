# Lie Groups and Lie Algebras

## I. Introduction and Theoretical Foundations

In robotics, state estimation, and control, we frequently manipulate rigid body
transformations. These transformations do not form vector spaces; they form **Lie Groups**.
A Lie Group $G$ is a set that is both a group and a smooth manifold, where the group operations
(multiplication and inversion) are smooth maps.

Because $G$ is a curved manifold, standard calculus (linear interpolation, additive Gaussian
noise, standard differentiation) cannot be applied directly. Instead, we utilize the
**Lie Algebra** $\mathfrak{g}$, which is the tangent space to the group at the identity
element ($T_e G$). The Lie algebra is a vector space, permitting linear operations.

### 1.1 The Manifold and the Tangent Space

* **Group ($G$):** The curved surface of transformations (e.g., Rotation matrices).
* **Algebra ($\mathfrak{g}$):** The flat tangent plane at the identity.
* **Generators:** The basis vectors of $\mathfrak{g}$. Any element in $\mathfrak{g}$ is
  a linear combination of these generators.

### 1.2 Mappings

The relationship between the group and the algebra is governed by the **Exponential Map**
and the **Logarithmic Map**:

$$
\exp: \mathfrak{g} \to G \quad \text{(Surjective onto the connected component of identity)}
$$$$
\log: G \to \mathfrak{g} \quad \text{(Local inverse near identity)}
$$

-----

## II. Common Groups in Robotics

The following table summarizes the dimensions and representations of the groups most
critical to spatial robotics.

| Group | Description | Dim | Matrix Representation | Algebra Generator Structure |
| :--- | :--- | :--- | :--- | :--- |
| $SO(2)$ | 2D Rotations | 1 | $2 \times 2$ Orthogonal, $\det=1$ | Skew-symmetric
$2 \times 2$ |
| $SE(2)$ | 2D Rigid Transform | 3 | $3 \times 3$ Homogeneous |
Top-left skew, top-right vector |
| $SO(3)$ | 3D Rotations | 3 | $3 \times 3$ Orthogonal, $\det=1$ |
Skew-symmetric $3 \times 3$ |
| $SE(3)$ | 3D Rigid Transform | 6 | $4 \times 4$ Homogeneous |
Top-left skew, top-right vector |
| $Sim(3)$| 3D Similarity (Scale) | 7 | $4 \times 4$ Homogeneous |
Coupled Rotation/Scale/Trans |

### 2.1 SO(3): 3D Rotations

The Lie algebra $\mathfrak{so}(3)$ consists of skew-symmetric matrices. We define the "hat"
operator $(\cdot)^\wedge$ (or $[\cdot]_\times$) taking a vector $\omega \in \mathbb{R}^3$
to a matrix:

$$
\omega = \begin{bmatrix} \omega_1 \\ \omega_2 \\ \omega_3 \end{bmatrix}, \quad
\omega^\wedge = [\omega]_\times =
\begin{bmatrix} 0 & -\omega_3 & \omega_2 \\
    \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{bmatrix}
$$

**Exponential Map (Rodrigues' Formula):**
Let $\theta = \|\omega\|$ and $\mathbf{a} = \omega / \theta$.

$$\\exp(\\omega^\\wedge) = I + (\\sin\\theta) \\mathbf{a}^\\wedge + (1 - \\cos\\theta) (\\mathbf{a}^\\wedge)^2
$$\#\#\# 2.2 SE(3): 3D Rigid Transformations

Elements of $SE(3)$ are represented by $4 \times 4$ matrices $T =
\begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$. The Lie algebra $\mathfrak{se}(3)$
consists of "twists". A twist $\xi \in \mathbb{R}^6$ comprises angular velocity $\omega$
and linear velocity $u$ (note: ordering conventions vary, often $\xi = [u, \omega]^T$ or
$[\omega, u]^T$).

Using $\xi = [u^T, \omega^T]^T$:

$$
\xi^\wedge = \begin{bmatrix} [\omega]_\times & u \\ 0 & 0 \end{bmatrix}
    \in \mathbb{R}^{4 \times 4}
$$**Exponential Map for SE(3):**

$$\\exp(\\xi^\\wedge) = \\begin{bmatrix} \\exp(\\omega^\\wedge) & V u \\ 0 & 1 \\end{bmatrix}
$$Where $V$ is the left Jacobian of $SO(3)$:

$$
V = I + \frac{1-\cos\theta}{\theta^2} [\omega]_\times + \frac{\theta - \sin\theta}{\theta^3}
([\omega]_\times)^2
$$

-----

## 3\. Calculus on Lie Groups

### 3.1 The Adjoint Representation

The Adjoint operator, $Adj_X$, maps tangent vectors from the tangent space at the right of
$X$ to the tangent space at the left of $X$. It is a linear mapping represented by a matrix.

**Definition:**

$$
\exp(Adj_X \cdot \eta) = X \cdot \exp(\eta) \cdot X^{-1}
$$

**Closed Forms:**

* **For $SO(3)$:** $Adj_R = R$.
* **For $SE(3)$:** Let $T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$.

$$
Adj\_T = \\begin{bmatrix} R & [t]\_\\times R \\ 0 & R \\end{bmatrix}
    \\in \\mathbb{R}^{6 \\times 6}
$$

*(Note: The structure of $Adj_T$ depends on the ordering of $\xi$. The above assumes
$\xi = [\omega, u]^T$. If $\xi = [u, \omega]^T$, swap the columns.)*

### 3.2 Perturbations and Derivatives

Since we cannot simply say $X + \Delta$, we use multiplicative perturbations.

  * **Right Perturbation (Local):** $\tilde{X} = X \cdot \exp(\xi^\wedge)$
  * **Left Perturbation (Global):** $\tilde{X} = \exp(\xi^\wedge) \cdot X$

To differentiate a function $f: G \to G$, we map input perturbations to output perturbations
via the Jacobian.

$$
\frac{\partial f(X)}{\partial X} \triangleq \lim_{\epsilon \to 0}
\frac{\log(f(X \cdot \exp(\epsilon)) \cdot f(X)^{-1})}{\epsilon}
$$

### 3.3 Jacobians of the Exponential Map

The relationship between a rate of change in the tangent vector $\dot{\xi}$ and the body
velocity is non-trivial.

$$( \\exp(\\xi(t)^\\wedge) )^{-1} \\frac{d}{dt} \\exp(\\xi(t)^\\wedge) = (J\_r(-\\xi) \\dot{\\xi})^\\wedge
$$Where $J_r$ is the **Right Jacobian** of the group. For $SO(3)$:

$$
J_r(\omega) = I - \frac{1-\cos\theta}{\theta^2}[\omega]_\times + \frac{\theta-\sin\theta}{\theta^3}([\omega]_\times)^2
$$*(Inverse Jacobians are required for iterative estimation, e.g., Gauss-Newton on manifolds).*

-----

## 4\. Uncertainty and Statistics

We define a Gaussian distribution on a manifold $\mathcal{N}(\mu, \Sigma)$ using tangent
space perturbations.
Let the mean be $\bar{X} \in G$ and covariance $\Sigma \in
\mathbb{R}^{dim(\mathfrak{g}) \times dim(\mathfrak{g})}$.
A random variable $X$ is distributed as:

$$X = \\bar{X} \\cdot \\exp(\\xi^\\wedge), \\quad \\xi \\sim \\mathcal{N}(0, \\Sigma)
$$\#\#\# 4.1 Covariance Propagation

To propagate covariance through a transformation $Y = T \cdot X$:

$$
\Sigma_Y \approx Adj_T \cdot \Sigma_X \cdot Adj_T^T
$$To propagate covariance through an inverse $Y = X^{-1}$:

$$\\Sigma\_Y \\approx Adj\_{X^{-1}} \\cdot \\Sigma\_X \\cdot Adj\_{X^{-1}}^T

$$

-----

## 5\. Implementation Reference

### 5.1 Numerical Stability (Small Angle Approx)

Direct implementation of terms like $\frac{\sin \theta}{\theta}$ causes division by zero as
$\theta \to 0$. Implementations must use Taylor Series expansions for small $\theta$ (typically
$\theta^2 < 10^{-4}$).

**For $SO(3)$ exponential coefficients:**
Let $A = \frac{\sin\theta}{\theta}$ and $B = \frac{1-\cos\theta}{\theta^2}$.

* If $\theta^2$ is small:
* $A \approx 1 - \theta^2/6 + \theta^4/120$
* $B \approx 1/2 - \theta^2/24 + \theta^4/720$

### 5.2 Algorithms

#### Algorithm 1: SE(3) Exponential Map

```python
def exp_se3(xi):
"""
Input: xi (twist vector [u, omega] of size 6)
Output: T (4x4 homogeneous matrix)
"""
u = xi[0:3]      # Linear component
omega = xi[3:6]  # Angular component
theta_sq = dot(omega, omega)
theta = sqrt(theta_sq)

Omega = skew(omega)  # 3x3 skew symmetric
Omega_sq = Omega @ Omega

if theta_sq < EPSILON:
# Taylor expansion for coefficients
A = 1.0 - theta_sq/6.0
B = 0.5 - theta_sq/24.0
C = 1.0/6.0 - theta_sq/120.0
else:
# Closed form
inv_theta = 1.0 / theta
A = sin(theta) * inv_theta
B = (1.0 - cos(theta)) * (inv_theta**2)
C = (1.0 - A) * (inv_theta**2)

# Rotation SO(3)
R = I + A * Omega + B * Omega_sq

# Left Jacobian of SO(3) * u
V = I + B * Omega + C * Omega_sq
t = V @ u

return [[R, t], [0, 1]]
```

#### Algorithm 2: Uncertainty Propagation (Composition)

```python
def compose_uncertainty(T1, Cov1, T2, Cov2):
"""
Computes the covariance of T_result = T1 * T2
Input: T1, T2 (SE3 matrices)
Cov1, Cov2 (6x6 covariance matrices in tangent space)
Output: Cov_result
"""
# The perturbation model is X_true = X_nom * exp(xi)
# result = T1 * exp(xi1) * T2 * exp(xi2)
#        = T1 * T2 * (T2_inv * exp(xi1) * T2) * exp(xi2)
#        = (T1 * T2) * exp(Adj_T2_inv * xi1) * exp(xi2)
# Assuming small noise, exp(a)*exp(b) ~ exp(a+b)
# xi_result = Adj_T2_inv * xi1 + xi2

Adj_inv = adjoint_se3(inverse(T2))

# Law of linear propagation
Cov_result = Adj_inv @ Cov1 @ transpose(Adj_inv) + Cov2

return Cov_result
```

#### Algorithm 3: SE(3) Logarithm

```python
def log_se3(T):
"""
Input: T (4x4 SE3 matrix)
Output: xi (twist vector [u, omega])
"""
R = T[0:3, 0:3]
t = T[0:3, 3]

cos_theta = 0.5 * (trace(R) - 1)
# Clamp to handle numerical noise
cos_theta = clamp(cos_theta, -1.0, 1.0)
theta = acos(cos_theta)
theta_sq = theta**2

if theta_sq < EPSILON:
# Small angle approximation for ln(R)
# R ~ I + Omega
omega = vee(R - R.T) / 2.0
# V_inv ~ I - 0.5*Omega
V_inv = I - 0.5 * skew(omega)
else:
scale = theta / (2 * sin(theta))
omega = scale * vee(R - R.T)

Omega = skew(omega)
A = sin(theta)/theta
B = (1 - cos(theta))/theta_sq

# V inverse formula
V_inv = I - 0.5*Omega + (1/(theta_sq) * (1 - A/(2*B))) * (Omega @ Omega)

u = V_inv @ t
return [u, omega]
```

### 5.3 Practical Pitfalls

1. **Coordinate Ordering:** Always verify if a library expects twists as
    `[linear, angular]` or `[angular, linear]`.
2. **Perturbation Side:** Mixing Global (Left) and Local (Right) perturbations is the most
    common source of derivation errors. Stick to one convention (usually Local/Right for estimation).
3. **Manifold Optimization:** When optimizing a pose $T$, update steps must be
    $T \leftarrow T \cdot \exp(\Delta \xi)$, not $T \leftarrow T + \Delta$.
4. **Normalization:** After repeated matrix multiplications, rotation matrices $R$ may
    drift from orthogonality. Periodically re-normalize using SVD or QR decomposition.
