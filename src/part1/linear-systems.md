# Introduction to Applied Linear Algebra and Differential Equations

@@boyd2018introduction

## I. Data Representations

### 1. Vectors

A vector is an ordered set of values, represented as a bold lowercase $\mathbf{v}$ or
lowercase with an overhead arrow $\overrightarrow{v}$

#### Vector space

* A vector space is a structure consisting of a set of vectors $V$ that
provides the following operators
  * **Addition** $V + V \to V$
    1. $a + b = b + a$ (commutative)
    2. $(a + b) + c = a + (b + c)$ (associative)
    3. $a + 0 = 0 + a = a$ (zero, identity)
  * **Scalar Multiplication** $F \times V \to V$
    1. $(\beta \gamma) a = \beta (\gamma a)$ (associative)
    2. $(\beta + \gamma) a = \beta a + \gamma a$ (left distributive)
    3. $\beta (a + b) = \beta a + \beta b$ (right distributive)
  * **Inner product** $V \cdot V \to F$
    1. $a^{T} b = b^{T} a$
    2. $(\gamma a)^{T} b = \gamma (a^{T} b)$
    3. $(a + b)^T c = a^T c + b^T c$

#### Linear Functions

* **Inner Products**
  * Suppose $f : \mathbb{R}^n \to \mathbb{R}$ is **linear**
    Then it can be expressed as
    $$f(x) = a^{\mathsf{T}} x$$
    for some $a \in \mathbb{R}^n$.

  * Specifically, the coefficients of $a$ are given by:
    $$a_i = f(e_i)$$
    where $e_i$ is the $i$-th standard basis vector

  * This follows from:
    $$\begin{aligned}
        f(x)
        &= f(x_1 e_1 + x_2 e_2 + \cdots + x_n e_n) \\
        &= x_1 f(e_1) + x_2 f(e_2) + \cdots + x_n f(e_n)
    \end{aligned}$$

* **Superposition**
  * $f(\alpha x + \beta y) = \alpha f(x) + \beta f(y)$

* **Affine Function**
A function that is **linear plus a constant** is called **affine**
  * General form:
  $$f(x) = a^{\mathsf{T}} x + b$$
  where $a$ is an $n$-vector and $b$ is a scalar
  * $f : \mathbb{R}^n \to \mathbb{R}$ is **affine** if and only if
  $$f(\alpha x + \beta y) = \alpha f(x) + \beta f(y)$$
  holds for all $\alpha, \beta$ with $\alpha + \beta = 1$, and all $x, y \in \mathbb{R}^n$

#### Norms

A **norm** is a function that measures the “size” or “length” of a vector
$x \in \mathbb{R}^n$.  
It assigns a nonnegative scalar to each vector, written $\|x\|$, and must satisfy the
following properties:

1. **Positive definiteness:** $\|x\| \ge 0$, and $\|x\| = 0 \iff x = 0$
2. **Homogeneity:** $\|\alpha x\| = |\alpha|\,\|x\|$ for any scalar $\alpha$
3. **Triangle inequality:** $\|x + y\| \le \|x\| + \|y\|$

##### General $L_p$-norm

The family of **$L_p$** norms (for $p \ge 1$) is defined as

$$\|x\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}$$

This expression measures vector “length” according to the exponent $p$:

* $p = 1$: Manhattan or taxicab norm  
* $p = 2$: Euclidean norm  
* $p \to \infty$: Maximum norm, $\|x\|_\infty = \max_i |x_i|$

##### Two-norm (Euclidean Distance)

The most common case is $p = 2$:

$$
\|x\|_2 = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}
        = \sqrt{x^{\mathsf{T}} x}
$$

This norm corresponds to the **Euclidean distance** from the origin to the point
represented by $x$ in $n$-dimensional space.

### 2. Matrices

* A **matrix** is a rectangular array of numbers, e.g.:
  $$\begin{bmatrix}
    a_{11} & a_{12} & a_{13} & a_{14} \\
    a_{21} & a_{22} & a_{23} & a_{24} \\
    a_{31} & a_{32} & a_{33} & a_{34} \\
  \end{bmatrix}$$

* The **size** of a matrix is *(rows, columns)*.  
  *Example: The matrix above is $3 \times 4$*

* Two matrices are **equal** ($A = B$) if:
  * They are the same size, and  
  * All corresponding entries are equal.

#### Matrix Types

* An $m \times n$ matrix $A$ is:
  * **Tall** if $m > n$
  * **Wide** if $m < n$
  * **Square** if $m = n$

#### Special Cases

* An $n \times 1$ matrix is an **$n$-vector** (column vector).  
* A $1 \times 1$ matrix is a **scalar** (number).  
* A $1 \times n$ matrix is a **row vector**, e.g.:

#### Span, Basis, and Dimension

* **Linear Combination**: A vector $\mathbf{v}$ is a linear combination of vectors
$\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$ if it can be written as:

$$\mathbf{v} = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_k \mathbf{v}_k$$

  for some scalars $c_1, \dots, c_k$.

* **Span**: The span of a set of vectors $S = \{\mathbf{v}_1, \dots, \mathbf{v}_k\}$ is
the set
of all possible linear combinations of those vectors. It forms a subspace.

$$\text{span}(S) = \left\{ \sum_{i=1}^k c_i \mathbf{v}_i \mid c_i \in \mathbb{R} \right\}$$

* **Linear Independence**: A set of vectors $S$ is **linearly independent** if the only
solution to
the equation

$$c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_k \mathbf{v}_k = \mathbf{0}$$

is the trivial solution $c_1 = c_2 = \cdots = c_k = 0$.

> If any non-zero solution exists, the set is linearly dependent.

* **Basis**: A basis for a vector space $V$ is a set of vectors $B$ that satisfies two conditions:

  * $B$ is **linearly independent**.
  * The vectors in $B$ span the space $V$ (i.e., $\text{span}(B) = V$).

* **Dimension**: The dimension of a vector space $V$, denoted $\dim(V)$, is the number of
vectors in any basis for $V$.

* The dimension of the Column Space, $\dim(C(A))$, is the rank ($r$) of the matrix $A$.
* The dimension of the Nullspace, $\dim(N(A))$, is the nullity ($n-r$) of the matrix $A$.

#### Matrix Operations

* **Addition**: If $A$ and $B$ are both $m \times n$, their sum is element-wise:

$$(A+B)_{ij} = A_{ij} + B_{ij}$$

* **Scalar Multiplication**: The product of a scalar $c$ and a matrix $A$ is element-wise:

$$(cA)_{ij} = c \cdot A_{ij}$$

* **Matrix Multiplication**: If $A$ is $m \times n$ and $B$ is $n \times p$, their product
$C = AB$ is an $m \times p$ matrix.

* The entry $(C)_{ij}$ is the inner product of the $i$-th row of $A$ and the $j$-th column
of $B$:

$$(AB)_{ij} = \sum_{k=1}^n A_{ik} B_{kj}$$

> **Note**: Matrix multiplication is not commutative ($AB \neq BA$ in general).

* **Matrix Inverse**: An $n \times n$ square matrix $A$ is invertible (or non-singular) if
there exists a matrix $A^{-1}$ such that:

  $$A A^{-1} = A^{-1} A = I_n$$

where $I_n$ is the $n \times n$ identity matrix.

* $A$ is invertible if and only if its rank is $n$ (full rank).
* **Properties**: $(AB)^{-1} = B^{-1} A^{-1}$ and $(A^T)^{-1} = (A^{-1})^T$.

#### Linear Transformations

A **transformation** (or map) $T: V \to W$ from a vector space $V$ to a vector
space $W$ is linear if it preserves vector addition and scalar multiplication:

1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$ *(Additivity)*
2. $T(c\mathbf{v}) = cT(\mathbf{v})$ *(Homogeneity)*

* These two rules combine to the superposition principle:
$T(c\mathbf{u} + d\mathbf{v}) = cT(\mathbf{u}) + dT(\mathbf{v})$.

* **Matrix of a Transformation**: Every linear transformation $T: \mathbb{R}^n \to \mathbb{R}^m$
can be represented by a unique $m \times n$ matrix $A$ such that $T(\mathbf{x}) = A\mathbf{x}$.

  * The columns of $A$ are the images of the standard basis vectors $\mathbf{e}_j$:

  $$A = \begin{bmatrix} | & & | \\ T(\mathbf{e}_1) & \cdots & T(\mathbf{e}_n) \\ | & & | \end{bmatrix}$$

* **Kernel and Image**:

  * **Kernel (Nullspace)**: The set of all vectors in $V$ that map to the zero vector in $W$.

  $$\ker(T) = \{ \mathbf{v} \in V \mid T(\mathbf{v}) = \mathbf{0} \}$$

  > *This is the abstract version of the Nullspace $N(A)$.*

  * **Image (Range)**: The set of all possible outputs in $W$.

  $$\text{Im}(T) = \{ T(\mathbf{v}) \mid \mathbf{v} \in V \}$$

  > *This is the abstract version of the Column Space $C(A)$.*

### Matrix Determinants

The **determinant** is a scalar value $\det(A)$ associated with an $n \times n$
square matrix $A$.

* For $2 \times 2$: $\det \begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$.
* For $n \times n$, it is computed via cofactor expansion.
* A matrix $A$ is invertible if and only if $\det(A) \neq 0$.

* **Key Properties**:

  * $\det(I) = 1$
  * $\det(AB) = \det(A) \det(B)$
  * $\det(A^T) = \det(A)$
  * $\det(A^{-1}) = 1 / \det(A)$
  * Swapping two rows multiplies the determinant by $-1$.
  * For $A \in \mathbb{R}^{n \times n}$, $\det(cA) = c^n \det(A)$.

> **Geometric Meaning**: $|\det(A)|$ is the volume of the $n$-dimensional parallelepiped formed
> by the column (or row) vectors of $A$.

## II. System Solving and Fundamental Subspaces

### 1. Taylor Approximation

**Differential calculus** provides an organized way to find an
**approximate affine model** of a differentiable function.

Suppose $f : \mathbb{R}^n \to \mathbb{R}$ is **differentiable**, meaning all partial
derivatives exist.

* Let $ z \in \mathbb{R}^n$.  
  The **first-order Taylor approximation** of $f$ near (or at) the point $z$:
  $$\hat{f}(x) = f(z)
  + \frac{\partial f}{\partial x_1}(z)(x_1 - z_1)
  + \cdots
  + \frac{\partial f}{\partial x_n}(z)(x_n - z_n)$$
  where $ \frac{\partial f}{\partial x_i}(z)$ denotes the partial derivative of $f$ with
  respect to its $i$-th argument, evaluated at $z$.
* The hat ($\hat{f}$) indicates that this is an **approximation** of $f$.  
* The approximation $ \hat{f}(x)$ is **accurate when** all $ x_i$ are near the
corresponding $z_i$.
* Sometimes we write the approximation as $\hat{f}(x\mid z)$ to emphasize that it is
developed at the point $z$.
* The **first term**, $f(z)$, is a constant. The **remaining terms** describe how the
function changes as $x$ deviates from $z$.
* The function $\hat{f}$ is **affine in $x$** (sometimes informally called "linear" near $z$).

#### Compact Notation Using the Gradient

Using **inner product notation**, the Taylor approximation can be written as:
$$\hat{f}(x) = f(z) + \nabla f(z)^{\mathsf{T}} (x - z)$$ where the **gradient** of $f$ at
$z$ is:
$$  \nabla f(z) =
  \begin{bmatrix}
  \frac{\partial f}{\partial x_1}(z) \\
  \vdots \\
  \frac{\partial f}{\partial x_n}(z)
  \end{bmatrix}$$

* The first term $ f(z)$ is the function value at $ x = z$  
  * The second term $ \nabla f(z)^{\mathsf{T}} (x - z)$ is the **inner product** between
  the gradient and the **perturbation** $ x - z$.

#### Equivalent Affine Form

- The Taylor approximation can also be expressed as a **linear function plus a constant**:
$$\hat{f}(x) = \nabla f(z)^{\mathsf{T}} x + \big(f(z) - \nabla f(z)^{\mathsf{T}} z\big)$$

- However, the compact form
$$\hat{f}(x) = f(z) + \nabla f(z)^{\mathsf{T}} (x - z)$$
  is often **easier to interpret** geometrically and conceptually.

- The first-order Taylor approximation provides a systematic way to construct an
**affine approximation** of a differentiable function
$f : \mathbb{R}^n \to \mathbb{R}$ near a given point $z$.

- For $ n = 1$, this corresponds to the familiar **tangent line approximation**—accurate
near $z$, but not over large intervals.

### 2. Solving Linear Equations ($A\mathbf{x}=\mathbf{b}$)

* Solve $A\mathbf{x} = \mathbf{b}$
  * find $x_i$ such that $\sum_{i=1}^{n} x_i \mathbf{a}_i = \mathbf{b}$, where
  $\mathbf{a}_i$ are columns of $A$
* **Gaussian Elimination:** $[A | \mathbf{b}] \to [U | \mathbf{c}]$ (Echelon)
$\to [R | \mathbf{d}]$ (Reduced Row Echelon Form, $R = \text{rref}(A)$)

### 3. The Four Fundamental Subspaces (Fundamental Theorem of Linear Algebra, Part I)

For $A \in \mathbb{R}^{m \times n}$ with $\text{rank}(A) = r$, solutions to
$A\mathbf{x} = \mathbf{b}$ depend on:

| Subspace | Definition/Property | Dimension |
| :--- | :--- | :--- |
| **Column Space** $C(A)$ |
$\text{span}\{\mathbf{a}_1, ..., \mathbf{a}_n\} =
\{A\mathbf{x} \mid \mathbf{x} \in \mathbb{R}^n\}$.
Solution exists $\iff \mathbf{b} \in C(A)$. | $r$ |
| **Row Space** $C(A^T)$ | $\text{span}\{\text{rows of } A\}$. | $r$ |
| **Nullspace** $N(A)$ | $\{\mathbf{x} \in \mathbb{R}^n \mid A\mathbf{x} = \mathbf{0}\}$.
| $n-r$ |
| **Left Nullspace** $N(A^T)$ |
$\{\mathbf{y} \in \mathbb{R}^m \mid A^T\mathbf{y} = \mathbf{0}\}$. | $m-r$ |

* **Orthogonality (Part II):**
* $C(A^T) \perp N(A) \implies \mathbb{R}^n = C(A^T) \oplus N(A)$
* $C(A) \perp N(A^T) \implies \mathbb{R}^m = C(A) \oplus N(A^T)$

## III. Eigenvalues, Eigenvectors, and Operators

### 4. Eigenvalues and Eigenvectors

* **Eigenvalue Equation:** $A\mathbf{x} = \lambda\mathbf{x}$, for **eigenvector**
$\mathbf{x} \neq \mathbf{0}$ and **eigenvalue** $\lambda \in \mathbb{C}$
* **Property:** The direction of $\mathbf{x}$ is invariant under $A$; $A$ only scales
$\mathbf{x}$ by $\lambda$
* **Finding $\lambda$:** Solve the characteristic equation:
$p(\lambda) = \det(A - \lambda I) = 0$
* **Diagonalization (EVD):** If $A$ has $n$ linearly independent eigenvectors (columns of
$S$), then $A = S\Lambda S^{-1}$, where $\Lambda = \text{diag}(\lambda_1, ..., \lambda_n)$
* **Spectral Theorem:** If $A = A^T$, then $A = Q\Lambda Q^T$, where $Q$ is orthogonal
($Q^T Q = I$) and $\Lambda$ is real
* **Properties:**
* **Trace:** $\text{Tr}(A) = \sum_{i=1}^{n} a_{ii} = \sum_{i=1}^{n} \lambda_i$
* **Determinant:** $\det(A) = \prod_{i=1}^{n} \lambda_i$
* **Application (ML):** **PCA** uses eigenvectors of the covariance matrix
$\Sigma = \frac{1}{N}X^T X$

## IV. Orthogonality and Least Squares

### 5. Concepts of Orthogonality

* **Inner Product:** $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^T\mathbf{y}$
(for $\mathbb{R}^n$). **Orthogonality:**
$\mathbf{x} \perp \mathbf{y} \iff \mathbf{x}^T\mathbf{y} = 0$
* **Norm:** $||\mathbf{x}||_2 = \sqrt{\mathbf{x}^T\mathbf{x}} = \sqrt{\sum x_i^2}$
* **Orthogonal Matrices ($Q$):** $Q \in \mathbb{R}^{n \times n}$ with
$\mathbf{q}_i^T \mathbf{q}_j = \delta_{ij}$ (orthonormal columns)
$Q^T Q = I \implies Q^T = Q^{-1}$. Preserves norms: $||Q\mathbf{x}||_2 = ||\mathbf{x}||_2$
* **Gram-Schmidt:** Converts basis $\{\mathbf{a}_i\}$ to orthonormal basis
$\{\mathbf{q}_i\}$. Yields $A=QR$

### 6. Least Squares and Projection

* **Motivation:** For $A\mathbf{x}=\mathbf{b}$ with $m > n$ (overdetermined) and
$\mathbf{b} \notin C(A)$ (no solution)
* **Goal:** Find $\mathbf{\hat{x}} = \arg \min_{\mathbf{x}} ||A\mathbf{x} - \mathbf{b}||_2^2$.
* **Normal Equations:** The solution $\mathbf{\hat{x}}$ satisfies
$A^T A\mathbf{\hat{x}} = A^T\mathbf{b}$
* **Projection:** $\mathbf{p} = \text{proj}_{C(A)}\mathbf{b} = A\mathbf{\hat{x}}$.
$\mathbf{p}$ is the closest vector in $C(A)$ to $\mathbf{b}$
* **Projection Matrix:** $\mathbf{p} = P\mathbf{b}$. If $A$ has full column rank,
$P = A(A^T A)^{-1} A^T$
* Error $\mathbf{e} = \mathbf{b} - \mathbf{p}$ satisfies $\mathbf{e} \in N(A^T)$, so
$A^T\mathbf{e} = \mathbf{0}$
* **Pseudoinverse ($A^+$):** Minimum-norm least-squares solution is
$\mathbf{\hat{x}} = A^+\mathbf{b}$. If $\text{rank}(A)=n$, $A^+ = (A^T A)^{-1} A^T$

## V. Essential Matrix Factorizations

| Factorization | Form | Requirements/Purpose |
| :--- | :--- | :--- |
| **Singular Value Decomposition (SVD)** | $A = U\Sigma V^T$ |
$A \in \mathbb{R}^{m \times n}$. $U \in \mathbb{R}^{m \times m}, V \in
\mathbb{R}^{n \times n}$
are orthogonal. $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal, $\Sigma_{ii} =
\sigma_i \ge 0$ (singular values). $\sigma_i = \sqrt{\lambda_i(A^T A)}$. |
| **QR Decomposition** | $A = QR$ | $A \in \mathbb{R}^{m \times n}$ (full column rank).
$Q \in \mathbb{R}^{m \times n}$ ($Q^T Q = I$). $R \in \mathbb{R}^{n \times n}$
(upper triangular). Solves $A\mathbf{x}=\mathbf{b} \to R\mathbf{x} = Q^T\mathbf{b}$. |
| **LU Decomposition** | $A = LU$ (or $PA=LU$) | $A \in \mathbb{R}^{n \times n}$
(invertible). $L$ (lower $\Delta$, $L_{ii}=1$), $U$ (upper $\Delta$). Solves
$A\mathbf{x}=\mathbf{b}$ via $L\mathbf{c}=\mathbf{b}$ then $U\mathbf{x}=\mathbf{c}$. |
| **Cholesky Decomposition** | $A = R^T R$ (or $LL^T$) | $A$ must be
**Symmetric Positive Definite** (SPD). $A = A^T$ and $\mathbf{x}^T A \mathbf{x} > 0$ for
$\mathbf{x} \neq \mathbf{0}$. $R$ is upper $\Delta$, $L$ is lower $\Delta$. |
| **Polar Decomposition** | $A = Q H$ | $A \in \mathbb{R}^{n \times n}$ (invertible). $Q$
is orthogonal ($Q^T Q = I$), $H$ is SPD ($H = \sqrt{A^T A}$). Separates rotation ($Q$)
from stretch ($H$). |

## VI. Advanced Topics: Tensors (Multi-linear Algebra)

### 7. Tensors in Engineering (ML/Controls/Robotics)

* **Definition:** A tensor $\mathcal{A} \in F^{I_1 \times I_2 \times \dots \times I_K}$
is a multi-way array of order $K$
* $K=1 \implies \text{vector}$
* $K=2 \implies \text{matrix}$
* **Motivation:** Avoids $\text{vec}(\mathcal{A})$ (vectorization), which loses multi-modal
structure
* **Multi-linear Extensions:** Generalize LA concepts using tensor products (e.g., Einstein
product $*_N$, t-product $*$)
* **Tensor Decompositions:**
* **Tensor EVD:** $\mathcal{A} * \mathcal{U} = \mathcal{U} * \mathcal{D}$. $\mathcal{D}$
holds eigenvalues $\lambda_i$, $\mathcal{U}$ holds eigentensors
$\mathcal{A} \succeq 0 \iff \lambda_i \ge 0$ for all $i$
* **Tensor SVD (t-SVD):** $\mathcal{A} = \mathcal{U} * \mathcal{S} * \mathcal{V}^T$
$\mathcal{U}, \mathcal{V}$ are orthogonal tensors, $\mathcal{S}$ is f-diagonal (frontal
slices are diagonal)
* **Tensor LU:** $\mathcal{A} = \mathcal{L} * \mathcal{U}$. Solves multi-linear systems
$\mathcal{A} * \mathcal{X} = \mathcal{B}$ via tensor forward/backward substitution

```rust
use nalgebra::{DMatrix, SVD};

/// Implements Dynamic Mode Decomposition with Control (DMDc).
///
/// This function discovers the best-fit linear system matrices (A, B) that
/// approximate the dynamics `x' ≈ Ax + Bu` given a time-series history of
/// state vectors `x_history` and control vectors `u_history`.
///
/// `x_history` is an `n x m` matrix, where `n` is the state dimension and `m` is the
/// number of snapshots.
/// `u_history` is a `p x m` matrix, where `p` is the control dimension and `m` is the
/// number of snapshots.
///
/// This is a practical application of the concepts above:
/// 1.  It solves a linear least-squares problem to find `[A, B]`.
/// 2.  It uses Singular Value Decomposition (SVD), which is built from
///     eigenvectors, to find the pseudoinverse robustly.
/// 3.  The resulting `A` and `B` matrices represent the **Linear Transformations** (Sec III.0)
///     that govern the system's dynamics.
fn dmdc(
    x_history: &DMatrix<f64>,
    u_history: &DMatrix<f64>,
) -> Result<(DMatrix<f64>, DMatrix<f64>), &'static str> {

    let n = x_history.nrows(); // State dimension
    let p = u_history.nrows(); // Control dimension
    let m = x_history.ncols(); // Number of snapshots

    if m != u_history.ncols() {
        return Err("x_history and u_history must have the same number of columns.");
    }
    if m < 2 {
        return Err("At least 2 snapshots are required.");
    }

    // 1. Create snapshot matrices X and X'
    // X = [x_1, x_2, ..., x_{m-1}]
    // X' = [x_2, x_3, ..., x_m]
    let x = x_history.columns(0, m - 1);
    let x_prime = x_history.columns(1, m - 1);

    // 2. Create control matrix U
    // U = [u_1, u_2, ..., u_{m-1}]
    let u = u_history.columns(0, m - 1);

    // 3. Form the augmented matrix Ω = [X; U]
    // This is a form of **Matrix Addition** (Sec II.I) / concatenation.
    let mut omega = DMatrix::<f64>::zeros(n + p, m - 1);
    omega.rows_mut(0, n).copy_from(&x);
    omega.rows_mut(n, p).copy_from(&u);

    // 4. Solve the least-squares problem X' ≈ G * Ω, where G = [A, B]
    // We want to find G that minimizes ||X' - GΩ||.
    // The solution is G = X' * Ω⁺ (where Ω⁺ is the pseudoinverse).
    // This is solved most stably using SVD.

    // SVD computes the pseudoinverse, which is related to the **Matrix Inverse** (Sec II.I).
    // The SVD itself finds the principal components, which form a **Basis** (Sec I.V)
    // for the input space.
    let svd = omega.transpose().svd(true, true);

    // Solves Ωᵀ * Gᵀ = X'ᵀ for Gᵀ
    let g_transpose = svd
        .solve(&x_prime.transpose(), 1e-10) // 1e-10 is a tolerance for singular values
        .map_err(|_| "SVD solve failed. Matrix may be singular.")?;

    // G = (Gᵀ)ᵀ
    // This uses the property (Aᵀ)ᵀ = A.
    let g = g_transpose.transpose();

    // 5. Extract A and B from G = [A, B]
    // `A` is the first `n` columns, `B` is the next `p` columns.
    // The `A` and `B` matrices are the **Matrix of a Transformation** (Sec III.0).
    let a = g.columns(0, n).clone_owned();
    let b = g.columns(n, p).clone_owned();

    // The next step in a real analysis would be to find the eigenvalues
    // of `A` (using **Determinants**, Sec III.V) to study the
    // stability and modes of the system.

    Ok((a, b))
}

// Example Usage (requires `nalgebra` crate in Cargo.toml):
fn main() {
    // n=2 states, p=1 control, m=6 snapshots
    let x_hist = DMatrix::from_vec(2, 6, vec![
        1.0, 0.0,  // x1
        1.0, 0.1,  // x2
        0.9, 0.2,  // x3
        0.7, 0.3,  // x4
        0.4, 0.4,  // x5
        0.0, 0.5   // x6
    ]);

    let u_hist = DMatrix::from_vec(1, 6, vec![
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5 // u1 to u6
    ]);

    match dmdc(&x_hist, &u_hist) {
        Ok((a, b)) => {
            println!("Found A matrix:\n{}", a);
            println!("Found B matrix:\n{}", b);
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}
```
