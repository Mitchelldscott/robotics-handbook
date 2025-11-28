# Introduction to Applied Linear Algebra and Differential Equations

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

A **matrix** is a rectangular array of numbers, e.g.:
  $$\begin{bmatrix}
    a_{11} & a_{12} & a_{13} & a_{14} \\
    a_{21} & a_{22} & a_{23} & a_{24} \\
    a_{31} & a_{32} & a_{33} & a_{34} \\
  \end{bmatrix}$$

The **size** of a matrix is *(rows, columns)*. *Example: The matrix above is $3 \times 4$*

* Two matrices are **equal** ($A = B$) if:
  * They are the same size, and  
  * All corresponding entries are equal.

* Common Matrix Types:
  * **Tall** $m > n$
  * **Wide** $m < n$
  * **Square** $m = n$
* Special Cases
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

#### Determinants

The **determinant** is a scalar value $\det(A)$ associated with an $n \times n$
square matrix $A$.

Let

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nn}
\end{bmatrix}
$$

The determinant of $A$, denoted $\det(A)$ or $|A|$, is defined by the **Leibniz formula**:

$$
\det(A)
= \sum_{\sigma \in S_n} \operatorname{sgn}(\sigma)
\prod_{i=1}^{n} a_{i, \sigma(i)}
$$

where:

* $S_n$ is the set of all permutations of ${1, 2, \dots, n}$,
* $\operatorname{sgn}(\sigma)$ is $+1$ for even and $-1$ for odd permutations.

Equivalently, it can be computed recursively via **cofactor expansion** along any row or column:

$$
\det(A)
= \sum_{j=1}^{n} (-1)^{1+j} a_{1j} \det(A_{1j})
$$

where ( A_{1j} ) is the ((n-1)\times(n-1)) submatrix obtained by removing the first row and
$j^{th}$ column.

##### **Properties**

For any $n\times n$ matrices $A, B$ and scalar $c$:

1. $\det(AB) = \det(A)\det(B)$
2. $\det(A^\top) = \det(A)$
3. $\det(cA) = c^n \det(A)$
4. $\det(A) = 0 \iff A \text{ is singular (non-invertible)}$

The determinant is **multilinear** and **alternating** in the rows (or columns):

* Linearity: scaling or adding rows scales/adds determinants accordingly.
* Alternation: if two rows are identical, $\det(A) = 0$.

### Jacobians

For a function $ \mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m $,

$$
J_{\mathbf{f}}(\mathbf{x}) =
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

Each row is the gradient of one output component $ f_i $;  
$ J_{\mathbf{f}} $ is the best linear approximation of $ \mathbf{f} $ near $ \mathbf{x} $.

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

For $A \in \mathbb{R}^{m \times n}$ with rank $r$:

| Subspace | Definition | Dim |
| :-- | :-- | :--: |
| **Column Space** $C(A)$ | $\{A\mathbf{x} \mid \mathbf{x} \in \mathbb{R}^n\}$ | $r$ |
| **Row Space** $C(A^\top)$ | Span of rows of $A$ | $r$ |
| **Nullspace** $N(A)$ | $\{\mathbf{x} \mid A\mathbf{x}=0\}$ | $n - r$ |
| **Left Nullspace** $N(A^\top)$ | $\{\mathbf{y} \mid A^\top\mathbf{y}=0\}$ | $m - r$ |

**Orthogonality:**  
$C(A^\top) \perp N(A)$, $C(A) \perp N(A^\top)$  
$\Rightarrow \mathbb{R}^n = C(A^\top) \oplus N(A)$, $\mathbb{R}^m = C(A) \oplus N(A^\top)$

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
($Q^T Q = I$) and $\Lambda$ is real.

## IV. Orthogonality and Least Squares

### 5. Concepts of Orthogonality

$$\mathbf{x} \perp \mathbf{y} \iff \mathbf{x}^T\mathbf{y} = 0$$

* **Orthogonal Matrix ($Q$):** (A square $n \times n$ matrix)
    * **Definition:** Has orthonormal columns ($\mathbf{q}_i^T \mathbf{q}_j = \delta_{ij}$).
    * **Property:** $Q^T Q = I$
    * **Consequence:** $Q^T = Q^{-1}$
    * **Preserves Norms:** $||Q\mathbf{x}||_2 = ||\mathbf{x}||_2$
* **Gram-Schmidt:** An algorithm that converts a set of linearly independent vectors
$\{\mathbf{a}_i\}$ into an orthonormal set $\{\mathbf{q}_i\}$. This is the foundation of
the $A=QR$ decomposition.

### 6. Least Squares and Projection

This section addresses solving an **overdetermined system** $A\mathbf{x}=\mathbf{b}$
(where $m > n$ and $\mathbf{b}$ is not in $C(A)$) by finding the "best fit" solution
$\mathbf{\hat{x}}$ that minimizes the squared error $||A\mathbf{x} - \mathbf{b}||_2^2$.

* **Normal Equations:** The equation to solve for the least-squares solution $\mathbf{\hat{x}}$:
    $$A^T A\mathbf{\hat{x}} = A^T\mathbf{b}$$
* **Projection ($\mathbf{p}$):** The vector $\mathbf{p}$ is the projection of
$\mathbf{b}$ onto the column space $C(A)$. It's the closest vector in $C(A)$ to $\mathbf{b}$.
    $$\mathbf{p} = A\mathbf{\hat{x}}$$

* **Error Vector ($\mathbf{e}$):** The residual vector, which is orthogonal to $C(A)$.
    $$\mathbf{e} = \mathbf{b} - \mathbf{p}$$
    * **Property:** $\mathbf{e} \in N(A^T)$, which means $A^T\mathbf{e} = \mathbf{0}$.
* **Projection Matrix ($P$):** The matrix that projects any vector onto $C(A)$
($\mathbf{p} = P\mathbf{b}$).
    *(if $A$ has full column rank)*: $$P = A(A^T A)^{-1} A^T$$
* **Pseudoinverse ($A^+$):** Gives the minimum-norm, least-squares solution.
    $$\mathbf{\hat{x}} = A^+\mathbf{b}$$
    *(if $A$ has full column rank)*: $$A^+ = (A^T A)^{-1} A^T$$

## V. Essential Matrix Factorizations

### **Singular Value Decomposition (SVD)**

```julia
function svd(A):
    // INPUT: A (an m x n matrix)
    //
    // REQUIREMENTS: None. Works for any m x n matrix.
    //
    // OUTPUTS: U, S, Vt
    //   U: m x m orthogonal matrix
    //   S: m x n diagonal matrix with non-negative singular values (sigma_i)
    //      (Often returned as a 1D vector of singular values)
    //   Vt: n x n orthogonal matrix (V-transpose)
    //
    // FORM: A = U * S * Vt

    (U, S, Vt) = compute_svd(A)
    return (U, S, Vt)
```

### **QR Decomposition**

```julia
function qr(A):
    // INPUT: A (an m x n matrix, usually with m >= n)
    //
    // REQUIREMENTS: Assumes A has full column rank (linearly independent columns)
    //               for the "thin" decomposition described.
    //
    // OUTPUTS: Q, R
    //   Q: m x n matrix with orthonormal columns (Q^T * Q = I)
    //   R: n x n upper triangular matrix
    //
    // FORM: A = Q * R

    (Q, R) = compute_qr(A)
    return (Q, R)
```

### **LU Decomposition (with Pivoting)**

```julia
function lu(A):
    // INPUT: A (an n x n square matrix)
    //
    // REQUIREMENTS: A must be invertible (non-singular).
    //               We use the PA=LU form for numerical stability
    //               and to handle all invertible cases.
    //
    // OUTPUTS: P, L, U
    //   P: n x n permutation matrix (tracks row swaps)
    //   L: n x n lower triangular matrix (with 1s on the diagonal)
    //   U: n x n upper triangular matrix
    //
    // FORM: P * A = L * U

    (P, L, U) = compute_lu_with_pivoting(A)
    return (P, L, U)
```

### **Cholesky Decomposition**

```julia
function cholesky(A):
    // INPUT: A (an n x n square matrix)
    //
    // REQUIREMENTS: A MUST be Symmetric Positive Definite (SPD).
    //   1. Symmetric: A = A^T
    //   2. Positive Definite: x^T * A * x > 0 for all non-zero vectors x
    //
    // OUTPUTS: L (or R)
    //   L: n x n lower triangular matrix
    //
    // FORM: A = L * L^T  (if returning L)
    //   or
    //   R: n x n upper triangular matrix
    // FORM: A = R^T * R  (if returning R)

    if not is_symmetric(A) or not is_positive_definite(A):
        error "Matrix is not Symmetric Positive Definite."

    L = compute_cholesky(A) // This computes the lower triangular L
    return L
```

### **Polar Decomposition**

```julia
function polar(A):
    // INPUT: A (an n x n square matrix)
    //
    // REQUIREMENTS: A must be invertible (non-singular).
    //
    // OUTPUTS: Q, H
    //   Q: n x n orthogonal matrix (rotation/reflection)
    //   H: n x n Symmetric Positive Definite (SPD) matrix (stretch)
    //
    // FORM: A = Q * H

    (Q, H) = compute_polar(A)
    return (Q, H)
```

## DMD Example

> Suppose that we are studying a dynamical system defined by $x_{j+1} = F(x_j)$
> for an unknown function \(F\). Given a collection of measurements $\{x_1, \dots, x_m\}$,
> Dynamic Mode Decomposition (DMD) identifies the best low-rank linear approximation of $F$.
> In other words, DMD seeks a low-rank matrix $A$ such that
> $x_{j+1} \approx A x_j, \quad j = 1, \dots, m$. Arranging the data measurements into
> matrices $X$ and $Y$ (see (2.1) for details) allows us to phrase the above formally as
$$\argmin_{\operatorname{rank}(A) \le r} \| Y - AX \|_F\tag{1.1}$$ After approximately
solving (1.1), the DMD process computes the dominant spectral properties of the learned
linear operator. @@badoo2023pidmd

<!-- Start of HTML iframe embed -->
<iframe
    src="dmdc/dmdc_example.html"
    width="100%"
    height="800px"
    style="border: none;"
>
    Your browser does not support iframes. Please view the static notebook directly.
</iframe>
<!-- End of HTML iframe embed -->

## Based on notes taken from:
- @@boyd2018introduction
- @@nathan_kutz_dynamic_2018
