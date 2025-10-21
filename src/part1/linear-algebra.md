# Linear Algebra

## I. Foundations and Data Representation

### 1. Vector Spaces & Data Representation

#### Vector spaces

* $(V, +, \cdot)$ over a field $F$ is a set $V$ with operations
  * addition $V + V \to V$
    1. $a + b = b + a$ (commutative)
    2. $(a + b) + c = a + (b + c)$ (associative)
    3. $a + 0 = 0 + a = a$ (zero, identity)
  * scalar multiplication $F \times V \to V$
    1. $(\beta \gamma) a = \beta (\gamma a)$ (associative)
    2. $(\beta + \gamma) a = \beta a + \gamma a$ (left distributive)
    3. $\beta (a + b) a = \beta a + \beta b$ (right distributive)
  * inner-product $V \cdot V \to F$$
    1. $a^{T} b = b^{T} a$
    2. $(\gamma a)^{T} b = \gamma (a^{T} b)$
    3. $(a + b)^T c = a^T c + b^T c$

* **Linear Map:** $T: V \to W$ s.t.
$T(c\mathbf{v} + d\mathbf{w}) = cT(\mathbf{v}) + dT(\mathbf{w})$ for
$c, d \in F$ and $\mathbf{v}, \mathbf{w} \in V$
* **Operator:** A linear map $T: V \to V$

## II. System Solving and Fundamental Subspaces

### 2. Solving Linear Equations ($A\mathbf{x}=\mathbf{b}$)

* Data problems $\implies$ **Linear Equations** $\implies A\mathbf{x}=\mathbf{b}$
* Solve $A\mathbf{x} = \mathbf{b}$: find $x_i$ such that
$\sum_{i=1}^{n} x_i \mathbf{a}_i = \mathbf{b}$, where $\mathbf{a}_i$ are columns of $A$
* **Gaussian Elimination:** $[A | \mathbf{b}] \to [U | \mathbf{c}]$ (Echelon)
$\to [R | \mathbf{d}]$ (Reduced Row Echelon Form, $R = \text{rref}(A)$)

### 3. The Four Fundamental Subspaces (Fundamental Theorem of Linear Algebra, Part I)

For $A \in \mathbb{R}^{m \times n}$ with $\text{rank}(A) = r$, solutions to
$A\mathbf{x} = \mathbf{b}$ depend on:

| Subspace | Definition/Property | Dimension |
| :--- | :--- | :--- |
| **Column Space** $C(A)$ | $\text{span}\{\mathbf{a}_1, ..., \mathbf{a}_n\} = \{A\mathbf{x} \mid \mathbf{x} \in \mathbb{R}^n\}$. Solution exists $\iff \mathbf{b} \in C(A)$. | $r$ |
| **Row Space** $C(A^T)$ | $\text{span}\{\text{rows of } A\}$. | $r$ |
| **Nullspace** $N(A)$ | $\{\mathbf{x} \in \mathbb{R}^n \mid A\mathbf{x} = \mathbf{0}\}$. | $n-r$ |
| **Left Nullspace** $N(A^T)$ | $\{\mathbf{y} \in \mathbb{R}^m \mid A^T\mathbf{y} = \mathbf{0}\}$. | $m-r$ |

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
| **Singular Value Decomposition (SVD)** | $A = U\Sigma V^T$ | $A \in \mathbb{R}^{m \times n}$. $U \in \mathbb{R}^{m \times m}, V \in \mathbb{R}^{n \times n}$ are orthogonal. $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal, $\Sigma_{ii} = \sigma_i \ge 0$ (singular values). $\sigma_i = \sqrt{\lambda_i(A^T A)}$. |
| **QR Decomposition** | $A = QR$ | $A \in \mathbb{R}^{m \times n}$ (full column rank). $Q \in \mathbb{R}^{m \times n}$ ($Q^T Q = I$). $R \in \mathbb{R}^{n \times n}$ (upper triangular). Solves $A\mathbf{x}=\mathbf{b} \to R\mathbf{x} = Q^T\mathbf{b}$. |
| **LU Decomposition** | $A = LU$ (or $PA=LU$) | $A \in \mathbb{R}^{n \times n}$ (invertible). $L$ (lower $\Delta$, $L_{ii}=1$), $U$ (upper $\Delta$). Solves $A\mathbf{x}=\mathbf{b}$ via $L\mathbf{c}=\mathbf{b}$ then $U\mathbf{x}=\mathbf{c}$. |
| **Cholesky Decomposition** | $A = R^T R$ (or $LL^T$) | $A$ must be **Symmetric Positive Definite** (SPD). $A = A^T$ and $\mathbf{x}^T A \mathbf{x} > 0$ for $\mathbf{x} \neq \mathbf{0}$. $R$ is upper $\Delta$, $L$ is lower $\Delta$. |
| **Polar Decomposition** | $A = Q H$ | $A \in \mathbb{R}^{n \times n}$ (invertible). $Q$ is orthogonal ($Q^T Q = I$), $H$ is SPD ($H = \sqrt{A^T A}$). Separates rotation ($Q$) from stretch ($H$). |

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
