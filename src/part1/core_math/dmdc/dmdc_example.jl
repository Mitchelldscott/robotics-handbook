### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# ╔═╡ 63a93b28-3d59-471c-9564-eb1875fae9b4
# We use LinearAlgebra for the robust least-squares solve (pseudoinverse)
using LinearAlgebra

# ╔═╡ f2691025-68f2-4255-bbc3-808efc539dac
"""
    dmdc(x_history::AbstractMatrix, u_history::AbstractMatrix) -> A, B

Implements Dynamic Mode Decomposition with Control (DMDc).

This function discovers the best-fit linear system matrices (A, B) that
approximate the dynamics `x' ≈ Ax + Bu` given a time-series history of
state vectors `x_history` and control vectors `u_history`.

Inputs:
- `x_history`: An `n x m` matrix, where `n` is the state dimension and `m` is the
  number of samples. Contains the sequence x(0), x(1), ..., x(m-1).
- `u_history`: A `p x m` matrix, where `p` is the control dimension and `m` is the
  number of samples. Contains the sequence u(0), u(1), ..., u(m-1).

Outputs:
- `A`: The `n x n` dynamic matrix (linear operator).
- `B`: The `n x p` control matrix.

The method solves the linear least-squares problem:
X₂ ≈ [A | B] * Ω, where Ω = [X₁; U]
"""
function dmdc(x_history::AbstractMatrix, u_history::AbstractMatrix)
    # 1. Define the data matrices
    # The current states (X₁) and controls (U) map to the next states (X₂).
    # We use all time steps except the last for the input side (X₁ and U).
    # We use all time steps except the first for the output side (X₂).

    # X₁: States x(0) to x(m-2)
    X₁ = x_history[:, 1:end-1]
    # X₂: States x(1) to x(m-1)
    X₂ = x_history[:, 2:end]
    # U: Controls u(0) to u(m-2)
    U = u_history[:, 1:end-1]

    # Check for consistent time steps
    if size(X₁)[2] != size(U)[2] || size(X₁)[2] != size(X₂)[2]
		println(size(X₁), size(X₂), size(U))
        error(
			"Input matrices X, U and output matrix X' must have the same number of columns."
		)
    end

    # 2. Form the concatenated input matrix Ω (Psi in some literature)
    # Ω = [X₁; U] has shape (n + p) x (m - 1)
    # This matrix contains the full set of inputs to the system at each step.
    Ω = vcat(X₁, U)

    # 3. Solve the least-squares problem: X₂ ≈ G * Ω
    # We want to find G = [A | B].
    # In Julia, the backslash operator `\` is the robust way to solve linear
    # least-squares: G = X₂ * pinv(Ω). 
	# The backslash operator handles this efficiently.
    G = X₂ / Ω  # G is (n) x (n + p)

    # 4. Extract A and B
    n = size(X₁, 1) # State dimension
    p = size(U, 1)  # Control dimension

    A = G[:, 1:n]      # First 'n' columns of G are the A matrix (n x n)
    B = G[:, n+1:end]  # Remaining 'p' columns of G are the B matrix (n x p)

    return A, B
end

# ==============================================================================
# Example System Dynamics and Data Generation
# ==============================================================================

# Define a simple 2D linear system with a 1D control input.
# True system: x(k+1) = A_true * x(k) + B_true * u(k)

# ╔═╡ ed2f4f7b-1c9f-4a5a-af19-e2a41569c0b2
const A_true = [0.9 0.1; 0.0 0.85];

# ╔═╡ 480654a4-da3a-40ab-891d-dd05aa7a1e26
const B_true = [0.5; 0.2];

# ╔═╡ 7b492ed8-7b4e-476a-960c-a97f939aa3b6
const n_states = size(A_true, 1);

# ╔═╡ bb85321b-f3f5-4cb5-a965-f264e82018fe
const n_controls = size(B_true, 2);

# ╔═╡ ec7451c7-cf43-4a88-9c7a-37ab5e5c60fd
const n_samples = 1000; # Number of time steps to simulate

# ╔═╡ 4db5617c-dc84-464b-90eb-857a7b0a8c82
const dt = 1.0;         # Time step (DMDc is inherently discrete-time)

# ╔═╡ 4ad3582c-7b8e-4b5d-984c-8a1897498d72
println("True A Matrix"); display(A_true);

# ╔═╡ 05ce2c35-dc42-4e1a-b688-9c5eded65964
println("True B Matrix"); display(B_true);

# ╔═╡ 3ec4f805-cef9-4535-bb3e-528ae6caa4eb
x_history = zeros(n_states, n_samples);

# ╔═╡ 5de66551-8c60-46c2-bcfa-6c342b7f9962
u_history = zeros(n_controls, n_samples);

# ╔═╡ 166eadca-a1d1-4531-984e-775a7c85f303
x_history[:, 1] = [10.0; -5.0]; # Initial state

# ╔═╡ 98a33af4-3c85-412d-b550-b252fb1e47a0
# Simulate the system to generate training data
for k in 1:n_samples-1
    # 1. Generate a random control input (u(k) is between -1 and 1)
    u_k = rand(1, n_controls) * 2 .- 1
    u_history[:, k] = u_k

    # 2. Compute the next state x(k+1)
    x_k = x_history[:, k]
    x_k_plus_1 = A_true * x_k + B_true * u_k

    # 3. Add small measurement noise (simulating real-world data)
    noise = randn(n_states) * 0.01
    x_history[:, k+1] = x_k_plus_1 + noise
end

# Set the last control input to zero (it's not used in the DMDc calculation)

# ╔═╡ 2860253f-bd0a-4d28-a301-3df010e2181e
u_history[:, end] .= 0.0;

# ╔═╡ 4c894550-e89a-4a47-8bc7-09199129ca2f
# ==============================================================================
# Run DMDc on the Generated Data
# ==============================================================================
A_dmdc, B_dmdc = dmdc(x_history, u_history)

# ╔═╡ e8fe6264-4d32-4f64-b0ea-c5f5bd90b98c
println("Recovered A Matrix"); display(round.(A_dmdc, digits=4))

# ╔═╡ 9072f392-bcdd-4920-a92b-4ec54d905d6f
println("Recovered B Matrix"); display(round.(B_dmdc, digits=4))

# ╔═╡ baf4369c-6aef-4fe7-b6cd-ec5f14ee8588
# Calculate the error (norm of the difference)
A_error = norm(A_true - A_dmdc) / norm(A_true)

# ╔═╡ 2575f59f-e938-4d19-935e-6df22930168a
B_error = norm(B_true - B_dmdc) / norm(B_true)

# ╔═╡ fb42d9b5-1545-4d03-8d15-763053c658bb
print("Relative Error in A: "); display(round(A_error, digits=6))

# ╔═╡ 035b7083-b793-45e6-9fac-4d2037e8e7e4
print("Relative Error in B: "); display(round(B_error, digits=6))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.7"
manifest_format = "2.0"
project_hash = "ac1187e548c6ab173ac57d4e72da1620216bce54"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"
"""

# ╔═╡ Cell order:
# ╠═63a93b28-3d59-471c-9564-eb1875fae9b4
# ╠═f2691025-68f2-4255-bbc3-808efc539dac
# ╠═ed2f4f7b-1c9f-4a5a-af19-e2a41569c0b2
# ╠═480654a4-da3a-40ab-891d-dd05aa7a1e26
# ╠═7b492ed8-7b4e-476a-960c-a97f939aa3b6
# ╠═bb85321b-f3f5-4cb5-a965-f264e82018fe
# ╠═ec7451c7-cf43-4a88-9c7a-37ab5e5c60fd
# ╠═4db5617c-dc84-464b-90eb-857a7b0a8c82
# ╠═4ad3582c-7b8e-4b5d-984c-8a1897498d72
# ╠═05ce2c35-dc42-4e1a-b688-9c5eded65964
# ╠═3ec4f805-cef9-4535-bb3e-528ae6caa4eb
# ╠═5de66551-8c60-46c2-bcfa-6c342b7f9962
# ╠═166eadca-a1d1-4531-984e-775a7c85f303
# ╠═98a33af4-3c85-412d-b550-b252fb1e47a0
# ╠═2860253f-bd0a-4d28-a301-3df010e2181e
# ╠═4c894550-e89a-4a47-8bc7-09199129ca2f
# ╠═e8fe6264-4d32-4f64-b0ea-c5f5bd90b98c
# ╠═9072f392-bcdd-4920-a92b-4ec54d905d6f
# ╠═baf4369c-6aef-4fe7-b6cd-ec5f14ee8588
# ╠═2575f59f-e938-4d19-935e-6df22930168a
# ╠═fb42d9b5-1545-4d03-8d15-763053c658bb
# ╠═035b7083-b793-45e6-9fac-4d2037e8e7e4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
