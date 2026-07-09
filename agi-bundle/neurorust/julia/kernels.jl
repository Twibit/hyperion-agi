# Julia companion module for neurorust.
#
# Loaded by `src/ffi/julia.rs` when the `julia` cargo feature is enabled.
# Kept as plain Julia code so it can also be exercised interactively:
#
#     julia> include("julia/kernels.jl")
#     julia> using .NeuroKernels
#     julia> NeuroKernels.solve_tridiag([-1.0,-1.0], [2.0,2.0,2.0], [-1.0,-1.0], [1.0,0.0,1.0])

module NeuroKernels

using LinearAlgebra
using SparseArrays

"""
    solve_tridiag(sub, diag, sup, b) -> Vector{Float64}

Solve `T x = b` where `T` is tridiagonal with sub-, main-, super-diagonals
`sub`, `diag`, `sup`. Dispatches to LAPACK `gtsv!` under the hood, giving
optimal O(n) performance.
"""
function solve_tridiag(sub::Vector{Float64}, diag::Vector{Float64},
                       sup::Vector{Float64}, b::Vector{Float64})
    T = Tridiagonal(sub, diag, sup)
    return T \ b
end

"""
    cable_step(V, g_ax, g_leak, e_leak, c_m, dt, i_inj) -> Vector{Float64}

Advance a single cable-equation time step for `N` compartments using an
implicit Crank–Nicolson scheme (unconditionally stable). Returns the new
voltage vector.
"""
function cable_step(V::Vector{Float64}, g_ax::Float64, g_leak::Float64,
                    e_leak::Float64, c_m::Float64, dt::Float64,
                    i_inj::Vector{Float64})
    n = length(V)
    α = g_ax * dt / (2 * c_m)
    β = g_leak * dt / (2 * c_m)
    # Tridiagonal (I − α L + β I) V^{n+1} = (I + α L − β I) V^{n} + dt/c_m i_inj
    sub  = fill(-α, n - 1)
    sup  = fill(-α, n - 1)
    main = fill(1.0 + 2α + β, n)
    main[1]   = 1.0 + α + β        # Neumann BC at soma
    main[end] = 1.0 + α + β        # Sealed dendrite tip
    rhs = similar(V)
    @inbounds for i in 1:n
        left  = i == 1 ? V[i] : V[i-1]
        right = i == n ? V[i] : V[i+1]
        rhs[i] = V[i] * (1.0 - 2α - β) + α * (left + right) +
                 β * e_leak + dt * i_inj[i] / c_m
    end
    return solve_tridiag(sub, main, sup, rhs)
end

"""
    spectral_radius(W::SparseMatrixCSC) -> Float64

Largest eigenvalue modulus — a useful network-stability diagnostic.
"""
function spectral_radius(W::SparseMatrixCSC{Float64,Int})
    vals = eigen(Matrix(W)).values
    return maximum(abs.(vals))
end

end # module
