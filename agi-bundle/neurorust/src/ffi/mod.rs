//! FFI layer for heavy numerical kernels.
//!
//! ## Design
//!
//! - **`pure_rust`** always compiles. Provides reference implementations of
//!   every kernel using `nalgebra`.
//! - **`julia`** is gated by the `julia` cargo feature. When enabled it calls
//!   into Julia via `jlrs` and executes the same kernels on Julia's LAPACK
//!   stack (OpenBLAS/MKL). This is the recommended path for large matrix
//!   solves, eigen-decompositions, and cable-equation tridiagonal factorings.
//!
//! Both submodules expose an identical function signature so client code can
//! feature-toggle without touching call sites — see [`solve_tridiag`].

pub mod pure_rust;

#[cfg(feature = "julia")]
pub mod julia;

/// Solve a tridiagonal linear system `T x = b` via the fastest backend
/// available at compile time. Feature `julia` picks the Julia LAPACK route;
/// otherwise this dispatches to [`pure_rust::solve_tridiag`].
///
/// `sub`, `diag`, `sup` are the sub-, main-, and super-diagonals.
pub fn solve_tridiag(sub: &[f64], diag: &[f64], sup: &[f64], b: &[f64]) -> Vec<f64> {
    #[cfg(feature = "julia")]
    { return julia::solve_tridiag(sub, diag, sup, b); }
    #[cfg(not(feature = "julia"))]
    { pure_rust::solve_tridiag(sub, diag, sup, b) }
}
