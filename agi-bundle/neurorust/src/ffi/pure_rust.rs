//! Reference pure-Rust numerical kernels (used when `julia` feature is off).

/// Thomas algorithm for tridiagonal systems: O(n) time, O(n) space.
///
/// Solves `T x = b` where `T` has sub-diagonal `sub` (len n-1), main diagonal
/// `diag` (len n), super-diagonal `sup` (len n-1).
pub fn solve_tridiag(sub: &[f64], diag: &[f64], sup: &[f64], b: &[f64]) -> Vec<f64> {
    let n = diag.len();
    assert_eq!(sub.len(), n - 1);
    assert_eq!(sup.len(), n - 1);
    assert_eq!(b.len(), n);

    let mut c_star = vec![0.0; n - 1];
    let mut d_star = vec![0.0; n];

    c_star[0] = sup[0] / diag[0];
    d_star[0] = b[0] / diag[0];

    for i in 1..n {
        let m = diag[i] - sub[i - 1] * c_star.get(i - 1).copied().unwrap_or(0.0);
        if i < n - 1 {
            c_star[i] = sup[i] / m;
        }
        d_star[i] = (b[i] - sub[i - 1] * d_star[i - 1]) / m;
    }

    let mut x = vec![0.0; n];
    x[n - 1] = d_star[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_star[i] - c_star[i] * x[i + 1];
    }
    x
}
