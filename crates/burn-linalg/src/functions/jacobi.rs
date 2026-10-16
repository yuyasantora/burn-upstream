use crate::Tensor;

/// Stopping criteria for iterative linear solvers.
#[derive(Debug, Clone, Copy)]
pub struct IterativeSolverOptions {
    /// Relative residual tolerance.
    ///
    /// The solver stops once `||b - A x|| / ||b|| < tol` holds for every system in the batch.
    /// Values below the machine epsilon of the input dtype may never be reached.
    pub tol: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
}

/// Result of an iterative linear solver.
#[derive(Debug, Clone)]
pub struct IterativeSolverOutput<const DV: usize> {
    /// Approximate solution `x` of `A x = b`.
    pub x: Tensor<DV>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Relative residual `||b - A x|| / ||b||` of each system, with the last dimension of size 1.
    pub residual: Tensor<DV>,
    /// Whether every system in the batch reached `tol` within `max_iter` iterations.
    ///
    /// `false` also when the iteration produced NaN, for example from a zero diagonal entry
    /// or a system with `b = 0`.
    pub converged: bool,
}
