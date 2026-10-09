# Optimization

Course work from **AMATH 515: Optimization** at the University of Washington (Applied and Computational Mathematics, Winter 2025). Computational homeworks with full research-paper-style reports covering first and second-order optimization algorithms and proximal methods for non-smooth objectives.

## Homeworks

### [`hw4-optimization-algorithms/`](./hw4-optimization-algorithms/)
**Convergence Analysis: Steepest Descent, Newton, DFP, and BFGS**

Empirical comparison of four optimization algorithms on convex and non-convex objectives. All methods follow the general descent update `x_{k+1} = x_k + α_k d_k = -B(x_k) ∇f(x_k)` with step size determined by bisection line search satisfying the Wolfe conditions. `B` distinguishes the methods: identity (steepest descent), exact inverse Hessian (Newton), and rank-2 quasi-Newton Hessian approximations (DFP, BFGS).

Tested on three functions: Rosenbrock (non-convex, non-Lipschitz gradient), Logistic Regression on MNIST binary classification (convex, varying regularization λ), and Trid (strongly coupled variables, convergence analyzed across dimensions n=2,5,10,50). Key findings: Newton achieves quadratic convergence on convex problems but per-step cost is high; BFGS is more robust than DFP on sigmoidal loss; Wolfe parameter sensitivity differs significantly by algorithm.

Key methods: Wolfe line search via bisection, steepest descent, Newton's method, DFP, BFGS, convergence analysis on convex and non-convex functions.

---

### [`hw7-fista-image-recovery/`](./hw7-fista-image-recovery/)
**Image Recovery via Proximal Gradient Descent and FISTA**

Recovers a corrupted image from masked observations by exploiting sparsity of natural images in the DCT domain. The reconstruction is posed as an L1-regularized least squares problem over DCT coefficients:

$$\min_X \frac{1}{2} \| M \circ (\text{iDCT}(X) - Y) \|_F^2 + \lambda \|X\|_1$$

Full mathematical derivations: proximal operator for L1 via case analysis and sub-differentials (soft thresholding), gradient of the data fidelity term via DCT adjoint property, and Lipschitz constant bounded to L ≤ 1 via spectral norm of the Hessian. Three implementations compared — CVXPY solver, PGD, and FISTA (momentum-accelerated PGD).

| Algorithm | Steps | Runtime | MSE | PSNR |
|---|---|---|---|---|
| CVXPY | — | 2.75s | 0.048 | 12.81 |
| PGD (downscaled) | 501 | 0.04s | 0.048 | 12.79 |
| PGD (original) | 658 | 1.12s | 0.009 | 20.23 |
| FISTA (original) | 329 | 0.59s | 0.009 | 20.23 |

FISTA halves convergence steps and wall time versus PGD at identical accuracy.

Key methods: DCT-domain sparsity, proximal gradient descent, soft thresholding, FISTA momentum, Lipschitz gradient analysis, PSNR/MSE evaluation.