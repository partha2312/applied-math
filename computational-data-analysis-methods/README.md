# Computational Methods for Data Analysis

Course work from **AMATH 582: Computational Methods for Data Analysis** at the University of Washington (Applied and Computational Mathematics, Winter 2026). Each homework is a self-contained study combining mathematical derivation, algorithmic implementation, and empirical analysis. Reports are written in research paper format with formal theoretical background sections and experimental results.

## Homeworks

### [`hw1-fourier-submarine-tracking/`](./hw1-fourier-submarine-tracking/)
**Finding Submarines: 3D FFT and Gaussian Spectral Filtering**

Localizes a moving submarine from 3D volumetric acoustic pressure measurements (64³ grid, 49 timesteps) corrupted by broadband stochastic noise. The frequency signature of the submarine is extracted by spectral averaging across all time snapshots — coherent signal components constructively accumulate while incoherent noise converges to a stable floor. A 3D Gaussian band-pass filter centered at the identified signature frequency then denoises each snapshot. The filtered signal is used to reconstruct the submarine's spatial trajectory over time.

Key methods: 3D FFT (Cooley-Tukey), spectral averaging for SNR enhancement, Gaussian band-pass filtering in the frequency domain, trajectory reconstruction via argmax tracking.

---

### [`hw2-wavelet-edge-detection/`](./hw2-wavelet-edge-detection/)
**Wavelet Level Selection: Scale-Noise Tradeoff Analysis**

Edge detection via Discrete Wavelet Transform with adaptive decomposition level selection. Fine-scale wavelet coefficients preserve spatial resolution but amplify noise; coarse scales suppress noise at the cost of localization accuracy. The algorithm automatically selects decomposition depth by estimating background noise standard deviation from the finest-scale diagonal sub-band using Median Absolute Deviation (MAD) — following the Donoho-Johnstone robust shrinkage estimator. Edge magnitude is computed as the Euclidean norm of horizontal, vertical, and diagonal detail coefficients at the selected scale, followed by relative thresholding and nearest-neighbor upsampling to reconstruct full-resolution edge maps.

Tested across four wavelet bases: Haar, db2, sym3, bior1.5. Results demonstrate that the adaptive approach preserves structural edges in high-noise environments where fixed-scale methods fail.

Key methods: 2D DWT, multi-resolution analysis (MRA), MAD-based noise estimation, adaptive level selection, Donoho-Johnstone shrinkage.

---

### [`hw3-svd-background-subtraction/`](./hw3-svd-background-subtraction/)
**Background Subtraction via Low-Rank Matrix Approximation**

Separates static background from moving foreground in video by exploiting the temporal structure of the data. A video of n_f frames at n_x × n_y resolution is reshaped into a (n_f × n_x·n_y) matrix. The static background has high temporal correlation and concentrates in the top singular vectors (low rank), while moving foreground objects are transient and sparse. The rank-1 background model (first singular component) is subtracted from each frame to isolate foreground.

Compares full deterministic SVD (O(min(mn², m²n))) against a randomized SVD following Martinsson and Tropp (2020). The randomized algorithm constructs an orthonormal range approximation via a Gaussian random sketch and QR factorization, then projects the data matrix onto this subspace for a cheaper deterministic decomposition. Top k=20 singular vectors are retained to prevent spectral leakage of foreground dynamics into the background model.

Key methods: PCA via SVD, randomized range finder, randomized SVD, rank-1 background modeling, foreground extraction.

---

### [`hw4-kernel-ridge-wine-classification/`](./hw4-kernel-ridge-wine-classification/)
**Classifying Red Wine Quality via Kernel Ridge Regression**

Multi-class classification of red wine quality scores on the UCI Wine Quality dataset using kernel methods. Ridge regularization controls model complexity by penalizing coefficient magnitude, preventing overfitting. The kernel trick implicitly maps inputs to a high-dimensional feature space, enabling the linear ridge classifier to capture non-linear decision boundaries without explicitly computing the feature map. Compared several kernel choices (linear, polynomial, RBF) within the Ridge Classification framework.

Derives the full MLE → least squares → ridge regularization chain and extends to multi-class via one-hot encoding. Includes hyperparameter tuning for the regularization parameter λ and kernel parameters, with validation on a held-out test set.

Key methods: MLE derivation, least squares regression, Ridge regularization, kernel trick, multi-class one-hot classification, bias-variance tradeoff analysis.

---

### [`hw5-spectral-clustering-congress/`](./hw5-spectral-clustering-congress/)
**Topological Data Analysis: Spectral Clustering on Congressional Voting Records**

Applies manifold learning to classify U.S. Congressional members by political affiliation using 16-dimensional voting record data (UCI House Votes 84 dataset). Demonstrates that k-Means, which enforces Euclidean distance boundaries, artificially severs continuous communities on non-linear manifolds — verified via t-SNE visualization. Spectral clustering overcomes this by constructing Laplacian embeddings in a lower-dimensional eigenspace that respects the underlying topology.

Implements three Graph Laplacian variants (unnormalized, normalized, random walk) and constructs diffusion maps from their eigendecomposition. The resulting embeddings are reused in a semi-supervised regression (SSR) framework to predict political affiliation from a small labeled subset. Results demonstrate strong sensitivity to embedding dimensionality and the fraction of labeled data used for learning.

Key methods: Graph Laplacian construction, diffusion maps, spectral embeddings, semi-supervised regression, k-Means on Laplacian features, t-SNE visualization.