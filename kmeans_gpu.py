"""
GPU-Accelerated K-Means Clustering
====================================
Author: Areesha Imtiaz
Course: CS 4220-01 — Parallel Computing
Instructor: Dr. Hao Ji

Compares CPU (NumPy) vs GPU (CuPy) K-Means clustering across
dataset sizes of 10K, 50K, and 100K points.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import make_blobs

# ── Try importing CuPy; fall back gracefully if not installed ──────────────
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print(f"CuPy detected. GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not found. GPU benchmarks will be skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
K = 8                          # number of clusters
MAX_ITER = 100                 # max K-Means iterations
TOL = 1e-4                     # centroid convergence tolerance
DATASET_SIZES = [10_000, 50_000, 100_000]
N_FEATURES = 2                 # 2D points for easy visualization
RANDOM_STATE = 42
COLORS = plt.cm.tab10.colors


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_data(n_samples: int, n_features: int = 2, k: int = 8) -> np.ndarray:
    """Generate synthetic clustered data using sklearn make_blobs."""
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=k,
        cluster_std=1.5,
        random_state=RANDOM_STATE,
    )
    return X.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CPU K-MEANS  (pure NumPy)
# ─────────────────────────────────────────────────────────────────────────────
def kmeans_cpu(X: np.ndarray, k: int, max_iter: int = MAX_ITER, tol: float = TOL):
    """
    K-Means on CPU using NumPy.

    Distance computation:  ||x - c||² = ||x||² - 2·x·cᵀ + ||c||²
    This broadcast trick avoids an explicit loop over all (point, centroid) pairs.

    Returns
    -------
    labels      : (n,)   cluster assignment for each point
    centroids   : (k, d) final centroid positions
    elapsed     : float  wall-clock seconds
    """
    n, d = X.shape
    rng = np.random.default_rng(RANDOM_STATE)

    # ── Initialise centroids by random sampling (K-Means++ would be better,
    #    but plain random keeps the focus on the GPU comparison) ──────────────
    idx = rng.choice(n, size=k, replace=False)
    centroids = X[idx].copy()

    labels = np.zeros(n, dtype=np.int32)

    t0 = time.perf_counter()
    for iteration in range(max_iter):
        # ── Assignment step ──────────────────────────────────────────────────
        # Squared distances: (n, k)
        #   ||x_i - c_j||² = ||x_i||² - 2 x_i·c_j + ||c_j||²
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)          # (n, 1)
        C_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T # (1, k)
        dot  = X @ centroids.T                                  # (n, k)
        dist2 = X_sq - 2 * dot + C_sq                          # (n, k)

        new_labels = np.argmin(dist2, axis=1).astype(np.int32)

        # ── Update step ───────────────────────────────────────────────────────
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = new_labels == j
            if mask.any():
                new_centroids[j] = X[mask].mean(axis=0)
            else:
                # Empty cluster: re-initialise to a random point
                new_centroids[j] = X[rng.integers(n)]

        # ── Convergence check ─────────────────────────────────────────────────
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        labels    = new_labels
        if shift < tol:
            print(f"  [CPU] converged at iteration {iteration + 1}")
            break

    elapsed = time.perf_counter() - t0
    return labels, centroids, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# GPU K-MEANS  (CuPy)
# ─────────────────────────────────────────────────────────────────────────────
def kmeans_gpu(X_cpu: np.ndarray, k: int, max_iter: int = MAX_ITER, tol: float = TOL):
    """
    K-Means on GPU using CuPy.

    The algorithm is identical to the CPU version; the only difference is
    that all arrays live in GPU memory (cp.ndarray) and CuPy dispatches
    the elementwise / BLAS operations to CUDA kernels automatically.

    A cp.cuda.Stream.null.synchronize() call is placed around the timed
    region so that async GPU work is fully included in the measurement.

    Returns
    -------
    labels      : (n,)   NumPy array of cluster assignments
    centroids   : (k, d) NumPy array of final centroid positions
    elapsed     : float  wall-clock seconds (GPU execution only)
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy is not available.")

    rng = np.random.default_rng(RANDOM_STATE)
    n, d = X_cpu.shape

    idx = rng.choice(n, size=k, replace=False)
    centroids_cpu = X_cpu[idx].copy()

    # ── Transfer data to GPU ──────────────────────────────────────────────────
    X        = cp.asarray(X_cpu)
    centroids = cp.asarray(centroids_cpu)

    labels = cp.zeros(n, dtype=cp.int32)

    # Warm-up: one small matrix multiply so driver initialisation time is
    # not charged to our benchmark
    _ = cp.dot(cp.ones((2, 2), dtype=cp.float32), cp.ones((2, 2), dtype=cp.float32))
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for iteration in range(max_iter):
        # ── Assignment step (same broadcast trick, on GPU) ────────────────────
        X_sq  = cp.sum(X ** 2, axis=1, keepdims=True)           # (n, 1)
        C_sq  = cp.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, k)
        dot   = X @ centroids.T                                   # (n, k)
        dist2 = X_sq - 2 * dot + C_sq                            # (n, k)

        new_labels = cp.argmin(dist2, axis=1).astype(cp.int32)

        # ── Update step ───────────────────────────────────────────────────────
        new_centroids = cp.zeros_like(centroids)
        for j in range(k):
            mask = new_labels == j
            if int(mask.sum()) > 0:
                new_centroids[j] = X[mask].mean(axis=0)
            else:
                new_centroids[j] = X[int(rng.integers(n))]

        # ── Convergence check ─────────────────────────────────────────────────
        shift = float(cp.linalg.norm(new_centroids - centroids))
        centroids = new_centroids
        labels    = new_labels
        if shift < tol:
            print(f"  [GPU] converged at iteration {iteration + 1}")
            break

    cp.cuda.Stream.null.synchronize()   # ensure all CUDA work is done
    elapsed = time.perf_counter() - t0

    return cp.asnumpy(labels), cp.asnumpy(centroids), elapsed


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARKING
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmarks():
    """Run CPU and GPU K-Means for each dataset size; return timing tables."""
    results = []          # list of dicts: {size, cpu_time, gpu_time, speedup}
    last_run = {}         # keep the 10K run for visualization

    print("\n" + "=" * 60)
    print(f"  K-Means Benchmark  |  K={K}  |  max_iter={MAX_ITER}")
    print("=" * 60)

    for n in DATASET_SIZES:
        print(f"\n── Dataset size: {n:,} points ──")
        X = generate_data(n)

        # CPU
        print("  Running CPU …")
        cpu_labels, cpu_centroids, cpu_time = kmeans_cpu(X, K)
        print(f"  CPU time: {cpu_time:.4f}s")

        # GPU
        gpu_time = None
        gpu_labels = None
        gpu_centroids = None
        if GPU_AVAILABLE:
            print("  Running GPU …")
            gpu_labels, gpu_centroids, gpu_time = kmeans_gpu(X, K)
            print(f"  GPU time: {gpu_time:.4f}s")
            speedup = cpu_time / gpu_time
            print(f"  Speedup:  {speedup:.2f}×")
        else:
            speedup = None

        results.append({
            "size":      n,
            "cpu_time":  cpu_time,
            "gpu_time":  gpu_time,
            "speedup":   speedup,
        })

        if n == DATASET_SIZES[0]:
            last_run = {
                "X":              X,
                "cpu_labels":     cpu_labels,
                "cpu_centroids":  cpu_centroids,
                "gpu_labels":     gpu_labels,
                "gpu_centroids":  gpu_centroids,
            }

    return results, last_run


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_clusters(X, labels, centroids, title, ax, k=K):
    """Scatter-plot the clustered data on a given Axes object."""
    for j in range(k):
        mask = labels == j
        ax.scatter(
            X[mask, 0], X[mask, 1],
            s=4, alpha=0.5, color=COLORS[j % len(COLORS)], rasterized=True,
        )
    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        s=200, marker="*", color="black", zorder=5, label="Centroids",
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(fontsize=8)


def plot_timing(results, ax_time, ax_speedup):
    """Bar chart of runtimes and line chart of speedup."""
    sizes  = [r["size"] for r in results]
    labels = [f"{s//1000}K" for s in sizes]
    cpu_t  = [r["cpu_time"] for r in results]

    x = np.arange(len(sizes))
    width = 0.35

    bars_cpu = ax_time.bar(x - width / 2, cpu_t, width, label="CPU", color="#4878CF")

    if GPU_AVAILABLE:
        gpu_t = [r["gpu_time"] for r in results]
        bars_gpu = ax_time.bar(x + width / 2, gpu_t, width, label="GPU", color="#6ACC65")
        ax_time.legend(fontsize=9)

        speedups = [r["speedup"] for r in results]
        ax_speedup.plot(labels, speedups, "o-", color="#D65F5F", linewidth=2, markersize=8)
        ax_speedup.set_ylabel("Speedup (×)", fontsize=10)
        ax_speedup.set_title("GPU Speedup over CPU", fontsize=12, fontweight="bold")
        ax_speedup.axhline(1, color="gray", linestyle="--", linewidth=0.8)
        ax_speedup.set_ylim(0, max(speedups) * 1.3)
        for i, (lbl, sp) in enumerate(zip(labels, speedups)):
            ax_speedup.annotate(f"{sp:.1f}×", (lbl, sp),
                                textcoords="offset points", xytext=(0, 8),
                                ha="center", fontsize=9)
    else:
        ax_speedup.text(0.5, 0.5, "GPU not available\n(no speedup data)",
                        ha="center", va="center", transform=ax_speedup.transAxes,
                        fontsize=11, color="gray")

    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels)
    ax_time.set_ylabel("Time (seconds)", fontsize=10)
    ax_time.set_title("Runtime Comparison: CPU vs GPU", fontsize=12, fontweight="bold")
    ax_time.set_xlabel("Dataset Size")
    if not GPU_AVAILABLE:
        ax_time.legend(fontsize=9)


def generate_figures(results, last_run):
    """Produce and save all project figures."""
    X             = last_run["X"]
    cpu_labels    = last_run["cpu_labels"]
    cpu_centroids = last_run["cpu_centroids"]

    # ── Figure 1: Cluster visualisation ──────────────────────────────────────
    if GPU_AVAILABLE and last_run["gpu_labels"] is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_clusters(X, cpu_labels, cpu_centroids, "CPU K-Means (10K points)", axes[0])
        plot_clusters(X, last_run["gpu_labels"], last_run["gpu_centroids"],
                      "GPU K-Means (10K points)", axes[1])
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]
        plot_clusters(X, cpu_labels, cpu_centroids, "CPU K-Means (10K points)", axes[0])

    fig.suptitle("K-Means Clustering Results", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig("cluster_visualization.png", dpi=150, bbox_inches="tight")
    print("\nSaved: cluster_visualization.png")
    plt.close(fig)

    # ── Figure 2: Timing & speedup ────────────────────────────────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_timing(results, ax1, ax2)
    fig2.suptitle("Performance Analysis: GPU-Accelerated K-Means", fontsize=13,
                  fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("performance_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: performance_comparison.png")
    plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results, last_run = run_benchmarks()

    print("\n" + "=" * 60)
    print(f"{'Size':>10}  {'CPU (s)':>10}  {'GPU (s)':>10}  {'Speedup':>10}")
    print("-" * 44)
    for r in results:
        gpu_str     = f"{r['gpu_time']:.4f}" if r["gpu_time"] else "N/A"
        speedup_str = f"{r['speedup']:.2f}×"  if r["speedup"] else "N/A"
        print(f"{r['size']:>10,}  {r['cpu_time']:>10.4f}  {gpu_str:>10}  {speedup_str:>10}")
    print("=" * 60)

    generate_figures(results, last_run)
    print("\nDone! Check cluster_visualization.png and performance_comparison.png")