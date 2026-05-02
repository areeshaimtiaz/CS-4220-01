# GPU-Accelerated K-Means Clustering

**Course:** CS 4220-01 — Parallel Computing  
**Author:** Areesha Imtiaz  
**Instructor:** Dr. Hao Ji

---

## Overview

This project implements K-Means clustering in two ways:

| Implementation | Library | Device |
|---|---|---|
| CPU baseline | NumPy | CPU |
| GPU-accelerated | CuPy | NVIDIA CUDA GPU |

The benchmark measures execution time across three dataset sizes (10K, 50K, 100K 2D points, K=8 clusters) and produces cluster visualizations and a speedup chart.

---

## Project Structure

```
kmeans_project/
├── kmeans_gpu.py          ← Main file: CPU + GPU K-Means + benchmarking + plots
└── README.md
```

### Key sections in `kmeans_gpu.py`

| Function | Purpose |
|---|---|
| `generate_data()` | Synthetic blob generation via scikit-learn |
| `kmeans_cpu()` | Pure NumPy K-Means (vectorized broadcast distance trick) |
| `kmeans_gpu()` | CuPy K-Means — same algorithm, all arrays on GPU VRAM |
| `run_benchmarks()` | Loops over dataset sizes, records wall-clock times |
| `generate_figures()` | Saves cluster scatter plots + timing/speedup bar charts |

---

## Requirements

### Python packages

```bash
pip install numpy scikit-learn matplotlib cupy-cuda12x
```

> **CuPy version:** Install the wheel that matches your CUDA toolkit.  
> - CUDA 11.x → `pip install cupy-cuda11x`  
> - CUDA 12.x → `pip install cupy-cuda12x`  
>
> Check your version: `nvcc --version`

### Hardware / Software

| Requirement | Details |
|---|---|
| Python | 3.9 or later |
| NVIDIA GPU | Any CUDA-capable GPU (tested on RTX-class) |
| CUDA Toolkit | 11.x or 12.x |
| OS | Linux or Windows with CUDA drivers |

---

## How to Run

```bash
# Clone the repo
git clone <your-repo-url>
cd kmeans_project

# Install dependencies
pip install numpy scikit-learn matplotlib cupy-cuda12x

# Run the benchmark
python kmeans_gpu.py
```

### Expected output

```
CuPy detected. GPU: NVIDIA GeForce RTX XXXX
============================================================
  K-Means Benchmark  |  K=8  |  max_iter=100
============================================================
── Dataset size: 10,000 points ──
  Running CPU …
  [CPU] converged at iteration XX
  CPU time: X.XXXXs
  Running GPU …
  [GPU] converged at iteration XX
  GPU time: X.XXXXs
  Speedup:  X.XX×
...
```

Two PNG files are saved in the working directory:
- `cluster_visualization.png` — scatter plot of CPU vs GPU clustering
- `performance_comparison.png` — runtime bar chart and speedup line chart

---

## Algorithm

Both implementations use the same vectorized distance formula to avoid slow Python loops:

```
||x_i - c_j||² = ||x_i||² - 2·(x_i · c_j) + ||c_j||²
```

This reduces the assignment step to a matrix multiply (`X @ C.T`) plus broadcasts — a pattern that maps naturally to GPU parallelism via CuPy's CUBLAS backend.

### GPU pipeline

```
NumPy array (RAM)
      │  cp.asarray()
      ▼
CuPy array (GPU VRAM)
      │
      ├─ Assignment step: cp.sum, @, cp.argmin   [CUDA kernels]
      ├─ Update step:     cp.zeros, boolean mask  [CUDA kernels]
      └─ Convergence:     cp.linalg.norm          [CUDA kernel]
      │  cp.asnumpy()
      ▼
NumPy array (RAM) — labels & centroids returned to host
```

---

## Configuration

Edit the constants at the top of `kmeans_gpu.py` to change behavior:

```python
K            = 8          # number of clusters
MAX_ITER     = 100        # max iterations
TOL          = 1e-4       # convergence tolerance
DATASET_SIZES = [10_000, 50_000, 100_000]
N_FEATURES   = 2          # dimensionality of points
```
