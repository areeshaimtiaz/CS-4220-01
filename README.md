# GPU-Accelerated K-Means Clustering

**Course:** CS 4220-01 — GPU Computing  
**Author:** Areesha Imtiaz  
**Instructor:** Dr. Hao Ji  

---

## Overview

This project implements K-Means clustering in two ways and benchmarks them on the NCSA Delta HPC cluster (NVIDIA A40 GPU):

| Implementation | Library | Device |
|---|---|---|
| CPU baseline | NumPy | CPU (login node) |
| GPU-accelerated | CuPy | NVIDIA A40 GPU (gpuA40x4 partition) |

Benchmarks are run across three dataset sizes (10K, 50K, 100K 2D points, K=8 clusters).

---

## Repository Structure

```
CS-4220-01/
├── kmeans_gpu.py                  <- Main file: CPU + GPU K-Means + benchmarking + plots
├── README.md                      <- This file
├── cluster_visualization.png      <- Output: scatter plot of clustering results
├── performance_comparison.png     <- Output: runtime and speedup bar charts
└── Final Project Proposal.docx   <- Original project proposal
```

### Key functions in `kmeans_gpu.py`

| Function | Purpose |
|---|---|
| `generate_data()` | Synthetic blob generation via scikit-learn |
| `kmeans_cpu()` | Pure NumPy K-Means (vectorized broadcast distance trick) |
| `kmeans_gpu()` | CuPy K-Means — same algorithm, all arrays on GPU VRAM |
| `run_benchmarks()` | Loops over dataset sizes, records wall-clock times |
| `generate_figures()` | Saves cluster scatter plots + timing/speedup bar charts |

---

## Results (NVIDIA A40 — NCSA Delta Cluster)

| Dataset Size | CPU Time (s) | GPU Time (s) | Speedup |
|---|---|---|---|
| 10,000  | 0.0515 | 0.2393 | 0.22x |
| 50,000  | 0.2069 | 0.1205 | 1.72x |
| 100,000 | 1.0177 | 0.2686 | 3.79x |

GPU is slower at 10K due to CUDA kernel launch and initialization overhead. Speedup grows as dataset size increases, reaching **3.79x** at 100K points.

---

## Requirements

### Python packages

```bash
pip install --user numpy scikit-learn matplotlib cupy-cuda12x
```

> **CuPy version:** Install the wheel matching your CUDA toolkit.
> - CUDA 11.x: `pip install --user cupy-cuda11x`
> - CUDA 12.x: `pip install --user cupy-cuda12x`
>
> Check your CUDA version: `nvcc --version`

### Hardware / Software

| Requirement | Details |
|---|---|
| Python | 3.9 (system Python on Delta) |
| NVIDIA GPU | A40 (tested) or any CUDA-capable GPU |
| CUDA Toolkit | 12.x (on NCSA Delta) |
| Cluster | NCSA Delta HPC |

---

## How to Run

### On NCSA Delta HPC (recommended)

**Step 1 — SSH into Delta via VS Code**

Install the *Remote - SSH* extension in VS Code, then connect to:
```
ssh aimtiaz@login.delta.ncsa.illinois.edu
```

**Step 2 — Request a GPU compute node**

```bash
srun --account=bchn-delta-gpu --partition=gpuA40x4-interactive --gpus=1 --time=01:00:00 --pty bash
```

Wait for the job to be allocated (your prompt changes to `[aimtiaz@gpuXXX ~]$`).

**Step 3 — Navigate to project and install dependencies (first time only)**

```bash
cd ~/Final\ Project/CS-4220-01
pip install --user numpy scikit-learn matplotlib cupy-cuda12x
```

**Step 4 — Run the benchmark**

```bash
python kmeans_gpu.py
```

### On a local machine with an NVIDIA GPU

```bash
git clone https://github.com/areeshaimtiaz/CS-4220-01.git
cd CS-4220-01
pip install numpy scikit-learn matplotlib cupy-cuda12x
python kmeans_gpu.py
```

### Expected output

```
CuPy detected. GPU: NVIDIA A40
============================================================
  K-Means Benchmark  |  K=8  |  max_iter=100
============================================================
-- Dataset size: 10,000 points --
  Running CPU ...
  [CPU] converged at iteration 41
  CPU time: 0.0515s
  Running GPU ...
  [GPU] converged at iteration 41
  GPU time: 0.2393s
  Speedup:  0.22x
...
Saved: cluster_visualization.png
Saved: performance_comparison.png
```

Two PNG files are saved in the working directory:
- `cluster_visualization.png` — scatter plot of clustering results
- `performance_comparison.png` — runtime bar chart and speedup chart

---

## Algorithm

Both implementations use the vectorized distance formula:

```
||x_i - c_j||^2 = ||x_i||^2 - 2*(x_i . c_j) + ||c_j||^2
```

This converts the assignment step into a single matrix multiply (X @ C.T) plus broadcasts — a pattern that maps directly to GPU parallelism via CuPy's cuBLAS backend.

### GPU pipeline

```
NumPy array (RAM)
      |  cp.asarray()
      v
CuPy array (GPU VRAM -- NVIDIA A40)
      |
      +-- Assignment: cp.sum, X @ C.T, cp.argmin    [CUDA kernels / cuBLAS]
      +-- Update:     boolean mask, cp.zeros, mean   [CUDA kernels]
      +-- Convergence: cp.linalg.norm                [CUDA kernel]
      |
      |  cp.cuda.Stream.null.synchronize()
      |  cp.asnumpy()
      v
NumPy array (RAM) -- labels and centroids returned to host
```

---

## Configuration

Edit the constants at the top of `kmeans_gpu.py`:

```python
K             = 8           # number of clusters
MAX_ITER      = 100         # max iterations
TOL           = 1e-4        # convergence tolerance
DATASET_SIZES = [10_000, 50_000, 100_000]
N_FEATURES    = 2           # 2D points
```
