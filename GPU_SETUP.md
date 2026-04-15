# GPU Setup Guide — Numerical Ax = b Lab

## Overview

The app supports both CPU and GPU execution.  The GPU path uses
[CuPy](https://cupy.dev/) — a drop-in NumPy/SciPy replacement that runs on
NVIDIA CUDA GPUs.  Switching between CPU and GPU is done via a radio button at
the top of the sidebar.

---

## 1. Check your CUDA version

```bash
nvidia-smi          # top-right corner shows "CUDA Version: XX.X"
nvcc --version      # shows "release X.Y"
```

---

## 2. Install CuPy

Install the wheel that matches your CUDA version:

| CUDA version | Install command               |
|---|---|
| 11.2 – 11.x  | `pip install cupy-cuda11x`    |
| 12.0 – 12.x  | `pip install cupy-cuda12x`    |
| Any (source) | `pip install cupy`            |

On university HPC clusters CuPy is often pre-installed as a module:

```bash
module avail | grep -i cupy    # list available modules
module load cupy               # load the module
```

### Verify the installation

```bash
python -c "import cupy as cp; cp.array([1.0]); print('CuPy OK,', cp.cuda.runtime.getDeviceCount(), 'device(s)')"
```

---

## 3. Install cupyx (bundled with CuPy)

`cupyx` is the SciPy-compatible layer of CuPy and is installed automatically
with `cupy-cuda11x` / `cupy-cuda12x`.  No separate install is needed.

---

## 4. File layout

The GPU support adds one new file to the `core/` directory:

```
numerical_lab/
├── app.py
├── requirements.txt
├── GPU_SETUP.md          ← this file
├── core/
│   ├── device.py         ← NEW: CPU/GPU abstraction layer
│   ├── problem_creation.py
│   ├── solvers.py
│   └── analysis.py
└── ui/
    ├── problem_ui.py
    ├── solver_ui.py
    └── analysis_ui.py
```

---

## 5. How the GPU path works

| Step | What happens |
|---|---|
| Matrix generation | Always on CPU (NumPy) — algebraically complex, negligible transfer cost |
| GPU transfer | `cupy.asarray(A)` after generation — one H→D copy per matrix |
| Solver | All operations (SVD, QR, LU, Cholesky, GMRES, CG) run on GPU via CuPy / cuPyx |
| High-precision residual | Pulled back to CPU via `to_numpy()` then run through mpmath/float128; result transferred back |
| Display | All arrays pulled to CPU via `to_numpy()` before Matplotlib/Seaborn rendering |

---

## 6. Solver GPU support table

| Solver | GPU library | Notes |
|---|---|---|
| SVD (Reduced / Full) | `cupy.linalg.svd` | ✅ Full support |
| QR Householder (Reduced / Full) | `cupy.linalg.qr` | ✅ Full support |
| QR Classical Gram-Schmidt | hand-rolled CuPy loop | ✅ Full support |
| QR Modified Gram-Schmidt | hand-rolled CuPy loop | ✅ Full support |
| LU (partial pivoting) | `cupyx.scipy.linalg.lu_factor/lu_solve` | ✅ Full support |
| LU (Sparse / SuperLU) | `cupyx.scipy.sparse.linalg.splu` | ✅ Falls back to `cupy.linalg.solve` if splu unavailable |
| Cholesky | `cupyx.scipy.linalg.cho_factor/cho_solve` | ✅ Full support |
| GMRES | `cupyx.scipy.sparse.linalg.gmres` | ✅ Full support |
| CG | `cupyx.scipy.sparse.linalg.cg` | ✅ Full support |

---

## 7. Troubleshooting

**`DeviceError: GPU requested but CuPy is not installed`**  
→ Install CuPy for your CUDA version (see step 2).

**`cupy.cuda.runtime.CUDARuntimeError: cudaErrorNoDevice`**  
→ No GPU visible.  Check `nvidia-smi`.  On HPC you may need to request a GPU node.

**`ImportError: libcuda.so.1: cannot open shared object file`**  
→ CUDA driver not loaded.  On HPC: `module load cuda`.

**`cupyx.scipy.linalg.lu_factor` not available on your CuPy version**  
→ Upgrade CuPy: `pip install --upgrade cupy-cuda12x`

**Sparse LU falls back to dense solve on GPU**  
→ This is normal for older CuPy versions.  The result is numerically identical;
only performance differs.

---

## 8. Performance notes

- GPU speedup is most noticeable for **large dense matrices** (m ≥ 500) and
  **iterative solvers** (GMRES, CG) where the matrix-vector product dominates.
- For **small matrices** (m < 100) CPU is often faster due to GPU launch overhead.
- The **high-precision residual** always runs on CPU (mpmath/float128), so the
  analysis metrics are equally accurate on both devices.
