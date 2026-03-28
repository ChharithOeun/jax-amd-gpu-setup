# Community Test Results

Each file in this directory is a real test run from a real AMD GPU.
This is the AMD JAX compatibility matrix that doesn't exist anywhere else.

## How to Submit Your Results

```bash
# Run the full suite (takes ~5-10 minutes)
python scripts/research_failure_suite.py

# Quick run (skips memory/performance tests)
python scripts/research_failure_suite.py --quick

# Auto-generate a GitHub issue draft
python scripts/research_failure_suite.py --submit
```

Then [open an issue](https://github.com/ChharithOeun/jax-amd-gpu-setup/issues/new?template=test-results.md) and paste your result JSON — or upload the file directly.

---

## Results by GPU

| GPU | Architecture | VRAM | OS | ROCm | JAX | Pass Rate | Key Failures | Contributor |
|-----|-------------|------|----|------|-----|-----------|-------------|-------------|
| RX 5700 XT | RDNA1 | 8GB | WSL2/Win11 | 6.1 | 0.4.25 | **69%** (29/42) | bfloat16 NaN (workaround ✅), Triton GEMM | [@ChharithOeun](https://github.com/ChharithOeun) |

> Add your GPU — run the suite and open an issue with your JSON result.

---

## RX 5700 XT Detailed Results (RDNA1, WSL2, ROCm 6.1)

**Date:** 2026-03-27 | **File:** `AMD_Radeon_RX_5700_XT_rocm6_1_20260327.json`

### What Works ✅
- `jit`, `vmap`, `grad`, `value_and_grad`, `lax.scan`, `lax.while_loop`, `lax.cond`
- Matrix ops: `dot`, `einsum`, `svd`, `eigvalsh` (Schrödinger solver)
- **Physics:** Schrödinger equation, wave PDE, variational Monte Carlo, Lennard-Jones MD
- **Quant:** Markowitz optimization, Monte Carlo option pricing, VaR/CVaR, Sharpe gradient
- **10,000 parallel backtests via `vmap` in 2.1s (18x speedup vs CPU)**
- **100,000 Monte Carlo option paths in 18.7ms**

### What Fails ❌

| Test | Error | Workaround |
|------|-------|-----------|
| bfloat16 matmul | NaN output — RDNA1 hardware bug | `XLA_FLAGS='--xla_gpu_enable_triton_gemm=false'` ✅ |
| Triton GEMM (default on) | NaN in matmul | Same fix above ✅ |
| Custom CUDA kernels | Not supported on AMD | Use XLA custom ops or Triton |

### What's Slow ⚠️

| Item | Time | Notes |
|------|------|-------|
| First JIT compile | ~42 seconds | Cache: `JAX_COMPILATION_CACHE_DIR=/tmp/jax-cache` |
| Neural ODE (jax.experimental.ode) | 3x slower than CPU | Use `diffrax.Dopri5` instead |
| Overall GPU efficiency | ~35-45% TFLOPS | Known RDNA1 + ROCm limitation |

### Working Environment
```bash
export XLA_FLAGS='--xla_gpu_enable_triton_gemm=false'
export MIOPEN_USER_DB_PATH=/tmp/miopen-cache
export JAX_COMPILATION_CACHE_DIR=/tmp/jax-cache
export HIP_VISIBLE_DEVICES=0
export JAX_PLATFORMS=gpu
```

---

## GPU Architecture Reference

| Architecture | GPUs | Status |
|-------------|------|--------|
| RDNA1 (GFX1010) | RX 5500/5600/5700 | Works with workarounds — see above |
| RDNA2 (GFX1030) | RX 6600/6700/6800/6900 | Better — most bfloat16 issues resolved |
| RDNA3 (GFX1100) | RX 7600/7700/7800/7900 | Best AMD support |
| CDNA (GFX908/GFX90A) | MI100/MI250 | Server GPUs, best ROCm support |

Submit your result to fill in the matrix.
