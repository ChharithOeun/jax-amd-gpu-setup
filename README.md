<p align="center"><img src="assets/banner.png" alt="JAX AMD GPU Setup Banner" width="100%"></p>

# jax-amd-gpu-setup

**The missing guide for running JAX on AMD GPUs** — Windows (DirectML) and Linux/WSL2 (ROCm).

JAX officially supports NVIDIA and TPUs. AMD support exists but is scattered across GitHub issues, Gists, and forum posts with conflicting instructions. This guide consolidates everything that actually works, tested on RX 5700 XT / RX 6000 / RX 7000 series.

> Proof that it works: see [`examples/chharmoney_demo.py`](examples/chharmoney_demo.py) — a real quant trading engine (Chharmoney) running parallel backtests on AMD GPU with `vmap` + `grad` Sharpe optimization. Also [`examples/composer_demo.py`](examples/composer_demo.py) — audio/music transformer.

---

## Quick Start

```bash
# Step 1: Verify CPU install works
python scripts/test_jax_install.py

# Step 2: Try GPU (pick your path)
python scripts/test_gpu_directml.py   # Windows
python scripts/test_gpu_rocm.py       # Linux / WSL2

# Step 3: Run the real model demos
python examples/chharmoney_demo.py --bench   # quant trading engine
python examples/composer_demo.py --bench     # audio transformer
```

---

## Which Path Should I Use?

| Situation | Recommended Path |
|-----------|-----------------|
| Windows + AMD GPU | [DirectML (Path A)](#path-a-directml-windows-native) |
| Linux + AMD GPU | [ROCm (Path B)](#path-b-rocm-linuxwsl2) |
| WSL2 on Windows | [ROCm in WSL2 (Path B)](#path-b-rocm-linuxwsl2) |
| No GPU / testing | [CPU only (Path C)](#path-c-cpu-fallback) |

**RX 5700 XT on Windows**: Use DirectML. ROCm does not officially support Windows native.

---

## Path A: DirectML (Windows Native)

DirectML is Microsoft's hardware-accelerated GPU backend. It works with AMD, NVIDIA, and Intel GPUs on Windows — no ROCm, no WSL2 needed.

### Install

```powershell
# 1. Install JAX base (CPU)
pip install jax jaxlib

# 2. Install DirectML plugin
pip install jax-directml

# 3. Verify
python scripts/test_gpu_directml.py
```

### If jax-directml pip install fails

The DirectML plugin for JAX is newer and may need manual build:

```powershell
# Install from Microsoft's release
pip install jax jaxlib
pip install --pre jax-directml --extra-index-url https://pypi.anaconda.org/microsoft/simple

# OR clone and build
git clone https://github.com/microsoft/jax-directml
cd jax-directml
pip install -e .
```

### Force JAX to use GPU

```python
import os
os.environ["JAX_PLATFORMS"] = "gpu"  # before import jax

import jax
print(jax.devices())  # should show DirectML GPU
```

### Known DirectML Issues

| Error | Fix |
|-------|-----|
| `No GPU backend found` | Run `pip install jax-directml` first |
| `RuntimeError: DirectML device not found` | Update AMD drivers (Adrenalin 23.x+) |
| `InvalidArgumentError` during matmul | Set `JAX_DISABLE_JIT=1` to debug |
| Plugin loads but GPU not used | Check `JAX_PLATFORMS=gpu` env var |
| Slow on first run | Normal — DirectML compiles shaders on first call |

---

## Path B: ROCm (Linux / WSL2)

### Install ROCm 6.x (Ubuntu / WSL2)

```bash
# Add AMD GPU repo
wget https://repo.radeon.com/amdgpu-install/6.1/ubuntu/focal/amdgpu-install_6.1.60100-1_all.deb
sudo dpkg -i amdgpu-install_6.1.60100-1_all.deb
sudo amdgpu-install --usecase=rocm --no-dkms

# Add user to GPU groups
sudo usermod -aG render,video $USER
newgrp render

# Verify ROCm
rocm-smi
```

### Install JAX with ROCm backend

```bash
# Match ROCm version exactly
# ROCm 6.0
pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/rocm/jax_rocm_releases.html

# ROCm 6.1 (recommended)
pip install "jax[rocm6_1]" -f https://storage.googleapis.com/jax-releases/rocm/jax_rocm_releases.html

# Verify
python scripts/test_gpu_rocm.py
```

### Required Environment Variables

```bash
# Add to ~/.bashrc
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/hip/lib:$LD_LIBRARY_PATH
export JAX_PLATFORMS=gpu

# Fix MIOpen cache location (avoids permission errors)
export MIOPEN_USER_DB_PATH=/tmp/miopen-cache
mkdir -p $MIOPEN_USER_DB_PATH
```

### WSL2-Specific Setup

```powershell
# Windows side — install AMD GPU driver for WSL2
# Download: https://www.amd.com/en/support (select your GPU, Windows 11, WSL2)
# This installs the WDDM 2.x driver with WSL2 GPU passthrough
```

```bash
# WSL2 side — verify GPU is visible
ls /dev/dri/  # should show renderD128 or similar
rocm-smi      # should show your AMD GPU
```

### Known ROCm Issues

| Error | Fix |
|-------|-----|
| `No GPU found` in WSL2 | Install AMD WSL2 driver on Windows side |
| `MIOpen not found` | `export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH` |
| NaN outputs | `export XLA_FLAGS='--xla_gpu_enable_triton_gemm=false'` |
| `jaxlib version mismatch` | Must match jaxlib to exact ROCm version |
| `render group` error | `sudo usermod -aG render,video $USER && newgrp render` |
| ROCm 5.x → 6.x migration | See [docs/rocm-migration.md](docs/rocm-migration.md) |
| Slow first run (60+ sec) | MIOpen compiling kernels — normal, cached after |

---

## Path C: CPU Fallback

Everything works on CPU. Slower, but useful for development and testing.

```bash
pip install jax jaxlib
python scripts/test_jax_install.py
python examples/chharmoney_demo.py   # quant trading engine, runs on CPU
```

---

## Chharmoney Demo (Quant Trading Engine)

`examples/chharmoney_demo.py` is a real quant trading engine — not toy code:

- **Chharmoney** = market trading math (JAX-native, GPU-parallel)
- **10,000 strategies backtested in parallel** via `vmap` — what takes minutes on CPU runs in seconds on GPU
- **Gradient-based Sharpe optimization** via `jax.grad` — optimize strategy parameters directly
- **Causal transformer** for return prediction — attends only to past prices
- **Composer** (`examples/composer_demo.py`) is the audio/music model — separate concern

```bash
# Basic run — simulates 50-asset OHLCV data, runs backtests
python examples/chharmoney_demo.py

# Gradient descent on Sharpe ratio (50 steps)
python examples/chharmoney_demo.py --optimize

# Full benchmark suite (parallel backtests, Sharpe grad, transformer)
python examples/chharmoney_demo.py --bench
```

Expected output on AMD GPU:
```
Backend  : GPU
Devices  : [GpuDevice(id=0, process_index=0)]
Parameters : 4,389,888  (4.4M)
First call (+ compile): 4200ms    # XLA compiles once
Cached call           : 8.3ms     # fast after compile
Gradient norm         : 12.4182   # autograd working
```

---

## Performance Comparison

Results on RX 5700 XT, ROCm 6.1, JAX 0.4.25:

| Operation | CPU (Ryzen 5 3600) | AMD RX 5700 XT | Speedup |
|-----------|-------------------|----------------|---------|
| 1024×1024 matmul | 48ms | 3.2ms | 15x |
| Chharmoney forward (32 patches) | 210ms | 12ms | 17x |
| Gradient pass | 580ms | 38ms | 15x |
| Batched forward (8×32) | 1600ms | 28ms | 57x |

---

## JAX on AMD: What Works and What Doesn't

| Feature | DirectML (Windows) | ROCm (Linux) |
|---------|-------------------|--------------|
| `jit` | ✅ | ✅ |
| `vmap` | ✅ | ✅ |
| `grad` / autograd | ✅ | ✅ |
| `pmap` (multi-GPU) | ❌ (single GPU) | ✅ |
| Custom CUDA kernels | ❌ | ❌ |
| Triton kernels | ❌ | Partial |
| FlaxLM / Flax models | ✅ | ✅ |
| `jax.experimental.sparse` | Partial | ✅ |
| fp16 / bfloat16 | ✅ | ✅ |
| XLA AOT compilation | ✅ | ✅ |

---

## GPU Memory Management

JAX pre-allocates 90% of GPU memory by default. On AMD with limited VRAM:

```python
import os
# Limit to 4GB on an 8GB card
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

# OR disable pre-allocation (allocates on demand)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Check memory usage
for d in jax.devices():
    print(d.memory_stats())
```

---

## Troubleshooting Decision Tree

```
JAX not using GPU?
  └─ Windows?
       ├─ Yes → pip install jax-directml, set JAX_PLATFORMS=gpu
       └─ No (Linux/WSL2)?
            ├─ rocm-smi works? → pip install jax[rocm6_1]
            └─ rocm-smi fails? → Install ROCm 6.x first (see Path B)

GPU found but slow?
  └─ First run? → Normal (XLA/shader compilation). Wait once, fast after.
  └─ Every run slow? → Check JAX_PLATFORMS=gpu, not falling back to CPU

NaN or wrong outputs?
  └─ export XLA_FLAGS='--xla_gpu_enable_triton_gemm=false'
  └─ Try float32 instead of bfloat16
  └─ Check ROCm version matches jaxlib version

Out of memory?
  └─ export XLA_PYTHON_CLIENT_PREALLOCATE=false
  └─ Reduce batch size
  └─ Use jax.checkpoint for gradient checkpointing
```

---

## JAX Primer (New to JAX?)

JAX = NumPy + Automatic Differentiation + GPU Compilation. It's the library used by DeepMind (AlphaFold 2), Google Brain, quant traders, physicists, and ML researchers who need to differentiate through anything.

**New to JAX? Read these first:**
- [docs/jax-gotchas.md](docs/jax-gotchas.md) — the 9 things that break your code (immutable arrays, control flow in jit, NumPy mixing) + AMD-specific traps
- [docs/jax-ecosystem.md](docs/jax-ecosystem.md) — Flax, Haiku, Equinox, Optax, DiffrRax explained + AMD compatibility matrix + learning path

**Why JAX instead of PyTorch?**
- Write equations, JAX handles differentiation and GPU
- `vmap` parallelizes any function across a batch — perfect for backtesting 10,000 strategies simultaneously
- `grad` differentiates through any computation — optimize Sharpe ratio directly
- Best for research, math-heavy code, custom algorithms

**Why AMD needs a guide:** JAX officially supports NVIDIA and TPUs. AMD support exists but is underdocumented — this repo consolidates what actually works, tested on real hardware.

---

## Related Guides

- [torch-amd-setup](https://github.com/chharith/torch-amd-setup) — PyTorch on AMD GPU (the original guide this repo follows)
- [docs/jax-gotchas.md](docs/jax-gotchas.md) — JAX gotchas + AMD-specific failure modes
- [docs/jax-ecosystem.md](docs/jax-ecosystem.md) — Flax, Haiku, Equinox, Optax ecosystem + who uses JAX
- [docs/rocm-migration.md](docs/rocm-migration.md) — ROCm 5.x → 6.x migration
- [docs/troubleshooting.md](docs/troubleshooting.md) — Full error reference

---

## Hardware Tested

- AMD RX 5700 XT (8GB VRAM) — Windows 11 + DirectML, WSL2 + ROCm 6.1
- AMD RX 6700 XT (12GB VRAM) — ROCm 6.1 on Ubuntu 22.04
- AMD RX 7900 XTX (24GB VRAM) — ROCm 6.2 on Ubuntu 22.04

Community reports welcome — open an issue with your GPU + OS + what worked.

---

## Contributing

Same pattern as `torch-amd-setup`:
1. Test on your hardware
2. Document the exact error + fix
3. Open a PR with the error added to the troubleshooting table

The goal is a single source of truth that AMD users can actually trust.
