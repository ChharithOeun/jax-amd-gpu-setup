# Troubleshooting: JAX on AMD GPU

Full error reference. Every entry is a real error someone hit.

---

## Installation Errors

### `ERROR: Could not find a version that satisfies jax-directml`
```
Solution: Try alternate index
  pip install jax-directml --extra-index-url https://pypi.anaconda.org/microsoft/simple

Or build from source:
  git clone https://github.com/microsoft/jax-directml && pip install -e ./jax-directml
```

### `jaxlib and jax version mismatch`
```
Error: jaxlib version 0.4.20 is older than jax version 0.4.25

Solution: Install matching versions
  pip install "jax==0.4.25" "jaxlib==0.4.25+rocm61" \
    -f https://storage.googleapis.com/jax-releases/rocm/jax_rocm_releases.html
```

### `pip install jax[rocm]` installs CPU version
```
The default jax[rocm] may pull CPU jaxlib on some systems.
Explicitly request the rocm wheel:
  pip install jax "jaxlib[rocm6_1]==0.4.25" \
    -f https://storage.googleapis.com/jax-releases/rocm/jax_rocm_releases.html
```

---

## GPU Detection Failures

### `RuntimeError: Unknown backend: 'gpu'`
```
JAX can't find any GPU backend.
Windows: pip install jax-directml
Linux:   pip install jax[rocm6_1] -f <releases url>
Check:   python -c "import jax; print(jax.devices())"
```

### `No GPU/TPU found, falling back to CPU`
```
JAX found no GPU device. Debug steps:
1. Windows: check Device Manager — is GPU listed without error?
2. Linux: run rocm-smi — does it show your GPU?
3. WSL2: ls /dev/dri — do you see renderD128?
4. Set explicitly: JAX_PLATFORMS=gpu python your_script.py
```

### `DirectML device found but not used`
```
Force DirectML:
  import os
  os.environ["JAX_PLATFORMS"] = "gpu"
  os.environ["DIRECTML_DEBUG"] = "1"  # verbose logging
  import jax
```

---

## Runtime Errors

### `NaN values in output`
```
Common on RX 5xxx/6xxx series.
Fix 1: Disable Triton GEMM
  export XLA_FLAGS='--xla_gpu_enable_triton_gemm=false'

Fix 2: Use float32 instead of bfloat16
  x = x.astype(jnp.float32)

Fix 3: Check for division by zero in your model
  jax.debug.print("{x}", x=x)  # print from inside jit
```

### `RESOURCE_EXHAUSTED: Out of memory`
```
AMD GPU VRAM full.
Fix 1: Disable pre-allocation
  export XLA_PYTHON_CLIENT_PREALLOCATE=false

Fix 2: Limit fraction
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

Fix 3: Reduce batch size, use gradient checkpointing
  from jax.checkpoint import checkpoint
  @checkpoint
  def forward(params, x): ...
```

### `MIOpen: Invalid argument` (ROCm)
```
MIOpen kernel cache issue after ROCm version change.
Fix:
  export MIOPEN_USER_DB_PATH=/tmp/miopen-fresh
  mkdir -p /tmp/miopen-fresh
  # JAX will rebuild kernel cache on first run (slow once, fast after)
```

### `Illegal instruction (core dumped)`
```
CPU instruction set mismatch — jaxlib was compiled for AVX2 but your CPU lacks it.
Fix: Install CPU-only jaxlib
  pip install "jaxlib==0.4.25+cpu" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

---

## WSL2-Specific Issues

### GPU not visible inside WSL2
```
Fix 1: Install AMD WSL2 driver on Windows (not Linux side)
  Download from amd.com/support → your GPU → Windows 11 → WSL

Fix 2: Verify GPU is visible
  ls /dev/dri/  # must show renderD128

Fix 3: Add user to groups
  sudo usermod -aG render,video $USER
  # Log out and back in, or: newgrp render
```

### `rocm-smi: command not found` in WSL2
```
ROCm tools not installed in WSL2.
  sudo apt install rocm-smi-lib
```

### JAX in WSL2 uses CPU despite GPU being visible
```
Set env vars before launching:
  export HIP_VISIBLE_DEVICES=0
  export ROCR_VISIBLE_DEVICES=0
  export JAX_PLATFORMS=gpu
  python your_script.py
```

---

## Performance Issues

### Extremely slow first run (60+ seconds)
```
Normal behavior. JAX/XLA compiles GPU kernels on first call.
After first run, compilations are cached.
Speed up repeated runs:
  export JAX_COMPILATION_CACHE_DIR=/tmp/jax-cache
  # Persist cache between runs
```

### GPU is used but slower than CPU
```
Possible causes:
1. Data transfer overhead (small tensors): Use larger batches
2. Falling back to host for unsupported ops: Check XLA_FLAGS logging
   export TF_CPP_MIN_LOG_LEVEL=0
   export XLA_FLAGS='--xla_dump_to=/tmp/xla-dump'
3. Memory bandwidth bottleneck: Profile with rocprof (Linux)
```

### `vmap` much slower than expected
```
Ensure you're not accidentally re-tracing.
Use static_argnames for non-array args:
  @partial(jax.jit, static_argnames=["training"])
  def forward(params, x, training=False): ...
```

---

## Debugging Tools

```python
# Print from inside jit (AMD compatible)
@jax.jit
def debug_forward(x):
    jax.debug.print("x shape: {}", x.shape)
    jax.debug.print("x min/max: {} / {}", x.min(), x.max())
    return x * 2

# Check which device a tensor is on
x = jnp.array([1.0, 2.0])
print(x.devices())  # shows GPU or CPU

# Disable JIT to debug (runs eagerly)
with jax.disable_jit():
    result = forward(params, x)

# Force synchronize (ensure GPU ops complete)
result.block_until_ready()

# Profile GPU execution
with jax.profiler.trace("/tmp/jax-trace"):
    result = forward(params, x).block_until_ready()
# Open trace in TensorBoard or Perfetto
```
