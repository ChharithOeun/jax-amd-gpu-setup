---
name: Bug Report
about: Something breaks on your AMD GPU
title: "[BUG] <brief description>"
labels: bug
assignees: ChharithOeun
---

## Error
<!-- Paste the exact error message -->
```
paste error here
```

## Environment
- **GPU:**
- **OS:** (Linux / WSL2 / Windows DirectML)
- **ROCm / DirectML version:**
- **JAX version:** (`python -c "import jax; print(jax.__version__)"`)

## Minimal Reproducer
```python
import jax
import jax.numpy as jnp

# paste the smallest code that reproduces the error
```

## Expected vs Actual
- **Expected:**
- **Actual:**

## Workarounds Tried
- [ ] `XLA_FLAGS='--xla_gpu_enable_triton_gemm=false'`
- [ ] Fresh MIOpen cache dir
- [ ] Disabled JAX pre-allocation
- [ ] Other:
