# JAX Gotchas — Why Beginners Struggle (And AMD Makes It Worse)

JAX has a learning curve that catches everyone off guard. These are the things that break your code silently or loudly. On AMD hardware, several of these gotchas interact with driver bugs to produce extra-confusing failures.

---

## 1. Arrays Are Immutable (No In-Place Ops)

JAX arrays cannot be modified in place. This is by design — JAX needs to trace your computation graph, and mutation breaks that.

```python
# BREAKS in JAX
x = jnp.array([1, 2, 3])
x[0] = 10  # TypeError: JAX arrays are immutable

# CORRECT
x = x.at[0].set(10)   # returns a new array
x = x.at[1:3].add(5)  # also works for slices
x = x.at[2].mul(3)
```

**Why AMD makes this worse:** If you accidentally run impure code under `@jax.jit`, JAX may silently ignore the mutation on first trace but behave differently on subsequent calls. On AMD + ROCm, this sometimes produces inconsistent results rather than clean errors, making it hard to diagnose.

---

## 2. Python Control Flow Breaks JIT

JAX's `@jit` traces your function once and compiles it. Python `if`/`for`/`while` that depend on array *values* will be traced with the value from the *first call only*.

```python
# BREAKS under jit — Python if uses traced value, not runtime value
@jax.jit
def bad_function(x):
    if x > 5:        # this condition is evaluated at trace time, not runtime!
        return x * 2
    else:
        return x

# CORRECT — use JAX control flow
@jax.jit
def good_function(x):
    return jnp.where(x > 5, x * 2, x)

# For loops: use lax.scan instead of Python for
# For while loops: use lax.while_loop
# For conditionals: use lax.cond for complex branches
```

**Common AMD failure mode:** Under ROCm, silently wrong outputs (not errors) when Python conditionals are used inside jit. The function compiles fine but returns the "always true" branch result regardless of input.

**JAX control flow equivalents:**

| Python | JAX equivalent | Notes |
|--------|---------------|-------|
| `if/else` | `jnp.where(cond, a, b)` | Simple branches |
| `if/else` | `jax.lax.cond(cond, f_true, f_false, x)` | Complex branches |
| `for i in range(n)` | `jax.lax.scan(f, init, xs)` | Fixed-length loops |
| `while cond:` | `jax.lax.while_loop(cond_fn, body_fn, init)` | Dynamic loops |

---

## 3. NumPy Does NOT Mix With JAX

Standard NumPy arrays and JAX arrays look identical but are incompatible inside jit-compiled code.

```python
import numpy as np
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])

# BREAKS — np.sin doesn't understand JAX arrays
y = np.sin(x)   # silently copies to CPU, loses GPU placement

# CORRECT — always use jnp inside jit-compiled functions
y = jnp.sin(x)
```

**Rule of thumb:** Inside any function decorated with `@jax.jit` or passed to `jax.grad`/`jax.vmap`, use **only** `jnp.*`. NumPy is fine for preprocessing data *before* it enters JAX.

**AMD-specific gotcha:** On DirectML (Windows), mixing `np` and `jnp` often causes silent fallback to CPU without error. You'll see GPU utilization at 0% and wonder why your code is slow.

---

## 4. Functional Programming — No Side Effects

JAX requires pure functions. Functions must not modify global state, use random state implicitly, or have hidden side effects.

```python
# JAX hates this — impure function
counter = 0
def bad_function(x):
    global counter
    counter += 1    # side effect!
    return x * counter

# JAX loves this — pure function, same input always gives same output
def good_function(x, counter):
    return x * counter
```

**Why?** JAX needs to analyze your code statically to compile it, compute derivatives, and parallelize it. Hidden state breaks all three.

---

## 5. Random Numbers Need Explicit Keys

JAX doesn't use global random state (that would break reproducibility and parallelization). You must explicitly thread random keys through your code.

```python
from jax import random

# Always start with a key
key = random.PRNGKey(42)

# Split key before using — never reuse a key
key, subkey = random.split(key)
x = random.normal(subkey, shape=(1000,))

# For multiple uses in one function, split multiple times
key, k1, k2, k3 = random.split(key, 4)
a = random.normal(k1, shape=(100,))
b = random.uniform(k2, shape=(100,))
c = random.randint(k3, shape=(100,), minval=0, maxval=10)
```

**AMD gotcha:** PRNG behavior on AMD GPUs has been reported to differ from CUDA in edge cases (particularly with very large arrays). Always validate that your random distributions have expected mean/variance when running on AMD for the first time.

---

## 6. Shape Must Be Known at Trace Time

JAX needs to know array shapes at compile time. Dynamic shapes (where shape depends on array values) don't work under jit.

```python
# BREAKS — shape depends on array value
@jax.jit
def bad_filter(x):
    return x[x > 0]   # output shape unknown at trace time!

# CORRECT — keep shape fixed, mask instead
@jax.jit
def good_filter(x):
    mask = x > 0
    return jnp.where(mask, x, 0.0)   # same shape, zeros for filtered values
```

---

## 7. First Run Is Always Slow

JAX compiles GPU kernels on the first call. This is **expected** behavior, not a bug.

```python
import time

@jax.jit
def matmul(a, b):
    return a @ b

a = jnp.ones((1000, 1000))
b = jnp.ones((1000, 1000))

# First call: slow (compilation)
t0 = time.time()
result = matmul(a, b).block_until_ready()
print(f"First call: {time.time()-t0:.2f}s")  # could be 10-60s on AMD

# Subsequent calls: fast (cached)
t0 = time.time()
result = matmul(a, b).block_until_ready()
print(f"Second call: {time.time()-t0:.4f}s")  # milliseconds
```

**AMD is especially slow on first compile** — ROCm's XLA kernel compilation for AMD is 2-5x slower than CUDA. On RX 5700 XT, expect 30-90 seconds for complex models. Persist the cache:

```bash
export JAX_COMPILATION_CACHE_DIR=/tmp/jax-cache
# Survives between Python runs, not between kernel version changes
```

---

## 8. `block_until_ready()` — JAX Is Asynchronous

JAX dispatches operations asynchronously. If you time your code without `block_until_ready()`, you'll measure dispatch time (microseconds), not actual computation.

```python
# WRONG timing — only measures dispatch
t0 = time.time()
result = jnp.dot(a, b)
print(time.time() - t0)   # looks instant, not actually done

# CORRECT timing — waits for GPU to finish
t0 = time.time()
result = jnp.dot(a, b).block_until_ready()
print(time.time() - t0)   # actual GPU compute time
```

**AMD gotcha:** On some ROCm versions, missing `block_until_ready()` can cause memory not to be freed correctly, leading to gradual VRAM exhaustion during benchmarks.

---

## 9. AMD-Specific Gotchas Not in Standard Docs

These are failure modes unique to AMD hardware that don't appear in the official JAX documentation:

### NaN Epidemic on RX 5xxx/6xxx
Matrix multiplications in bfloat16 on certain AMD GPUs produce NaN silently.
```bash
# Fix: disable Triton GEMM
export XLA_FLAGS='--xla_gpu_enable_triton_gemm=false'
```

### MIOpen Kernel Cache Corruption
After any ROCm version change, MIOpen's kernel cache becomes invalid and causes `Invalid argument` errors.
```bash
export MIOPEN_USER_DB_PATH=/tmp/miopen-$(date +%Y%m%d)
mkdir -p $MIOPEN_USER_DB_PATH
```

### DirectML on Windows: No JIT Error Messages
When JAX-DirectML fails to compile a kernel, it sometimes silently falls back to CPU without warning. Enable debug logging:
```python
import os
os.environ["DIRECTML_DEBUG"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
import jax
```

### bfloat16 Is Broken on Some AMD GPUs
```python
# Check if bfloat16 works on your hardware before using it
x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.bfloat16)
y = jnp.dot(x, x)
if jnp.isnan(y):
    print("bfloat16 is broken on this GPU — use float32")
```

---

## Quick Reference Card

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| `TypeError: JAX arrays are immutable` | In-place assignment | Use `.at[i].set(v)` |
| Wrong output silently | Python if inside jit | Use `jnp.where` or `lax.cond` |
| Code runs on CPU despite GPU install | `np.*` inside jit | Replace all `np.` with `jnp.` |
| NaN output | AMD bfloat16 bug | `export XLA_FLAGS='--xla_gpu_enable_triton_gemm=false'` |
| `MIOpen: Invalid argument` | Stale kernel cache | Set `MIOPEN_USER_DB_PATH` to fresh dir |
| First run takes 60+ seconds | XLA kernel compilation | Normal, use compilation cache |
| Benchmarks show 0ms | Missing `block_until_ready()` | Add `.block_until_ready()` |
| OOM on large arrays | JAX pre-allocates VRAM | Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` |

---

See also: [troubleshooting.md](troubleshooting.md) for full error reference, [rocm-migration.md](rocm-migration.md) for ROCm version issues.
