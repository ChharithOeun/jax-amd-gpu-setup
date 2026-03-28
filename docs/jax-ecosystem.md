# JAX Ecosystem — What Builds on JAX

JAX is the foundation. Researchers and engineers have built a rich set of libraries on top of it. Here's the map — plus notes on which ones actually work on AMD.

---

## Core JAX Libraries

| Library | What It Does | AMD Status | Install |
|---------|-------------|------------|---------|
| **Flax** | PyTorch-like neural networks | Works (CPU/GPU) | `pip install flax` |
| **Haiku** | Functional neural networks (DeepMind) | Works | `pip install dm-haiku` |
| **Equinox** | Neural nets + differential equations | Works | `pip install equinox` |
| **Optax** | Optimization algorithms (Adam, SGD, etc.) | Works | `pip install optax` |
| **Jaxopt** | Implicit differentiation + optimization | Works (CPU) | `pip install jaxopt` |
| **DiffrRax** | Differential equations (ODEs, SDEs, CDEs) | Works | `pip install diffrax` |
| **Pax** | Probabilistic programming | Partial | `pip install pax-ml` |

---

## Library Deep Dives

### Flax — PyTorch for JAX
The most popular high-level neural network library for JAX. If you're coming from PyTorch, start here.

```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class MLP(nn.Module):
    features: tuple = (128, 64, 10)

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        return nn.Dense(self.features[-1])(x)

model = MLP()
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones((1, 784)))   # MNIST input
output = model.apply(params, jnp.ones((32, 784)))
```

**AMD compatibility:** Flax works well on AMD ROCm. No known AMD-specific issues with the core API.

---

### Haiku — Functional Neural Networks (DeepMind style)
Haiku is what DeepMind uses internally. Cleaner and more explicit than Flax. Better for researchers who want full control.

```python
import haiku as hk
import jax
import jax.numpy as jnp

def forward(x):
    mlp = hk.Sequential([
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64),  jax.nn.relu,
        hk.Linear(10),
    ])
    return mlp(x)

model = hk.transform(forward)
key = jax.random.PRNGKey(42)
params = model.init(key, jnp.ones((1, 784)))
output = model.apply(params, key, jnp.ones((32, 784)))
```

**AMD compatibility:** Works on ROCm. The `hk.transform` pattern plays well with JAX's functional style.

---

### Equinox — Neural Nets + Differential Equations
Equinox is the cutting-edge choice for research at the intersection of deep learning and scientific computing. Neural ODEs, physics-informed neural networks, score-based diffusion — this is the library.

```python
import equinox as eqx
import jax
import jax.numpy as jnp

class SimpleNet(eqx.Module):
    layers: list

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(2, 64, key=k1),
            eqx.nn.Linear(64, 1, key=k2),
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

model = SimpleNet(jax.random.PRNGKey(0))
x = jnp.array([1.0, 2.0])
y = model(x)
```

**AMD compatibility:** Works. Equinox's pure-function approach is very compatible with JAX's requirements.

---

### Optax — Gradient-Based Optimization
Optax provides all the standard optimizers: Adam, SGD, RMSProp, AdaGrad, plus gradient clipping, learning rate schedules, and more.

```python
import optax

# Adam with learning rate schedule
schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=1000)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),    # gradient clipping
    optax.adam(learning_rate=schedule)
)

# Initialize optimizer state
opt_state = optimizer.init(params)

# Training step
@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```

**AMD compatibility:** Fully compatible — pure JAX operations, no GPU-specific code.

---

### DiffrRax — Differential Equations
Solve ODEs, SDEs, and CDEs entirely in JAX. Differentiable through the solver. Used for Neural ODEs, physics simulations, financial models.

```python
import diffrax

def vector_field(t, y, args):
    return -y  # dy/dt = -y, solution is e^(-t)

term = diffrax.ODETerm(vector_field)
solver = diffrax.Dopri5()

solution = diffrax.diffeqsolve(
    term,
    solver,
    t0=0.0,
    t1=5.0,
    dt0=0.1,
    y0=jnp.array([1.0]),
)
print(solution.ys)  # [e^0, e^-1, e^-2, ...]
```

**AMD compatibility:** Works on ROCm. The ODE solver is implemented in pure JAX, so it runs wherever JAX runs. The `research_failure_suite.py` includes a Neural ODE test using DiffrRax.

---

### Jaxopt — Implicit Differentiation
Jaxopt provides optimizers that are themselves differentiable — you can compute gradients through the optimization process itself.

```python
import jaxopt

solver = jaxopt.LBFGS(fun=loss_fn)
result = solver.run(params_init, data=data)
optimal_params = result.params
```

**AMD compatibility:** CPU-only works well. GPU acceleration of inner solvers has mixed results on AMD.

---

## Who Uses JAX (Real Examples)

### DeepMind
Most of DeepMind's recent research uses JAX. AlphaFold 2, Chinchilla, Gato — all written in JAX. They built Haiku specifically for their workflow. When you read a DeepMind paper with published code, it's almost always JAX + Haiku.

### Google Brain
Google Brain develops JAX itself and uses it for research. Flax is their recommended high-level framework. T5X (large language model training) is built on JAX.

### Quant Traders / Finance
JAX's killer feature for quant finance is `vmap` + `grad`:
- Run 10,000 backtests in parallel with `vmap`
- Optimize strategy parameters with `grad` (gradient descent on Sharpe ratio)
- Bayesian inference on market models with automatic differentiation

This is exactly what **Chharmoney** exploits — see `examples/chharmoney_demo.py`.

### Physicists
Solving partial differential equations, quantum circuit simulation, molecular dynamics. JAX's automatic differentiation through any computation (including ODE solvers) makes it the tool of choice for physics-informed neural networks.

### Biologists / AlphaFold 2
AlphaFold 2 — the protein structure prediction model that won the Nobel Prize — is written in JAX. The ability to differentiate through complex geometric operations made JAX the right tool.

### ML Researchers
Researchers who need custom gradient computations (things that don't exist in PyTorch's autograd), differentiable programming, or who want to write math-first code that compiles efficiently.

---

## AMD Compatibility Matrix

| Library | CPU | ROCm (Linux) | DirectML (Windows) | Notes |
|---------|-----|-------------|-------------------|-------|
| Flax | ✅ | ✅ | ✅ | No known issues |
| Haiku | ✅ | ✅ | ✅ | No known issues |
| Equinox | ✅ | ✅ | ✅ | Works great |
| Optax | ✅ | ✅ | ✅ | Pure JAX, fully compatible |
| DiffrRax | ✅ | ✅ | ⚠️ | Some solvers slow on DirectML |
| Jaxopt | ✅ | ⚠️ | ❌ | GPU acceleration limited |
| Pax | ✅ | ⚠️ | ❌ | Sparse AMD testing |

---

## Learning Path (From Primer to AMD Research)

This is the recommended progression, adapted for AMD users:

**Step 1 — Get NumPy fluent** (you probably know this)
```python
import numpy as np
x = np.array([1, 2, 3])
```

**Step 2 — Learn `jax.grad`** (automatic differentiation)
```python
import jax
f = lambda x: x**2 + jnp.sin(x)
df = jax.grad(f)
print(df(2.5))
```

**Step 3 — Learn `jax.jit`** (JIT compilation)
```python
@jax.jit
def fast_fn(x): return jnp.dot(x, x)
```

**Step 4 — Learn `jax.vmap`** (vectorization — the AMD killer feature)
```python
single = lambda x: jnp.sum(x**2)
batched = jax.vmap(single)
result = batched(jnp.ones((1000, 100)))  # 1000 parallel
```

**Step 5 — Learn `jax.lax.scan`** (loops in JAX)
```python
def rolling_sum(carry, x):
    return carry + x, carry + x
_, cumsum = jax.lax.scan(rolling_sum, 0.0, data)
```

**Step 6 — Pick a library**: Flax (neural nets) or Equinox (research)

**Step 7 — Run the research failure suite** (see `scripts/research_failure_suite.py`)
This maps exactly where JAX breaks on your specific AMD GPU. The failures ARE the documentation.

**Step 8 — Read DeepMind papers** — their JAX code shows advanced patterns

---

## AMD Compatibility Status

All libraries above work on AMD — with caveats. Here's the current state:

| Library | Windows (DirectML) | Linux/WSL2 (ROCm) | Notes |
|---------|-------------------|-------------------|-------|
| Flax | ✅ | ✅ | No known AMD issues |
| Haiku | ✅ | ✅ | No known AMD issues |
| Equinox | ✅ | ✅ | Works well |
| Optax | ✅ | ✅ | Pure JAX — fully portable |
| DiffrRax | ✅ | ✅ | Some solvers slower on DirectML |
| Jaxopt | ✅ | ⚠️ | GPU acceleration limited |

Installation on AMD is non-trivial. The ecosystem was built assuming CUDA. Known issues:
- First-run compilation is 2-5x slower than NVIDIA (ROCm XLA kernel compile)
- bfloat16 produces NaN on some RX 5xxx/6xxx GPUs (workaround in [jax-gotchas.md](jax-gotchas.md))
- Some ops fall back to CPU silently — no error, just slow

Run the [research failure suite](../scripts/research_failure_suite.py) on your hardware and submit your results. See [RESULTS/README.md](../RESULTS/README.md) for the community compatibility matrix.

---

See also: [jax-gotchas.md](jax-gotchas.md), [troubleshooting.md](troubleshooting.md)
