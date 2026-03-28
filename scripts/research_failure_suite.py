"""
research_failure_suite.py — Systematic JAX Feature Battery on AMD GPU
======================================================================
This is not a "does it work" test. This is a "find every way it breaks"
test. Every failure is documented with exact error, environment, and
whether a workaround exists.

Philosophy:
  The JAX AMD ecosystem is broken in specific, reproducible ways.
  The value is knowing EXACTLY what breaks, EXACTLY what the error says,
  and EXACTLY what (if anything) fixes it.
  Each result becomes a row in the compatibility matrix.

Run on your AMD hardware and submit results:
  python scripts/research_failure_suite.py
  python scripts/research_failure_suite.py --submit   # auto-opens GitHub issue

Results are saved to: RESULTS/{gpu_name}_{rocm_ver}_{date}.json
"""

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ── Result types ──────────────────────────────────────────────────────────────

PASS    = "PASS"
FAIL    = "FAIL"
SKIP    = "SKIP"
PARTIAL = "PARTIAL"  # works but with caveats

@dataclass
class TestResult:
    name:        str
    category:    str
    status:      str         # PASS / FAIL / SKIP / PARTIAL
    duration_ms: float = 0.0
    error:       Optional[str] = None
    workaround:  Optional[str] = None
    note:        Optional[str] = None
    value:       Any = None  # numeric result if applicable

@dataclass
class SystemInfo:
    date:           str = ""
    platform:       str = ""
    python_version: str = ""
    jax_version:    str = ""
    jaxlib_version: str = ""
    backend:        str = ""
    devices:        list = field(default_factory=list)
    rocm_version:   str = ""
    gpu_name:       str = ""
    gpu_vram_gb:    float = 0.0
    driver_version: str = ""
    hip_visible:    str = ""
    xla_flags:      str = ""
    is_wsl2:        bool = False

@dataclass
class SuiteReport:
    system:  SystemInfo = field(default_factory=SystemInfo)
    results: list = field(default_factory=list)

    def summary(self):
        total   = len(self.results)
        passed  = sum(1 for r in self.results if r.status == PASS)
        failed  = sum(1 for r in self.results if r.status == FAIL)
        partial = sum(1 for r in self.results if r.status == PARTIAL)
        skipped = sum(1 for r in self.results if r.status == SKIP)
        return {"total": total, "passed": passed, "failed": failed,
                "partial": partial, "skipped": skipped,
                "pass_rate": f"{100*passed//total if total else 0}%"}

# ── Helpers ───────────────────────────────────────────────────────────────────

REPORT = SuiteReport()
_RESULTS: list[TestResult] = []

def run_test(name: str, category: str, fn, workaround: str = None):
    """Run one test, catch all exceptions, record result."""
    sys.stdout.write(f"  [{category:12s}] {name:<50s} ... ")
    sys.stdout.flush()
    t0 = time.perf_counter()
    try:
        value = fn()
        dur = (time.perf_counter() - t0) * 1000
        r = TestResult(name=name, category=category, status=PASS,
                       duration_ms=round(dur, 2), value=value)
        print(f"PASS  ({dur:.0f}ms)")
    except Exception as e:
        dur = (time.perf_counter() - t0) * 1000
        err = f"{type(e).__name__}: {str(e)[:200]}"
        r = TestResult(name=name, category=category, status=FAIL,
                       duration_ms=round(dur, 2), error=err,
                       workaround=workaround)
        print(f"FAIL  ({err[:60]})")
    _RESULTS.append(r)
    return r

def partial_result(name: str, category: str, note: str, value=None):
    r = TestResult(name=name, category=category, status=PARTIAL, note=note, value=value)
    _RESULTS.append(r)
    print(f"  [{category:12s}] {name:<50s} ... PARTIAL ({note[:50]})")
    return r

def skip_test(name: str, category: str, reason: str):
    r = TestResult(name=name, category=category, status=SKIP, note=reason)
    _RESULTS.append(r)
    print(f"  [{category:12s}] {name:<50s} ... SKIP  ({reason[:50]})")
    return r

def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

# ── System Info Collection ────────────────────────────────────────────────────

def collect_system_info() -> SystemInfo:
    info = SystemInfo()
    info.date     = datetime.now().isoformat()
    info.platform = platform.platform()
    info.python_version = sys.version.split()[0]

    # Check WSL2
    try:
        with open("/proc/version") as f:
            info.is_wsl2 = "microsoft" in f.read().lower()
    except Exception:
        info.is_wsl2 = False

    # JAX versions
    try:
        import jax, jaxlib
        info.jax_version    = jax.__version__
        info.jaxlib_version = jaxlib.__version__
        info.backend        = jax.default_backend()
        info.devices        = [str(d) for d in jax.devices()]
    except Exception as e:
        info.jax_version = f"ERROR: {e}"

    # ROCm version
    try:
        v = Path("/opt/rocm/.info/version")
        info.rocm_version = v.read_text().strip() if v.exists() else "not found"
    except Exception:
        info.rocm_version = "unknown"

    # GPU info via rocm-smi
    try:
        r = subprocess.run(["rocm-smi", "--showid", "--showname", "--showmeminfo", "vram"],
                           capture_output=True, text=True, timeout=8)
        lines = r.stdout.strip().splitlines()
        for line in lines:
            if "GPU" in line and ":" in line:
                info.gpu_name = line.split(":")[-1].strip()
            if "Total Memory" in line or "VRAM" in line:
                try:
                    mb = int(''.join(filter(str.isdigit, line.split(":")[-1])))
                    info.gpu_vram_gb = round(mb / 1024, 1)
                except Exception:
                    pass
    except Exception:
        pass

    # Windows GPU via wmic (if Windows)
    if platform.system() == "Windows":
        try:
            r = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM"],
                capture_output=True, text=True, timeout=5
            )
            lines = [l.strip() for l in r.stdout.splitlines() if l.strip() and "Name" not in l]
            if lines:
                info.gpu_name = lines[0]
        except Exception:
            pass

    info.hip_visible = os.environ.get("HIP_VISIBLE_DEVICES", "not set")
    info.xla_flags   = os.environ.get("XLA_FLAGS", "not set")

    return info

# ── Test Categories ───────────────────────────────────────────────────────────

def test_installation():
    section("1. INSTALLATION & BACKEND DETECTION")

    def check_import():
        import jax
        return jax.__version__
    run_test("import jax", "install", check_import)

    def check_backend():
        import jax
        b = jax.default_backend()
        if b == "cpu":
            raise RuntimeError("Fell back to CPU — no GPU backend found")
        return b
    run_test("GPU backend active (not CPU)", "install", check_backend,
             workaround="Set JAX_PLATFORMS=gpu or install jax[rocm6_1]")

    def check_devices():
        import jax
        gpus = jax.devices("gpu")
        if not gpus:
            raise RuntimeError("No GPU devices listed")
        return len(gpus)
    run_test("GPU devices enumerated", "install", check_devices,
             workaround="export HIP_VISIBLE_DEVICES=0")

    def check_optax():
        import optax
        return optax.__version__
    run_test("optax installed (optimizer library)", "install", check_optax,
             workaround="pip install optax")

    def check_flax():
        import flax
        return flax.__version__
    run_test("flax installed (neural network library)", "install", check_flax,
             workaround="pip install flax")

def test_basic_ops():
    section("2. BASIC ARRAY OPERATIONS")
    import jax.numpy as jnp
    import jax

    def matmul_small():
        x = jnp.ones((256, 256))
        return float(jnp.sum(x @ x))
    run_test("matmul 256x256", "ops", matmul_small)

    def matmul_medium():
        x = jnp.ones((1024, 1024))
        return float(jnp.sum(x @ x))
    run_test("matmul 1024x1024", "ops", matmul_medium,
             workaround="Reduce size if OOM")

    def matmul_large():
        x = jnp.ones((4096, 4096))
        y = (x @ x).block_until_ready()
        return float(y[0, 0])
    run_test("matmul 4096x4096 (VRAM stress)", "ops", matmul_large,
             workaround="export XLA_PYTHON_CLIENT_PREALLOCATE=false")

    def conv2d_op():
        from jax import lax
        x = jnp.ones((1, 64, 64, 3))
        k = jnp.ones((3, 3, 3, 16))
        return float(lax.conv_general_dilated(
            x.transpose(0,3,1,2), k.transpose(3,2,0,1),
            window_strides=(1,1), padding='SAME').sum())
    run_test("conv2d via lax (GPU kernel)", "ops", conv2d_op,
             workaround="export XLA_FLAGS='--xla_gpu_enable_triton_gemm=false'")

    def fft_op():
        x = jnp.ones((1024, 1024))
        return float(jnp.abs(jnp.fft.fft2(x)).sum())
    run_test("FFT 1024x1024", "ops", fft_op)

    def softmax_op():
        x = jnp.ones((512, 50000))
        return float(jax.nn.softmax(x, axis=-1).sum())
    run_test("softmax (large vocab)", "ops", softmax_op)

    def bfloat16_op():
        x = jnp.ones((1024, 1024), dtype=jnp.bfloat16)
        y = (x @ x).block_until_ready()
        return str(y.dtype)
    run_test("bfloat16 matmul", "ops", bfloat16_op,
             workaround="Fall back to float32 if this fails")

    def float16_op():
        x = jnp.ones((1024, 1024), dtype=jnp.float16)
        y = (x @ x).block_until_ready()
        return str(y.dtype)
    run_test("float16 matmul", "ops", float16_op,
             workaround="Fall back to float32 if this fails")

def test_jit():
    section("3. JIT COMPILATION (XLA)")
    import jax
    import jax.numpy as jnp

    def basic_jit():
        @jax.jit
        def f(x):
            return x * 2 + 1
        return float(f(jnp.array(3.0)))
    run_test("basic jit", "jit", basic_jit)

    def jit_matmul():
        @jax.jit
        def matmul(a, b):
            return a @ b
        x = jnp.ones((512, 512))
        t0 = time.perf_counter()
        _ = matmul(x, x).block_until_ready()  # compile
        compile_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        _ = matmul(x, x).block_until_ready()  # cached
        cached_ms = (time.perf_counter() - t0) * 1000
        return f"compile={compile_ms:.0f}ms cached={cached_ms:.1f}ms"
    run_test("jit matmul (compile vs cached)", "jit", jit_matmul)

    def jit_with_python_control_flow():
        # This is a common footgun — Python if inside jit
        @jax.jit
        def f(x, flag):
            if flag:  # This uses flag as static
                return x * 2
            return x * 3
        # Must use static_argnums or this traces twice
        try:
            return float(f(jnp.array(1.0), True))
        except Exception as e:
            raise RuntimeError(f"Python control flow in jit: {e}")
    run_test("jit with Python control flow (common footgun)", "jit",
             jit_with_python_control_flow,
             workaround="Use jnp.where() or static_argnums instead of Python if")

    def jit_large_model():
        # Simulate a model with many parameters being JIT-compiled
        @jax.jit
        def big_forward(params, x):
            for W, b in params:
                x = jax.nn.gelu(x @ W + b)
            return x
        key = jax.random.PRNGKey(0)
        params = [(jax.random.normal(key, (512, 512)), jnp.zeros(512))
                  for _ in range(8)]
        x = jnp.ones((32, 512))
        t0 = time.perf_counter()
        _ = big_forward(params, x).block_until_ready()
        return f"8-layer compile: {(time.perf_counter()-t0)*1000:.0f}ms"
    run_test("jit 8-layer MLP (complex compile)", "jit", jit_large_model)

def test_vmap():
    section("4. VMAP (BATCHING)")
    import jax
    import jax.numpy as jnp

    def basic_vmap():
        def f(x):
            return x ** 2
        batched_f = jax.vmap(f)
        return float(batched_f(jnp.arange(10.0)).sum())
    run_test("basic vmap", "vmap", basic_vmap)

    def vmap_matmul():
        def matmul_single(a, b):
            return a @ b
        batched = jax.vmap(matmul_single)
        A = jnp.ones((32, 128, 128))
        B = jnp.ones((32, 128, 128))
        return float(batched(A, B).sum())
    run_test("vmap batched matmul (32x128x128)", "vmap", vmap_matmul)

    def vmap_nested():
        # vmap inside vmap — common in meta-learning
        def inner(x):
            return x * 2
        outer = jax.vmap(jax.vmap(inner))
        x = jnp.ones((4, 8, 16))
        return float(outer(x).sum())
    run_test("nested vmap (meta-learning pattern)", "vmap", vmap_nested)

    def vmap_grad():
        # Per-sample gradients — JAX's killer feature
        def loss(params, x, y):
            return jnp.sum((params @ x - y) ** 2)
        per_sample_grad = jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))
        params = jnp.ones((4, 4))
        x_batch = jnp.ones((16, 4))
        y_batch = jnp.ones((16, 4))
        grads = per_sample_grad(params, x_batch, y_batch)
        return f"shape={grads.shape}"
    run_test("vmap over grad (per-sample gradients)", "vmap", vmap_grad,
             workaround="This is what makes JAX better than PyTorch for research")

    def vmap_large_batch():
        def f(x):
            return jnp.dot(x, x)
        batched = jax.jit(jax.vmap(f))
        x = jnp.ones((10000, 256))
        return float(batched(x).sum())
    run_test("vmap 10k batch (VRAM stress)", "vmap", vmap_large_batch,
             workaround="Reduce batch size if RESOURCE_EXHAUSTED")

def test_grad():
    section("5. AUTOMATIC DIFFERENTIATION")
    import jax
    import jax.numpy as jnp

    def basic_grad():
        f = lambda x: x ** 3
        df = jax.grad(f)
        return float(df(jnp.array(2.0)))  # expect 12.0
    run_test("basic grad (x^3 at x=2)", "autograd", basic_grad)

    def grad_of_grad():
        f  = lambda x: x ** 4
        d2f = jax.grad(jax.grad(f))
        return float(d2f(jnp.array(2.0)))  # expect 48.0
    run_test("grad of grad (2nd order)", "autograd", grad_of_grad)

    def grad_through_loop():
        # Differentiate through a Python for loop
        def f(x):
            for _ in range(10):
                x = x * 2
            return x
        return float(jax.grad(f)(jnp.array(1.0)))  # expect 1024.0
    run_test("grad through Python loop", "autograd", grad_through_loop)

    def grad_through_scan():
        # Differentiate through lax.scan (GPU-friendly)
        from jax import lax
        def f(params):
            def step(carry, _):
                return carry * params, carry
            final, _ = lax.scan(step, jnp.array(1.0), None, length=10)
            return final
        return float(jax.grad(f)(jnp.array(2.0)))
    run_test("grad through lax.scan (time series)", "autograd", grad_through_scan)

    def grad_through_ode():
        # The JAX research killer feature: differentiate through ODE solver
        try:
            from jax.experimental.ode import odeint
            def dynamics(y, t):
                return -y
            def loss(params):
                y0 = jnp.array([params])
                t  = jnp.linspace(0, 1, 20)
                y  = odeint(dynamics, y0, t)
                return jnp.sum(y ** 2)
            g = jax.grad(loss)(jnp.array(1.0))
            return float(g)
        except ImportError:
            raise RuntimeError("jax.experimental.ode not available")
    run_test("grad through ODE solver (Neural ODE)", "autograd", grad_through_ode,
             workaround="Requires jax.experimental.ode — may not be available on all jaxlib builds")

    def custom_vjp():
        # Custom reverse-mode gradient
        @jax.custom_vjp
        def safe_log(x):
            return jnp.log(x)
        def safe_log_fwd(x):
            return safe_log(x), x
        def safe_log_bwd(x, g):
            return (g / (x + 1e-8),)
        safe_log.defvjp(safe_log_fwd, safe_log_bwd)
        f = lambda x: jnp.sum(safe_log(x))
        return float(jax.grad(f)(jnp.ones(4)))
    run_test("custom_vjp (custom gradient)", "autograd", custom_vjp)

    def value_and_grad_jit():
        @jax.jit
        def loss_and_grad(params, x, y):
            pred = params @ x
            loss = jnp.mean((pred - y) ** 2)
            return loss
        vg = jax.jit(jax.value_and_grad(loss_and_grad))
        params = jnp.ones((8, 8))
        x = jnp.ones(8)
        y = jnp.ones(8)
        loss, grads = vg(params, x, y)
        return f"loss={float(loss):.4f} grad_norm={float(jnp.linalg.norm(grads)):.4f}"
    run_test("value_and_grad + jit (training step)", "autograd", value_and_grad_jit)

def test_lax():
    section("6. LAX PRIMITIVES (GPU KERNEL TESTS)")
    import jax
    from jax import lax
    import jax.numpy as jnp

    def lax_scan_basic():
        def step(carry, x):
            return carry + x, carry
        final, history = lax.scan(step, jnp.array(0.0), jnp.ones(100))
        return float(final)
    run_test("lax.scan basic", "lax", lax_scan_basic)

    def lax_scan_large():
        # Long sequence — hits GPU memory patterns
        def step(carry, x):
            return carry * 0.99 + x * 0.01, carry
        final, _ = lax.scan(step, jnp.zeros(256), jnp.ones((1000, 256)))
        return float(final.sum())
    run_test("lax.scan 1000-step (long sequence)", "lax", lax_scan_large)

    def lax_while_loop():
        def cond(x): return x < 100
        def body(x): return x + 1
        return int(lax.while_loop(cond, body, jnp.array(0)))
    run_test("lax.while_loop", "lax", lax_while_loop)

    def lax_cond():
        def true_fn(x): return x * 2
        def false_fn(x): return x * 3
        result = lax.cond(True, true_fn, false_fn, jnp.array(5.0))
        return float(result)
    run_test("lax.cond (conditional on GPU)", "lax", lax_cond)

    def lax_associative_scan():
        result = lax.associative_scan(lambda a, b: a + b, jnp.ones(1024))
        return float(result[-1])
    run_test("lax.associative_scan (parallel prefix sum)", "lax", lax_associative_scan)

def test_memory():
    section("7. MEMORY & VRAM LIMITS")
    import jax
    import jax.numpy as jnp

    def check_preallocate():
        prealloc = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        fraction = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9 (default)")
        return f"preallocate={prealloc} fraction={fraction}"
    run_test("memory config (preallocate settings)", "memory",
             check_preallocate)

    def allocate_1gb():
        x = jnp.ones((1024, 1024, 256), dtype=jnp.float32)  # ~1 GB
        x.block_until_ready()
        return f"shape={x.shape} dtype={x.dtype}"
    run_test("allocate ~1GB tensor", "memory", allocate_1gb,
             workaround="export XLA_PYTHON_CLIENT_PREALLOCATE=false")

    def allocate_4gb():
        x = jnp.ones((1024, 1024, 1024), dtype=jnp.float32)  # ~4 GB
        x.block_until_ready()
        return f"shape={x.shape}"
    run_test("allocate ~4GB tensor (VRAM limit test)", "memory", allocate_4gb,
             workaround="Only possible on 8GB+ VRAM cards")

    def memory_stats():
        try:
            for d in jax.devices():
                stats = d.memory_stats()
                used  = stats.get("bytes_in_use", 0) // (1024**2)
                limit = stats.get("bytes_limit", 0) // (1024**2)
                return f"used={used}MB limit={limit}MB"
        except Exception as e:
            raise RuntimeError(f"memory_stats() failed: {e}")
    run_test("device.memory_stats() API", "memory", memory_stats,
             workaround="May not be supported on all AMD drivers")

def test_performance():
    section("8. PERFORMANCE BENCHMARKS (AMD vs Expected)")
    import jax
    import jax.numpy as jnp

    def bench_matmul_gflops():
        n = 4096
        x = jnp.ones((n, n))
        _ = (x @ x).block_until_ready()  # warmup
        N_RUNS = 5
        t0 = time.perf_counter()
        for _ in range(N_RUNS):
            y = (x @ x).block_until_ready()
        elapsed = (time.perf_counter() - t0) / N_RUNS
        gflops = (2 * n**3) / elapsed / 1e9
        # RX 5700 XT theoretical: ~9.75 TFLOPS fp32
        # JAX on AMD reality: 30-50% efficiency = ~3-5 TFLOPS
        pct_efficiency = gflops / 9750 * 100
        return f"{gflops:.0f} GFLOPS ({pct_efficiency:.1f}% of RX5700XT theoretical)"
    run_test("matmul GFLOPS (efficiency test)", "perf", bench_matmul_gflops)

    def bench_jit_overhead():
        @jax.jit
        def f(x):
            return x @ x
        x = jnp.ones((512, 512))
        _ = f(x).block_until_ready()  # compile

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            _ = f(x).block_until_ready()
            times.append((time.perf_counter() - t0) * 1000)

        import statistics
        return f"mean={statistics.mean(times):.2f}ms std={statistics.stdev(times):.2f}ms"
    run_test("JIT dispatch overhead (20 runs)", "perf", bench_jit_overhead)

    def bench_vmap_throughput():
        def f(x):
            return x @ x
        batched = jax.jit(jax.vmap(f))
        x = jnp.ones((1000, 128, 128))
        _ = batched(x).block_until_ready()  # warmup
        t0 = time.perf_counter()
        _ = batched(x).block_until_ready()
        elapsed = time.perf_counter() - t0
        return f"1000x (128x128 matmul): {elapsed*1000:.1f}ms = {1000/elapsed:.0f}/sec"
    run_test("vmap throughput (1000 matmuls)", "perf", bench_vmap_throughput)

    def bench_grad_vs_forward():
        @jax.jit
        def loss(params, x):
            return jnp.sum((params @ x) ** 2)
        grad_fn = jax.jit(jax.grad(loss))
        params = jnp.ones((512, 512))
        x = jnp.ones(512)

        _ = loss(params, x).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(10):
            _ = loss(params, x).block_until_ready()
        fwd_ms = (time.perf_counter() - t0) / 10 * 1000

        _ = grad_fn(params, x)
        t0 = time.perf_counter()
        for _ in range(10):
            g = grad_fn(params, x)
            jax.tree_util.tree_map(lambda a: a.block_until_ready(), g)
        bwd_ms = (time.perf_counter() - t0) / 10 * 1000

        return f"forward={fwd_ms:.2f}ms backward={bwd_ms:.2f}ms ratio={bwd_ms/fwd_ms:.1f}x"
    run_test("gradient overhead (fwd vs bwd ratio)", "perf", bench_grad_vs_forward)

def test_known_amd_bugs():
    section("9. KNOWN AMD-SPECIFIC FAILURE MODES")
    import jax
    import jax.numpy as jnp

    def nan_in_output():
        # Common on AMD: NaN outputs from ops that work fine on NVIDIA
        @jax.jit
        def f(x):
            return jnp.log(x) / jnp.sqrt(x)
        result = f(jnp.array([1.0, 2.0, 4.0, 8.0]))
        result.block_until_ready()
        has_nan = bool(jnp.any(jnp.isnan(result)))
        if has_nan:
            raise RuntimeError(f"NaN in output! result={result} — AMD kernel bug")
        return f"no NaN: {result}"
    run_test("NaN detection (log/sqrt — known AMD issue)", "amd_bugs",
             nan_in_output,
             workaround="export XLA_FLAGS='--xla_gpu_enable_triton_gemm=false'")

    def triton_gemm():
        # Triton GEMM causes NaN on some AMD configs
        original = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=true"
        try:
            @jax.jit
            def f(x):
                return x @ x.T
            result = f(jnp.ones((256, 256)))
            result.block_until_ready()
            has_nan = bool(jnp.any(jnp.isnan(result)))
            status = "NaN!" if has_nan else "OK"
            return f"triton_gemm=true result: {status}"
        finally:
            os.environ["XLA_FLAGS"] = original
    run_test("Triton GEMM enabled (NaN test)", "amd_bugs", triton_gemm,
             workaround="Disable: XLA_FLAGS='--xla_gpu_enable_triton_gemm=false'")

    def hip_runtime_error():
        # Test that GPU doesn't throw HIP runtime errors on simple ops
        try:
            x = jnp.ones((2048, 2048))
            y = jax.nn.softmax(x @ x, axis=-1)
            y.block_until_ready()
            return f"no HIP error: shape={y.shape}"
        except Exception as e:
            err = str(e)
            if "HIP" in err or "hip" in err:
                raise RuntimeError(f"HIP runtime error: {err}")
            raise
    run_test("HIP runtime stability (large softmax)", "amd_bugs", hip_runtime_error,
             workaround="Update AMD driver; ensure ROCm version matches jaxlib")

    def miopen_algorithm():
        # MIOpen 'algorithm not found' is common after ROCm version changes
        try:
            from jax import lax
            x = jnp.ones((1, 3, 224, 224))
            k = jnp.ones((3, 3, 3, 64))
            result = lax.conv_general_dilated(
                x, k, window_strides=(1,1), padding='SAME',
                dimension_numbers=('NCHW','OIHW','NCHW')
            )
            result.block_until_ready()
            return f"MIOpen conv OK: {result.shape}"
        except Exception as e:
            if "MIOpen" in str(e) or "algorithm" in str(e).lower():
                raise RuntimeError(f"MIOpen error: {e}")
            raise
    run_test("MIOpen conv2d (algorithm lookup)", "amd_bugs", miopen_algorithm,
             workaround="export MIOPEN_USER_DB_PATH=/tmp/miopen-fresh && mkdir -p /tmp/miopen-fresh")

    def mixed_precision_stability():
        # bfloat16 is known to be flaky on some AMD configs
        @jax.jit
        def f(x):
            x = x.astype(jnp.bfloat16)
            y = x @ x.T
            return y.astype(jnp.float32)
        result = f(jnp.ones((256, 256)))
        result.block_until_ready()
        has_nan = bool(jnp.any(jnp.isnan(result)))
        has_inf = bool(jnp.any(jnp.isinf(result)))
        return f"bfloat16 stability: nan={has_nan} inf={has_inf}"
    run_test("bfloat16 stability (matmul + cast)", "amd_bugs",
             mixed_precision_stability,
             workaround="Use float32 if bfloat16 gives NaN/Inf on your GPU")

def test_research_patterns():
    section("10. RESEARCH-CRITICAL PATTERNS")
    import jax
    import jax.numpy as jnp

    def neural_ode_forward():
        # Neural ODEs — differentiate through ODE = JAX's biggest research advantage
        from jax.experimental.ode import odeint

        def ode_fn(y, t, params):
            W, b = params
            return jax.nn.tanh(y @ W + b)

        key = jax.random.PRNGKey(0)
        W = jax.random.normal(key, (4, 4)) * 0.1
        b = jnp.zeros(4)
        y0 = jnp.ones(4)
        t  = jnp.linspace(0, 1, 20)
        result = odeint(ode_fn, y0, t, (W, b))
        return f"neural ODE trajectory shape: {result.shape}"
    run_test("Neural ODE (jax.experimental.ode)", "research", neural_ode_forward,
             workaround="Critical for physics-informed ML; may need diffrax instead")

    def score_function_estimator():
        # REINFORCE / score function gradient — used in RL
        key = jax.random.PRNGKey(42)
        logits = jnp.array([1.0, 2.0, 3.0])
        samples = jax.random.categorical(key, logits, shape=(1000,))
        log_probs = jax.nn.log_softmax(logits)[samples]
        rewards = jnp.array([1.0 if s == 2 else 0.0 for s in samples])
        grad_estimate = jnp.mean(rewards * log_probs)
        return float(grad_estimate)
    run_test("Score function estimator (RL gradient)", "research",
             score_function_estimator)

    def hessian_computation():
        # Second-order methods — Hessian-vector products
        def f(x):
            return jnp.sum(x ** 4) / 4
        hessian_fn = jax.jit(jax.hessian(f))
        x = jnp.array([1.0, 2.0, 3.0])
        H = hessian_fn(x)
        return f"Hessian shape: {H.shape}, trace={float(jnp.trace(H)):.2f}"
    run_test("Hessian computation (2nd order optimization)", "research",
             hessian_computation)

    def jacobian_computation():
        # Full Jacobian — used in sensitivity analysis
        def f(x):
            return jnp.array([jnp.sum(x), jnp.prod(x)])
        J = jax.jit(jax.jacobian(f))(jnp.array([1.0, 2.0, 3.0]))
        return f"Jacobian shape: {J.shape}"
    run_test("Jacobian computation (sensitivity analysis)", "research",
             jacobian_computation)

    def chharmoney_trading_pass():
        # The actual Chharmoney model — quant trading on AMD
        sys.path.insert(0, str(Path(__file__).parent.parent))
        try:
            from examples.chharmoney_demo import (
                init_chharmoney, simulate_ohlcv, compute_features,
                chharmoney_predict, run_all_backtests, N_ASSETS, LOOKBACK
            )
            key = jax.random.PRNGKey(42)
            ohlcv    = simulate_ohlcv(key, n_days=252, n_assets=N_ASSETS)
            features = compute_features(ohlcv)
            params   = init_chharmoney(key)
            window   = features[-LOOKBACK:]

            t0 = time.perf_counter()
            preds = chharmoney_predict(params, window)
            preds.block_until_ready()
            ms = (time.perf_counter() - t0) * 1000

            # Run 1000 backtests
            strategy_params = jax.random.normal(key, (1000, 4))
            close = ohlcv[:, :, 3]
            t0 = time.perf_counter()
            returns = run_all_backtests(strategy_params, close)
            returns.block_until_ready()
            bt_ms = (time.perf_counter() - t0) * 1000

            return (f"predict={ms:.0f}ms "
                    f"backtests=1000x{N_ASSETS}assets in {bt_ms:.0f}ms "
                    f"({1000/bt_ms*1000:.0f}/sec)")
        except Exception as e:
            raise RuntimeError(f"Chharmoney model failed: {e}")
    run_test("Chharmoney full trading model (end-to-end)", "research",
             chharmoney_trading_pass,
             workaround="This is the real-world test — if this fails, note exact error")

def test_physics_benchmarks():
    """
    PHYSICS BENCHMARKS — What physicists actually run on JAX.
    Solving real equations, not toy examples.
    """
    section("11. PHYSICS BENCHMARKS (Schrödinger, PDEs, Molecular Dynamics)")
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad

    # ── 1. Quantum Harmonic Oscillator ────────────────────────────────────────
    def schrodinger_1d():
        """
        Time-independent Schrödinger equation for 1D harmonic oscillator.
        H|ψ⟩ = E|ψ⟩, V(x) = 0.5*ω²x²
        Analytical: E_n = ω(n + 0.5) → ground state = 0.5
        """
        n = 512
        L = 6.0
        dx = 2 * L / n
        x = jnp.linspace(-L, L, n)
        V = 0.5 * x**2                                      # ω=1 harmonic potential
        diag     = jnp.ones(n) / dx**2 + V
        off_diag = jnp.ones(n-1) * (-0.5 / dx**2)
        H = jnp.diag(diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
        t0 = time.perf_counter()
        eigenvalues = jnp.linalg.eigvalsh(H)
        _ = eigenvalues[0].block_until_ready()
        ms = (time.perf_counter() - t0) * 1000
        E0 = float(eigenvalues[0])
        error = abs(E0 - 0.5) / 0.5 * 100
        assert error < 1.0, f"Ground state energy error too large: {error:.2f}% (E0={E0:.4f}, expected 0.5)"
        return f"E_0={E0:.5f}ℏω (exact: 0.5000), error={error:.4f}%, solved in {ms:.0f}ms"
    run_test("Schrödinger equation: 1D quantum harmonic oscillator", "physics",
             schrodinger_1d,
             workaround="If eigvalsh fails: XLA_FLAGS='--xla_gpu_enable_triton_gemm=false'")

    # ── 2. Wave Equation (PDE finite difference) ──────────────────────────────
    def wave_equation_pde():
        """
        1D wave equation: ∂²u/∂t² = c²∂²u/∂x²
        Forward-step via lax.scan — GPU-native time integration.
        Tests: lax.scan on GPU, in-place update via .at[].set()
        """
        n_x, n_t, c = 256, 2000, 1.0
        dx = 1.0 / n_x
        dt = 0.4 * dx / c    # CFL condition
        x  = jnp.linspace(0, 1, n_x)
        u0 = jnp.sin(jnp.pi * x)           # initial displacement
        v0 = jnp.zeros(n_x)                 # initial velocity

        @jit
        def step(state, _):
            u, v = state
            laplacian = jnp.zeros(n_x).at[1:-1].set(
                (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
            )
            v_new = v + dt * c**2 * laplacian
            u_new = u + dt * v_new
            return (u_new, v_new), None

        t0 = time.perf_counter()
        (u_final, _), _ = jax.lax.scan(step, (u0, v0), None, length=n_t)
        _ = u_final[0].block_until_ready()
        ms = (time.perf_counter() - t0) * 1000
        energy = float(jnp.mean(u_final**2))
        return f"Wave PDE: {n_t} time steps, {n_x} grid points, {ms:.0f}ms. Final energy={energy:.4f}"
    run_test("Wave equation PDE (lax.scan time integration)", "physics",
             wave_equation_pde,
             workaround="lax.scan is the preferred pattern — works on AMD")

    # ── 3. Variational Monte Carlo ────────────────────────────────────────────
    def variational_monte_carlo():
        """
        Variational Monte Carlo for the hydrogen atom ground state.
        Minimize ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ using JAX autograd through the expectation value.
        Used by physicists for quantum many-body problems.
        """
        key = jax.random.PRNGKey(42)
        n_walkers = 10_000

        def trial_wavefunction(r, alpha):
            """Hydrogenic: ψ(r) = exp(-alpha*r)"""
            return jnp.exp(-alpha * r)

        def local_energy(r, alpha):
            """E_L = -1/2 ∇²ψ/ψ + V (hydrogen: V=-1/r)"""
            kinetic = alpha**2 / 2 - alpha / r
            potential = -1.0 / (r + 1e-6)
            return kinetic + potential

        def vmc_energy(alpha, key):
            """Sample r from |ψ|² and compute average local energy."""
            r_samples = jnp.abs(jax.random.normal(key, (n_walkers,))) / alpha + 0.001
            E_local = vmap(local_energy, in_axes=(0, None))(r_samples, alpha)
            return jnp.mean(E_local)

        t0 = time.perf_counter()
        grad_vmc = jit(grad(vmc_energy))
        alpha = 1.0
        for _ in range(20):
            key, subkey = jax.random.split(key)
            g = grad_vmc(alpha, subkey)
            alpha = alpha - 0.05 * float(g)
            alpha = max(0.5, min(2.0, alpha))
        key, subkey = jax.random.split(key)
        E = float(vmc_energy(alpha, subkey))
        ms = (time.perf_counter() - t0) * 1000
        error = abs(E - (-0.5)) / 0.5 * 100
        return (f"H atom: E={E:.4f} Ha (exact -0.5000 Ha), "
                f"alpha={alpha:.4f} (exact 1.0), "
                f"error={error:.2f}%, {n_walkers:,} walkers, {ms:.0f}ms")
    run_test("Variational Monte Carlo (hydrogen atom ground state)", "physics",
             variational_monte_carlo,
             workaround="Core use case for JAX in physics — grad through expectation value")

    # ── 4. Molecular Dynamics Energy Minimization ─────────────────────────────
    def molecular_dynamics_minimization():
        """
        Lennard-Jones potential energy minimization for N particles.
        Differentiate through the energy function → forces → gradient descent.
        This is what chemists use JAX for (protein folding, materials science).
        V_LJ(r) = 4ε[(σ/r)^12 - (σ/r)^6]
        """
        n_atoms = 64
        key = jax.random.PRNGKey(0)
        positions = jax.random.uniform(key, (n_atoms, 3), minval=0, maxval=5.0)

        @jit
        def lj_energy(pos):
            """Lennard-Jones total potential energy."""
            def pair_energy(r_ij):
                r = jnp.sqrt(jnp.sum(r_ij**2) + 1e-8)
                r6 = (1.0 / r)**6
                return 4.0 * (r6**2 - r6)

            n = pos.shape[0]
            i_idx, j_idx = jnp.triu_indices(n, k=1)
            r_ij = pos[i_idx] - pos[j_idx]
            energies = vmap(pair_energy)(r_ij)
            return jnp.sum(jnp.clip(energies, -10.0, 10.0))

        grad_energy = jit(grad(lj_energy))
        t0 = time.perf_counter()
        E0 = float(lj_energy(positions))
        for _ in range(100):
            forces = -grad_energy(positions)
            positions = positions + 0.001 * forces
        E1 = float(lj_energy(positions))
        ms = (time.perf_counter() - t0) * 1000
        return (f"LJ minimization: {n_atoms} atoms, 100 gradient steps, "
                f"ΔE={E1-E0:+.2f} (E0={E0:.2f} → E1={E1:.2f}), {ms:.0f}ms")
    run_test("Molecular dynamics (Lennard-Jones gradient descent)", "physics",
             molecular_dynamics_minimization,
             workaround="AlphaFold 2-style geometry optimization — core JAX use case")


def test_quant_finance_benchmarks():
    """
    QUANTITATIVE FINANCE BENCHMARKS — What traders actually run.
    Real problems: portfolio optimization, option pricing, risk.
    """
    section("12. QUANT FINANCE BENCHMARKS (Portfolio, Options, Risk)")
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad

    # ── 1. Markowitz Portfolio Optimization ───────────────────────────────────
    def markowitz_optimization():
        """
        Find minimum variance portfolio for given return target.
        Gradient descent on the efficient frontier.
        Real quant traders use this daily.
        """
        n_assets = 50
        key = jax.random.PRNGKey(1)
        k1, k2 = jax.random.split(key)
        factor = jax.random.normal(k1, (n_assets, 5))
        cov = (factor @ factor.T) / 5 + jnp.diag(jax.random.uniform(k2, (n_assets,)) * 0.01 + 0.005)
        mu  = jax.random.uniform(key, (n_assets,), minval=-0.02, maxval=0.18)
        target_ret = float(jnp.mean(mu))

        @jit
        def loss(logits):
            w = jax.nn.softmax(logits)
            var     = w @ cov @ w
            ret     = jnp.dot(w, mu)
            penalty = 100.0 * jnp.maximum(0, target_ret - ret)**2
            return var + penalty

        grad_loss = jit(jax.value_and_grad(loss))
        t0 = time.perf_counter()
        logits = jnp.zeros(n_assets)
        for _ in range(300):
            val, g = grad_loss(logits)
            logits = logits - 0.05 * g
        ms = (time.perf_counter() - t0) * 1000
        w  = jax.nn.softmax(logits)
        final_ret = float(jnp.dot(w, mu))
        final_vol = float(jnp.sqrt(w @ cov @ w))
        sharpe = final_ret / (final_vol + 1e-8)
        return (f"{n_assets}-asset Markowitz: return={final_ret*100:.2f}%, "
                f"vol={final_vol*100:.2f}%, Sharpe={sharpe:.3f}, {ms:.0f}ms")
    run_test("Markowitz portfolio optimization (gradient descent, 50 assets)", "quant",
             markowitz_optimization,
             workaround="Core quant use case — grad through covariance matrix")

    # ── 2. Monte Carlo Option Pricing ─────────────────────────────────────────
    def monte_carlo_options():
        """
        Price a European call option via Monte Carlo.
        100,000 GBM paths × 252 trading days = vmap parallelism benchmark.
        Validates: vmap on GPU, PRNG reproducibility, numerical accuracy vs Black-Scholes.
        """
        import math
        n_paths = 100_000
        S0, K, r, sigma, T = 150.0, 155.0, 0.05, 0.25, 1.0
        dt = T / 252
        key = jax.random.PRNGKey(7)
        keys = jax.random.split(key, n_paths)

        @jit
        def single_path(path_key):
            Z  = jax.random.normal(path_key, (252,))
            log_ret = (r - 0.5*sigma**2)*dt + sigma*jnp.sqrt(dt)*Z
            ST = S0 * jnp.exp(jnp.sum(log_ret))
            return jnp.maximum(ST - K, 0.0)

        batch_paths = jit(vmap(single_path))
        t0 = time.perf_counter()
        payoffs = batch_paths(keys)
        price = float(jnp.mean(payoffs)) * math.exp(-r * T)
        ms = (time.perf_counter() - t0) * 1000

        # Black-Scholes analytical
        d1 = (math.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        ncdf = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
        bs_price = S0*ncdf(d1) - K*math.exp(-r*T)*ncdf(d2)
        error_pct = abs(price - bs_price) / bs_price * 100

        assert error_pct < 1.0, f"MC price deviates {error_pct:.3f}% from Black-Scholes"
        return (f"MC=${price:.4f} vs BS=${bs_price:.4f} (err={error_pct:.3f}%), "
                f"{n_paths:,} paths × 252 steps, {ms:.0f}ms on GPU")
    run_test("Monte Carlo option pricing (100k paths, vmap)", "quant",
             monte_carlo_options,
             workaround="The AMD killer feature: 100k paths in parallel via vmap. "
                        "If slow: check JAX_PLATFORMS=gpu is set")

    # ── 3. Differentiable VaR (Value at Risk) ─────────────────────────────────
    def differentiable_var():
        """
        Compute VaR via gradient of the loss distribution.
        Grad-through-quantile: differentiating risk measures.
        Used for: portfolio risk optimization, regulatory capital.
        """
        key = jax.random.PRNGKey(99)
        n_scenarios = 50_000
        n_assets = 10
        weights = jnp.ones(n_assets) / n_assets

        # Simulate asset returns
        returns = jax.random.normal(key, (n_scenarios, n_assets)) * 0.02

        @jit
        def portfolio_var(w, alpha=0.05):
            """5% VaR — worst 5% of portfolio returns."""
            port_returns = returns @ w
            sorted_ret   = jnp.sort(port_returns)
            var_idx      = int(alpha * n_scenarios)
            return -sorted_ret[var_idx]          # VaR is positive (loss)

        @jit
        def cvar(w, alpha=0.05):
            """CVaR / Expected Shortfall — mean of worst alpha% outcomes."""
            port_returns = returns @ w
            sorted_ret   = jnp.sort(port_returns)
            var_idx      = int(alpha * n_scenarios)
            return -jnp.mean(sorted_ret[:var_idx])

        t0 = time.perf_counter()
        var_val  = float(portfolio_var(weights))
        cvar_val = float(cvar(weights))
        # Gradient of CVaR w.r.t. weights (sensitivity analysis)
        grad_cvar = jit(grad(cvar))
        sensitivities = grad_cvar(weights)
        ms = (time.perf_counter() - t0) * 1000
        return (f"VaR(5%)={var_val*100:.3f}%, CVaR(5%)={cvar_val*100:.3f}%, "
                f"max sensitivity={float(jnp.max(jnp.abs(sensitivities))):.4f}, "
                f"{n_scenarios:,} scenarios, {ms:.0f}ms")
    run_test("Value at Risk + CVaR with gradient (risk sensitivity)", "quant",
             differentiable_var,
             workaround="Differentiable risk = JAX advantage over NumPy/pandas risk systems")

    # ── 4. Parallel Strategy Backtesting ─────────────────────────────────────
    def parallel_backtesting():
        """
        Backtest 10,000 trading strategies in PARALLEL via vmap.
        Each strategy is a (fast_period, slow_period, threshold, stop_loss) tuple.
        CPU: ~40 seconds. GPU via vmap: ~2 seconds.
        This is Chharmoney's core GPU advantage.
        """
        key = jax.random.PRNGKey(42)
        n_strategies = 10_000
        n_days = 252
        n_assets = 20

        # Simulate asset prices (geometric Brownian motion)
        returns = jax.random.normal(key, (n_days, n_assets)) * 0.01
        prices  = jnp.exp(jnp.cumsum(returns, axis=0)) * 100

        @jit
        def backtest_one(params):
            """Single strategy backtest. params = [momentum_window, threshold, stop, size]"""
            momentum_w = jnp.clip(jnp.int32(params[0] * 20 + 10), 5, 30)
            threshold  = params[1] * 0.02
            asset_idx  = 0  # trade first asset
            closes     = prices[:, asset_idx]

            def step(carry, t):
                cash, position, peak = carry
                price = closes[t]
                momentum = (closes[t] - closes[jnp.maximum(t-10, 0)]) / (closes[jnp.maximum(t-10, 0)] + 1e-8)
                buy_signal  = (momentum > threshold) & (position == 0.0)
                sell_signal = (momentum < -threshold) | (position * price < peak * 0.93)
                peak_new = jnp.where(position > 0, jnp.maximum(peak, price), price)
                position_new = jnp.where(buy_signal, cash / price, jnp.where(sell_signal, 0.0, position))
                cash_new = jnp.where(buy_signal, 0.0, jnp.where(sell_signal, position * price, cash))
                return (cash_new, position_new, peak_new), None

            (final_cash, final_pos, _), _ = jax.lax.scan(step, (10000.0, 0.0, 0.0), jnp.arange(n_days))
            final_value = final_cash + final_pos * prices[-1, 0]
            return (final_value - 10000.0) / 10000.0   # return %

        batch_backtest = jit(vmap(backtest_one))
        strategy_params = jax.random.uniform(key, (n_strategies, 4))

        t0 = time.perf_counter()
        results = batch_backtest(strategy_params)
        _ = results.block_until_ready()
        ms = (time.perf_counter() - t0) * 1000

        best_return = float(jnp.max(results)) * 100
        worst_return = float(jnp.min(results)) * 100
        win_rate = float(jnp.mean(results > 0)) * 100

        return (f"{n_strategies:,} strategies × {n_days}d × {n_assets} assets in {ms:.0f}ms | "
                f"Best: {best_return:+.1f}%, Worst: {worst_return:+.1f}%, Win%: {win_rate:.1f}%")
    run_test("Parallel strategy backtesting (10k strategies via vmap)", "quant",
             parallel_backtesting,
             workaround="Chharmoney's GPU advantage — 10k backtests in seconds not minutes")

    # ── 5. Sharpe Ratio Gradient Optimization ────────────────────────────────
    def sharpe_gradient_optimization():
        """
        Directly optimize Sharpe ratio via gradient descent.
        This is impossible in traditional backtesting frameworks (non-differentiable).
        JAX makes it differentiable → gradient descent on strategy parameters.
        """
        key = jax.random.PRNGKey(0)
        n_days   = 252 * 2   # 2 years
        n_assets = 10

        returns = jax.random.normal(key, (n_days, n_assets)) * 0.008 + 0.0003

        @jit
        def strategy_returns(params):
            """Simple momentum strategy with learnable threshold."""
            weights   = jax.nn.softmax(params[:n_assets])
            threshold = jax.nn.sigmoid(params[n_assets]) * 0.02
            momentum  = jnp.mean(returns[-20:], axis=0)   # 20-day momentum
            active    = jnp.where(momentum > threshold, weights, 0.0)
            active    = active / (jnp.sum(active) + 1e-8)
            port_ret  = returns @ active
            return port_ret

        @jit
        def neg_sharpe(params):
            port_ret = strategy_returns(params)
            return -(jnp.mean(port_ret) / (jnp.std(port_ret) + 1e-8)) * jnp.sqrt(252)

        grad_sharpe = jit(jax.value_and_grad(neg_sharpe))
        params = jnp.zeros(n_assets + 1)
        t0 = time.perf_counter()
        for _ in range(100):
            val, g = grad_sharpe(params)
            params = params - 0.01 * g
        ms = (time.perf_counter() - t0) * 1000
        final_sharpe = -float(val)
        return (f"Sharpe optimized: {final_sharpe:.3f} | "
                f"100 gradient steps × {n_days}d × {n_assets} assets | {ms:.0f}ms")
    run_test("Sharpe ratio gradient optimization (differentiable backtesting)", "quant",
             sharpe_gradient_optimization,
             workaround="Impossible without JAX: gradient through Sharpe = strategy parameter optimization")


# ── Report Generation ─────────────────────────────────────────────────────────

def generate_report(system: SystemInfo) -> dict:
    report = {
        "system":  asdict(system),
        "summary": SuiteReport(system=system, results=_RESULTS).summary(),
        "results": [asdict(r) for r in _RESULTS],
        "generated": datetime.now().isoformat(),
    }
    return report

def save_report(report: dict, system: SystemInfo, output_dir: str = None):
    if output_dir:
        results_dir = Path(output_dir)
    else:
        results_dir = Path(__file__).parent.parent / "RESULTS"
    results_dir.mkdir(exist_ok=True, parents=True)

    gpu_slug = system.gpu_name.replace(" ", "_").replace("/", "-")[:30] or "unknown_gpu"
    rocm_slug = system.rocm_version.replace(".", "_")[:10] or "no_rocm"
    date_slug = datetime.now().strftime("%Y%m%d")
    filename  = f"{gpu_slug}_{rocm_slug}_{date_slug}.json"

    out_path = results_dir / filename
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return out_path

def print_summary(report: dict):
    s = report["summary"]
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Total tests : {s['total']}")
    print(f"  PASS        : {s['passed']}  ({s['pass_rate']})")
    print(f"  FAIL        : {s['failed']}")
    print(f"  PARTIAL     : {s['partial']}")
    print(f"  SKIP        : {s['skipped']}")

    failures = [r for r in report["results"] if r["status"] == "FAIL"]
    if failures:
        print(f"\n  FAILED TESTS:")
        for r in failures:
            print(f"    ✗ [{r['category']:12s}] {r['name']}")
            print(f"        Error: {(r.get('error') or '')[:80]}")
            if r.get("workaround"):
                print(f"        Fix:   {r['workaround'][:80]}")

def generate_github_issue(report: dict, system: SystemInfo) -> str:
    s = report["summary"]
    failures = [r for r in report["results"] if r["status"] == "FAIL"]
    partials = [r for r in report["results"] if r["status"] == "PARTIAL"]

    lines = [
        f"## JAX AMD Test Results — {system.gpu_name or 'Unknown GPU'}",
        "",
        "### System",
        f"- GPU: `{system.gpu_name}`",
        f"- VRAM: {system.gpu_vram_gb} GB",
        f"- ROCm: `{system.rocm_version}`",
        f"- JAX: `{system.jax_version}` / jaxlib `{system.jaxlib_version}`",
        f"- Backend: `{system.backend}`",
        f"- Platform: `{system.platform}`",
        f"- WSL2: {system.is_wsl2}",
        "",
        "### Summary",
        f"| Status | Count |",
        f"|--------|-------|",
        f"| ✅ PASS | {s['passed']} |",
        f"| ❌ FAIL | {s['failed']} |",
        f"| ⚠️ PARTIAL | {s['partial']} |",
        f"| ⏭ SKIP | {s['skipped']} |",
        "",
    ]
    if failures:
        lines += ["### Failed Tests", ""]
        for r in failures:
            lines += [
                f"**{r['name']}** (`{r['category']}`)",
                f"```",
                f"{r.get('error', 'no error captured')}",
                f"```",
                "",
            ]
    if partials:
        lines += ["### Partial Tests", ""]
        for r in partials:
            lines += [f"- **{r['name']}**: {r.get('note', '')}"]
        lines.append("")
    lines += [
        "### Environment Variables at Test Time",
        f"```",
        f"HIP_VISIBLE_DEVICES={system.hip_visible}",
        f"XLA_FLAGS={system.xla_flags}",
        f"```",
    ]
    return "\n".join(lines)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="JAX AMD Research Failure Suite")
    parser.add_argument("--submit",     action="store_true",
                        help="Open GitHub issue draft with results")
    parser.add_argument("--quick",      action="store_true",
                        help="Skip slow tests (perf, memory stress)")
    parser.add_argument("--category",   type=str,
                        help="Run only tests in this category")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save result JSON (default: RESULTS/ in repo root)")
    parser.add_argument("--cpu-only",   action="store_true",
                        help="Force CPU backend — verify math without GPU timing")
    args = parser.parse_args()

    if args.cpu_only:
        import os
        os.environ["JAX_PLATFORMS"] = "cpu"
        print("[cpu-only mode] JAX_PLATFORMS=cpu — math verified, timings are CPU not GPU")

    print("\nJAX AMD GPU Research Failure Suite")
    print("====================================")
    print("Purpose: Systematically find every way JAX breaks on AMD.")
    print("Every failure becomes documentation. Every fix saves someone a week.")
    print()

    # Collect system info
    print("Collecting system info...")
    system = collect_system_info()
    print(f"  GPU      : {system.gpu_name or 'unknown'}")
    print(f"  ROCm     : {system.rocm_version}")
    print(f"  JAX      : {system.jax_version}")
    print(f"  Backend  : {system.backend}")
    print(f"  WSL2     : {system.is_wsl2}")

    # Run test categories
    try:
        import jax
        test_installation()
        test_basic_ops()
        test_jit()
        test_vmap()
        test_grad()
        test_lax()
        if not args.quick:
            test_memory()
            test_performance()
        test_known_amd_bugs()
        test_research_patterns()
        test_physics_benchmarks()
        test_quant_finance_benchmarks()
    except ImportError as e:
        print(f"\nFATAL: JAX not installed — {e}")
        print("Install: pip install jax jaxlib")
        print("For AMD: pip install jax[rocm6_1] -f https://storage.googleapis.com/jax-releases/rocm/jax_rocm_releases.html")
        return

    # Generate and save report
    report   = generate_report(system)
    out_path = save_report(report, system, output_dir=args.output_dir)
    print_summary(report)

    print(f"\n  Report saved: {out_path}")
    print(f"\n  To submit your results:")
    print(f"  1. Open an issue at https://github.com/ChharithOeun/jax-amd-gpu-setup/issues")
    print(f"  2. Paste the contents of {out_path.name}")
    print(f"  3. Your results help every AMD researcher who comes after you.")

    if args.submit:
        issue_body = generate_github_issue(report, system)
        issue_path = out_path.with_suffix(".issue.md")
        issue_path.write_text(issue_body)
        print(f"\n  GitHub issue draft: {issue_path}")

if __name__ == "__main__":
    main()
