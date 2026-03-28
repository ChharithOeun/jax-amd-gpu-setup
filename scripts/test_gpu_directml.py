"""
test_gpu_directml.py — Windows AMD GPU via DirectML plugin
DirectML is the recommended path for AMD GPU on Windows (no WSL2 needed).

Requires:
    pip install jax-directml
    (or the Microsoft DirectML JAX plugin when available)

Usage:
    python scripts/test_gpu_directml.py
"""

import sys
import time

def banner(text, char="="):
    line = char * 54
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")

def check_directml_plugin():
    banner("DirectML Plugin Detection")
    # Method 1: jax_plugins style (JAX 0.4.14+)
    try:
        import jax_plugins
        print(f"  jax_plugins found: {jax_plugins.__version__}")
    except ImportError:
        print("  jax_plugins: not installed")

    # Method 2: direct jax-directml package
    try:
        import jax_directml_plugin
        print(f"  jax_directml_plugin: FOUND")
        return True
    except ImportError:
        pass

    # Method 3: Check if GPU backend is available (plugin may auto-register)
    try:
        import jax
        gpus = jax.devices("gpu")
        if gpus:
            print(f"  GPU backend active! Devices: {gpus}")
            return True
    except Exception:
        pass

    print("  DirectML plugin: NOT found")
    print()
    print("  Install options:")
    print("    pip install jax-directml")
    print("    # OR (if above fails):")
    print("    pip install jax[directml]")
    print()
    print("  If pip install fails, try building from source:")
    print("    https://github.com/microsoft/jax-directml")
    return False

def check_amd_gpu_info():
    banner("AMD GPU Info (Windows)")
    try:
        import subprocess
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM"],
            capture_output=True, text=True, timeout=5
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip() and "Name" not in l]
        for line in lines:
            print(f"  GPU: {line}")
    except Exception as e:
        print(f"  Could not query GPU info: {e}")
        print("  (Run this check on Windows, not WSL2)")

def run_gpu_benchmark():
    banner("GPU Benchmark (if GPU available)")
    try:
        import jax
        import jax.numpy as jnp
        import time

        backend = jax.default_backend()
        print(f"  Active backend: {backend.upper()}")

        if backend == "cpu":
            print("  Running on CPU — no GPU backend detected.")
            print("  Results below are CPU baseline for comparison.")

        # Matrix multiply benchmark
        sizes = [512, 1024, 2048]
        for n in sizes:
            key = jax.random.PRNGKey(0)
            A = jax.random.normal(key, (n, n))
            B = jax.random.normal(key, (n, n))

            # Warmup
            C = jnp.dot(A, B).block_until_ready()

            # Timed run
            t0 = time.perf_counter()
            for _ in range(5):
                C = jnp.dot(A, B).block_until_ready()
            elapsed = (time.perf_counter() - t0) / 5

            gflops = (2 * n**3) / elapsed / 1e9
            print(f"  {n}x{n} matmul: {elapsed*1000:.1f}ms  ({gflops:.1f} GFLOPS)")

    except Exception as e:
        print(f"  Benchmark error: {e}")

def main():
    print("\nJAX AMD GPU Setup — DirectML Backend Test")
    print("==========================================")
    print("  Target: AMD RX 5700 XT / RX 6000 / RX 7000 on Windows")
    print("  Method: Microsoft DirectML plugin for JAX")
    print()

    check_amd_gpu_info()
    found = check_directml_plugin()
    run_gpu_benchmark()

    banner("Summary")
    if found:
        print("  DirectML found — JAX can use your AMD GPU on Windows!")
        print("  Next: Run examples/chharmoney_demo.py")
    else:
        print("  DirectML not found. Options:")
        print("  A) Install jax-directml (Windows native, recommended)")
        print("  B) Use WSL2 + ROCm (see scripts/test_gpu_rocm.py)")
        print("  C) CPU fallback (slower, but everything still runs)")

if __name__ == "__main__":
    main()
