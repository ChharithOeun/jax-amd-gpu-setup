"""
test_jax_install.py — Step 1: Verify JAX is installed and working (CPU)
Run this first before attempting any GPU setup.

Usage:
    python scripts/test_jax_install.py
"""

import sys

def banner(text, char="="):
    line = char * 54
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")

def check_python():
    banner("Python Version")
    v = sys.version_info
    print(f"  Python {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        print("  WARNING: JAX requires Python 3.9+")
    else:
        print("  OK")

def check_jax():
    banner("JAX Installation")
    try:
        import jax
        print(f"  JAX version      : {jax.__version__}")
        import jaxlib
        print(f"  jaxlib version   : {jaxlib.__version__}")
    except ImportError as e:
        print(f"  NOT INSTALLED: {e}")
        print("\n  Install with:")
        print("    pip install jax jaxlib")
        return False
    return True

def check_backends():
    banner("Available Backends")
    try:
        import jax
        backends = []
        for backend in ["cpu", "gpu", "tpu"]:
            try:
                jax.devices(backend)
                backends.append(backend.upper())
            except Exception:
                pass
        print(f"  Active backends  : {', '.join(backends) if backends else 'None detected'}")
        print(f"  Default backend  : {jax.default_backend().upper()}")
        print(f"  All devices      :")
        for d in jax.devices():
            print(f"    - {d}")
    except Exception as e:
        print(f"  Error: {e}")

def check_gpu_devices():
    banner("GPU Device Details")
    try:
        import jax
        gpu_devices = jax.devices("gpu")
        if gpu_devices:
            print(f"  GPU devices found: {len(gpu_devices)}")
            for d in gpu_devices:
                print(f"    {d}")
        else:
            print("  No GPU devices found via JAX.")
            print("  This is expected if you only have CPU installed.")
            print("  See docs/setup-directml.md or docs/setup-wsl2-rocm.md")
    except Exception:
        print("  No GPU backend available (CPU-only install is fine for now).")

def run_basic_computation():
    banner("Basic JAX Computation (CPU)")
    try:
        import jax
        import jax.numpy as jnp

        # Simple array op
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        y = jnp.dot(x, x)
        print(f"  dot([1,2,3,4], [1,2,3,4]) = {float(y):.1f}  (expected 30.0)")

        # JIT compile
        @jax.jit
        def relu(x):
            return jnp.maximum(0, x)

        z = relu(jnp.array([-1.0, 0.5, 2.0]))
        print(f"  relu([-1, 0.5, 2]) = {z}  (expected [0.  0.5 2. ])")

        # Gradient
        def loss(x):
            return jnp.sum(x ** 2)

        grad = jax.grad(loss)(jnp.array([1.0, 2.0, 3.0]))
        print(f"  grad(sum(x^2)) at [1,2,3] = {grad}  (expected [2. 4. 6.])")

        print("\n  All CPU computations passed!")
    except Exception as e:
        print(f"  FAILED: {e}")

def main():
    print("\nJAX AMD GPU Setup — Install Verification")
    print("==========================================")
    check_python()
    ok = check_jax()
    if ok:
        check_backends()
        check_gpu_devices()
        run_basic_computation()
    banner("Next Steps")
    print("  1. CPU working? -> try scripts/test_gpu_directml.py")
    print("  2. On Linux/WSL2? -> try scripts/test_gpu_rocm.py")
    print("  3. See README.md for full setup guide")

if __name__ == "__main__":
    main()
