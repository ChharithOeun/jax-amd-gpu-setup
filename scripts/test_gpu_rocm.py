"""
test_gpu_rocm.py — AMD GPU via ROCm backend (Linux / WSL2)
This is the Linux-native path. Run inside WSL2 on Windows.

Requires:
    ROCm 6.x installed (see docs/setup-wsl2-rocm.md)
    pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/rocm/jax_rocm_releases.html

Usage (in WSL2):
    python scripts/test_gpu_rocm.py
"""

import os
import subprocess
import sys
import time

def banner(text, char="="):
    line = char * 54
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")

def check_environment():
    banner("Environment Check")
    # Are we in WSL2?
    is_wsl = os.path.exists("/proc/version") and "microsoft" in open("/proc/version").read().lower()
    is_linux = sys.platform == "linux"
    print(f"  Platform    : {sys.platform}")
    print(f"  WSL2        : {'YES' if is_wsl else 'NO'}")
    if not is_linux:
        print()
        print("  STOP: ROCm only works on Linux / WSL2.")
        print("  On Windows, use scripts/test_gpu_directml.py instead.")
        return False
    return True

def check_rocm():
    banner("ROCm Installation")
    # Check rocm-smi
    try:
        result = subprocess.run(["rocm-smi", "--showid", "--showname"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("  rocm-smi output:")
            for line in result.stdout.strip().splitlines():
                print(f"    {line}")
            return True
        else:
            print(f"  rocm-smi error: {result.stderr.strip()}")
    except FileNotFoundError:
        print("  rocm-smi: NOT FOUND")
    except Exception as e:
        print(f"  rocm-smi error: {e}")

    print()
    print("  ROCm is not installed. Install guide:")
    print("  Ubuntu/WSL2:")
    print("    wget https://repo.radeon.com/amdgpu-install/6.1/ubuntu/focal/amdgpu-install_6.1.60100-1_all.deb")
    print("    sudo dpkg -i amdgpu-install_*.deb")
    print("    sudo amdgpu-install --usecase=rocm --no-dkms")
    print("    sudo usermod -aG render,video $USER")
    return False

def check_rocm_version():
    banner("ROCm Version")
    version_file = "/opt/rocm/.info/version"
    if os.path.exists(version_file):
        version = open(version_file).read().strip()
        print(f"  ROCm version: {version}")
        major = int(version.split(".")[0])
        if major < 6:
            print("  WARNING: ROCm 6.x recommended. You have an older version.")
            print("  JAX ROCm wheels are built against ROCm 6.x.")
        else:
            print("  Version OK (6.x)")
    else:
        print("  Could not determine ROCm version.")

def check_jax_rocm():
    banner("JAX ROCm Backend")
    try:
        import jax
        import jaxlib
        print(f"  JAX version    : {jax.__version__}")
        print(f"  jaxlib version : {jaxlib.__version__}")

        # Check if GPU backend is available
        try:
            gpus = jax.devices("gpu")
            print(f"  GPU devices    : {len(gpus)}")
            for d in gpus:
                print(f"    {d}")
            return True
        except Exception as e:
            print(f"  GPU backend    : NOT available ({e})")
            print()
            print("  Install jax[rocm]:")
            print("    pip install --upgrade pip")
            print("    pip install jax[rocm] \\")
            print("      -f https://storage.googleapis.com/jax-releases/rocm/jax_rocm_releases.html")
            return False
    except ImportError:
        print("  JAX not installed.")
        print("  pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/rocm/jax_rocm_releases.html")
        return False

def check_hip_visible_devices():
    banner("HIP / GPU Visibility")
    hip_visible = os.environ.get("HIP_VISIBLE_DEVICES", "not set")
    rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES", "not set")
    print(f"  HIP_VISIBLE_DEVICES  : {hip_visible}")
    print(f"  ROCR_VISIBLE_DEVICES : {rocr_visible}")
    print()
    print("  If JAX can't find your GPU, try:")
    print("    export HIP_VISIBLE_DEVICES=0")
    print("    export ROCR_VISIBLE_DEVICES=0")
    print("    export JAX_PLATFORMS=gpu")

def run_gpu_test():
    banner("GPU Computation Test")
    try:
        import jax
        import jax.numpy as jnp

        backend = jax.default_backend()
        print(f"  Backend: {backend.upper()}")

        # Simple compute on GPU
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (1024, 1024))

        t0 = time.perf_counter()
        result = jnp.dot(x, x.T).block_until_ready()
        elapsed = time.perf_counter() - t0

        print(f"  1024x1024 matmul : {elapsed*1000:.1f}ms")
        print(f"  Result shape     : {result.shape}")
        print(f"  Device           : {result.devices()}")
        print()
        if backend == "gpu":
            print("  GPU acceleration confirmed!")
        else:
            print("  Still on CPU. Check HIP_VISIBLE_DEVICES env var.")
    except Exception as e:
        print(f"  Error: {e}")

def known_issues():
    banner("Known Issues (RX 5700 XT on ROCm)")
    issues = [
        ("NaN in outputs", "Set XLA_FLAGS='--xla_gpu_enable_triton_gemm=false'"),
        ("MIOpen not found", "export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH"),
        ("GPU not visible in WSL2", "Add user to 'render' group: sudo usermod -aG render $USER"),
        ("jaxlib version mismatch", "Match jaxlib to ROCm version exactly: pip install jax[rocm6.1]"),
        ("Slow first run", "Normal — XLA compiles kernels on first call. JIT caches after."),
        ("Out of memory", "Use jax.device_put() explicitly; avoid implicit copies"),
    ]
    for problem, fix in issues:
        print(f"  Problem : {problem}")
        print(f"  Fix     : {fix}")
        print()

def main():
    print("\nJAX AMD GPU Setup — ROCm Backend Test (Linux/WSL2)")
    print("====================================================")

    env_ok = check_environment()
    if not env_ok:
        return

    check_rocm_version()
    check_rocm()
    check_hip_visible_devices()
    check_jax_rocm()
    run_gpu_test()
    known_issues()

if __name__ == "__main__":
    main()
