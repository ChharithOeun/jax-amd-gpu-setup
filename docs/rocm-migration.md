# ROCm 5.x to 6.x Migration Guide

ROCm 6.x is NOT backwards compatible. If you updated ROCm and JAX broke, you're in the right place.

## What Changed in ROCm 6.x

| Component | ROCm 5.7 | ROCm 6.0+ |
|-----------|----------|-----------|
| MIOpen API | v2 | v3 (breaking) |
| Library paths | `/opt/rocm-5.x/` | `/opt/rocm/` (symlink) |
| HIP runtime | 5.7 | 6.0 |
| Python wheels | `jax[rocm]` 0.4.x | `jax[rocm6_1]` 0.4.25+ |
| Driver requirement | AMDGPU-PRO 22.x | AMDGPU-PRO 23.x+ |

## Migration Steps

### 1. Remove old ROCm
```bash
sudo amdgpu-install --uninstall
sudo apt remove --purge rocm-* hip-* miopen-* rocsolver-*
sudo rm -rf /opt/rocm-5* /opt/rocm
```

### 2. Install ROCm 6.1
```bash
wget https://repo.radeon.com/amdgpu-install/6.1/ubuntu/focal/amdgpu-install_6.1.60100-1_all.deb
sudo dpkg -i amdgpu-install_6.1.60100-1_all.deb
sudo amdgpu-install --usecase=rocm --no-dkms
```

### 3. Reinstall JAX with matching version
```bash
pip uninstall jax jaxlib -y
pip install "jax[rocm6_1]" \
  -f https://storage.googleapis.com/jax-releases/rocm/jax_rocm_releases.html
```

### 4. Reset environment variables
```bash
# Remove old ROCm paths, add new
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/hip/lib
export PATH=/opt/rocm/bin:$PATH
export HIP_VISIBLE_DEVICES=0
```

## Common Post-Migration Errors

**"MIOpen: Invalid argument"**
```bash
export MIOPEN_USER_DB_PATH=/tmp/miopen-$(whoami)
mkdir -p $MIOPEN_USER_DB_PATH
# MIOpen rebuilds its kernel cache for ROCm 6.x
```

**"libamdhip64.so not found"**
```bash
sudo ldconfig /opt/rocm/lib
echo '/opt/rocm/lib' | sudo tee /etc/ld.so.conf.d/rocm.conf
sudo ldconfig
```

**"JAX was compiled with ROCm 5.x but found ROCm 6.x"**
```bash
# Must reinstall jaxlib — can't mix versions
pip install "jax[rocm6_1]" -f https://storage.googleapis.com/jax-releases/rocm/jax_rocm_releases.html
```

## Keeping Both Versions (Parallel Install)

If you need ROCm 5.7 for other tools alongside 6.x:

```bash
# Install 6.x to non-default path
sudo ROCM_PATH=/opt/rocm-6.1 amdgpu-install --usecase=rocm --no-dkms

# Switch between versions
alias use-rocm5='export PATH=/opt/rocm-5.7/bin:$PATH && export LD_LIBRARY_PATH=/opt/rocm-5.7/lib'
alias use-rocm6='export PATH=/opt/rocm-6.1/bin:$PATH && export LD_LIBRARY_PATH=/opt/rocm-6.1/lib'
```
