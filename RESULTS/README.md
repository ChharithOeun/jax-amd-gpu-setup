# Community Test Results

Each file here is a real test run from a real AMD GPU.
This is the compatibility matrix no one else has built.

## How to Submit Your Results

```bash
python scripts/research_failure_suite.py
```

Then open an issue at:
https://github.com/ChharithOeun/jax-amd-gpu-setup/issues

Paste your result JSON or use `--submit` to auto-generate a draft.

## Results by GPU

| GPU | ROCm | JAX | Pass Rate | Contributor |
|-----|------|-----|-----------|-------------|
| RX 5700 XT | TBD | TBD | TBD | @ChharithOeun |

*(Run the suite and add your row)*

## What We Learn From Failures

Every FAIL in the suite becomes a row in the troubleshooting guide.
Every workaround that works gets added to docs/troubleshooting.md.
The failures ARE the product.
