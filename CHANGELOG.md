# Changelog

All notable changes to **jax-amd-gpu-setup** are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Auto-updated by `.github/workflows/changelog.yml` on every push to `main`.

> **Verification policy:** Every claim in this repo is backed by runnable code.
> Numbers marked ✅ are computed by the actual algorithm. Numbers marked 🔬 require
> running `python scripts/research_failure_suite.py` on real AMD hardware.

---

## [Unreleased]

_Next planned:_
- Real measured results from RX 5700 XT hardware run (replace projected JSON)
- RDNA2 / RDNA3 community result submissions

---

## [0.4.0] — 2026-03-27

### Fixed
- **YAML bug** — `issue-responder.yml` was invalid YAML (template literal content
  at column 0 terminated the block scalar). Rewrote all multi-line responses as
  `.join('\n')` arrays. Validated with `python3 -c "import yaml; yaml.safe_load(...)"`.
- **Fabricated Monte Carlo result** — RESULTS JSON claimed MC=$2.49 vs BS=$2.49
  for params S0=150/K=155/sigma=25%/T=1yr. Actual Black-Scholes: $16.109.
  MC 100k paths (NumPy verified): $16.067, error=0.261%. ✅
- **Fabricated Schrödinger error** — Claimed 0.004%. NumPy simulation of the
  identical n=512 finite-difference code gives E₀=0.500961, error=0.192%. ✅
- **Fabricated GPU timings** — All timing fields in RESULTS JSON set to `null`
  pending a real hardware run. Added `_note` field marking the JSON as PROJECTED.
- **VaR/CVaR values verified** — VaR(5%)=1.047%, CVaR(5%)=1.306% via NumPy
  simulation of 50k scenarios, 10 equal-weight assets, σ=2%. ✅

### Changed
- RESULTS JSON `_note` field makes projected vs measured status explicit
- RESULTS/README.md entry now marked "PROJECTED" with verified math outputs inline

---

## [0.3.0] — 2026-03-27

### Added
- `examples/composer_demo.py` — Composer audio engine JAX demo (properly named;
  separate from Chharmoney trading demo)
- `RESULTS/AMD_Radeon_RX_5700_XT_rocm6_1_20260327.json` — first community result
  file (RDNA1 baseline, now correctly marked PROJECTED)
- `RESULTS/README.md` — GPU compatibility matrix, architecture reference table,
  per-section breakdown of what works/fails/is slow
- `.github/workflows/issue-responder.yml` — auto-responds to opened issues,
  pattern-matches 7 known AMD error types, auto-labels
- `.github/ISSUE_TEMPLATE/bug-report.md` — structured AMD JAX bug report template
- `.github/ISSUE_TEMPLATE/test-results.md` — results submission template

### Fixed
- `examples/composer_demo.py` — all internal references renamed from Chharmoney
  to Composer (function names, docstring, argparse, print statements)
- `docs/jax-ecosystem.md` — removed internal "Why AMD specifically needs this"
  justification note; replaced with public AMD Compatibility Status table

---

## [0.2.0] — 2026-03-27

### Added
- `scripts/research_failure_suite.py` — physics benchmarks:
  - `schrodinger_1d()` — 1D harmonic oscillator via eigvalsh, validates E₀=0.5ℏω (error ✅ 0.192% at n=512)
  - `wave_equation_pde()` — CFL-stable lax.scan time integration, 256×2000 grid ✅
  - `variational_monte_carlo()` — hydrogen atom grad(VMC_energy), target E=−0.5 Ha 🔬
  - `molecular_dynamics_minimization()` — 64-atom Lennard-Jones, 100 gradient steps 🔬
- `scripts/research_failure_suite.py` — quant finance benchmarks:
  - `markowitz_optimization()` — 50-asset Sharpe gradient descent 🔬
  - `monte_carlo_options()` — 100k GBM paths via vmap, BS validation (✅ err=0.261% NumPy verified)
  - `differentiable_var()` — VaR + CVaR with grad sensitivity (✅ VaR≈1.0%, CVaR≈1.3%)
  - `parallel_backtesting()` — 10k strategies × 252 days × 20 assets via vmap 🔬
  - `sharpe_gradient_optimization()` — 100 gradient steps on Sharpe ratio 🔬

---

## [0.1.0] — 2026-03-27

### Added
- `docs/jax-gotchas.md` — 9 JAX gotchas with AMD-specific failure modes
  - Quick reference table: error → root cause → fix
  - AMD-specific: bfloat16 NaN fix, MIOpen cache, DirectML silent CPU fallback
- `docs/jax-ecosystem.md` — JAX ecosystem map for AMD users
  - Flax, Haiku, Equinox, Optax, DiffrRax, Jaxopt with AMD compatibility ratings
  - 8-step learning path: NumPy → grad → jit → vmap → scan → library → failure suite
  - AMD Compatibility Status table (DirectML vs ROCm per-library)
- `examples/chharmoney_demo.py` — JAX quant trading engine demo (AMD GPU)
- `scripts/research_failure_suite.py` — systematic AMD JAX failure documentation
  - 12 test categories, bfloat16 NaN documented (workaround included)
  - RDNA1 facts: first JIT ~30-90s, MIOpen cache issues
- `README.md` — initial setup guide

---

_Legend:_ ✅ = verified by running actual code (NumPy or JAX). 🔬 = requires real AMD GPU hardware run.
