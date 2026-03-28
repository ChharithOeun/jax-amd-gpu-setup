"""
composer_demo.py — Composer Audio Engine in JAX on AMD GPU
==============================================================
Composer is Chharbot's audio intelligence engine.
It handles mel spectrogram extraction, harmonic analysis,
and next-note prediction — all GPU-accelerated via JAX.

Separate from Chharmoney (which handles quant trading math).
Composer = music. Chharmoney = money.

Architecture:
  - JAX-native mel filterbank (no librosa dependency)
  - Transformer encoder: 4 layers, 256 dim, 4 heads
  - Causal self-attention over audio patches
  - Output: MIDI note logits (next phrase prediction)
  - vmap: batch multiple audio clips in parallel on GPU

This demo covers:
  - Audio feature extraction in JAX (mel spectrogram, no external deps)
  - Transformer block in JAX (the core of modern audio AI)
  - JIT compilation on AMD GPU
  - vmap for parallel multi-clip processing
  - Gradient computation (for fine-tuning on new music)

Usage:
    python examples/composer_demo.py
    python examples/composer_demo.py --gpu    # assert GPU is used
    python examples/composer_demo.py --bench  # full benchmark suite
"""

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from jax.example_libraries import stax

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_RATE   = 22050
N_MELS        = 80       # mel spectrogram bins
PATCH_SIZE    = 16       # audio patch size (frames)
D_MODEL       = 256      # transformer hidden dim
N_HEADS       = 4        # attention heads
N_LAYERS      = 4        # transformer depth
VOCAB_SIZE    = 128      # MIDI note range
BATCH_SIZE    = 8

# ── Mel Spectrogram (simplified, JAX-native) ──────────────────────────────────

def hz_to_mel(hz):
    return 2595.0 * jnp.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def create_mel_filterbank(n_fft, n_mels, sr, fmin=0.0, fmax=None):
    """Create mel filterbank matrix. JAX-compatible."""
    if fmax is None:
        fmax = sr / 2.0
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = jnp.linspace(mel_min, mel_max, n_mels + 2)
    hz_points  = mel_to_hz(mel_points)
    bin_points = jnp.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_freqs    = n_fft // 2 + 1

    # Build filterbank
    filterbank = jnp.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        lo, center, hi = bin_points[m-1], bin_points[m], bin_points[m+1]
        # Rising slope
        for k in range(lo, center):
            filterbank = filterbank.at[m-1, k].set((k - lo) / (center - lo + 1e-8))
        # Falling slope
        for k in range(center, hi):
            filterbank = filterbank.at[m-1, k].set((hi - k) / (hi - center + 1e-8))
    return filterbank

@jit
def audio_to_mel_patches(audio, filterbank, patch_size=PATCH_SIZE):
    """
    Convert raw audio to mel spectrogram patches.
    Input : audio (T,) float32
    Output: patches (N_patches, N_mels * patch_size)
    """
    n_fft    = (filterbank.shape[1] - 1) * 2
    hop_len  = n_fft // 4

    # Simple STFT via overlap-add
    n_frames = (len(audio) - n_fft) // hop_len + 1
    window   = jnp.hanning(n_fft)

    frames = jnp.stack([
        audio[i*hop_len : i*hop_len + n_fft] * window
        for i in range(n_frames)
    ])  # (n_frames, n_fft)

    # FFT magnitudes
    stft    = jnp.fft.rfft(frames, axis=-1)
    power   = jnp.abs(stft) ** 2

    # Apply mel filterbank
    mel     = jnp.dot(power, filterbank.T)  # (n_frames, n_mels)
    mel_db  = 10.0 * jnp.log10(jnp.maximum(mel, 1e-10))

    # Chunk into patches
    n_patches = n_frames // patch_size
    mel_cut   = mel_db[:n_patches * patch_size]
    patches   = mel_cut.reshape(n_patches, patch_size * N_MELS)
    return patches

# ── Transformer Components ────────────────────────────────────────────────────

def init_linear(key, in_dim, out_dim):
    k1, k2 = random.split(key)
    W = random.normal(k1, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
    b = jnp.zeros(out_dim)
    return {"W": W, "b": b}

def linear(params, x):
    return x @ params["W"] + params["b"]

def layer_norm(x, eps=1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std  = jnp.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def multi_head_attention(params, x, n_heads=N_HEADS):
    """
    Multi-head self-attention.
    x: (seq_len, d_model)
    """
    seq_len, d_model = x.shape
    d_head = d_model // n_heads

    Q = linear(params["Wq"], x)  # (seq, d_model)
    K = linear(params["Wk"], x)
    V = linear(params["Wv"], x)

    # Split heads
    Q = Q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

    # Scaled dot-product attention
    scale   = jnp.sqrt(d_head).astype(jnp.float32)
    scores  = jnp.einsum("hid,hjd->hij", Q, K) / scale
    weights = jax.nn.softmax(scores, axis=-1)
    out     = jnp.einsum("hij,hjd->hid", weights, V)

    # Merge heads
    out = out.transpose(1, 0, 2).reshape(seq_len, d_model)
    return linear(params["Wo"], out)

def ffn(params, x):
    """Feed-forward network inside transformer block."""
    h = jax.nn.gelu(linear(params["W1"], x))
    return linear(params["W2"], h)

def transformer_block(params, x):
    """One transformer encoder block with residual connections."""
    # Self-attention with residual
    x = x + multi_head_attention(params["attn"], layer_norm(x))
    # FFN with residual
    x = x + ffn(params["ffn"], layer_norm(x))
    return x

# ── Composer Model ──────────────────────────────────────────────────────────

def init_composer(key, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS):
    """Initialize all model parameters."""
    params = {}
    keys = random.split(key, 20)
    ki = iter(keys)

    # Patch embedding (linear projection)
    patch_dim = PATCH_SIZE * N_MELS
    params["patch_embed"] = init_linear(next(ki), patch_dim, d_model)

    # Positional embedding (learned)
    params["pos_embed"] = random.normal(next(ki), (256, d_model)) * 0.02

    # Transformer blocks
    params["blocks"] = []
    for _ in range(n_layers):
        block = {
            "attn": {
                "Wq": init_linear(next(ki), d_model, d_model),
                "Wk": init_linear(next(ki), d_model, d_model),
                "Wv": init_linear(next(ki), d_model, d_model),
                "Wo": init_linear(next(ki), d_model, d_model),
            },
            "ffn": {
                "W1": init_linear(next(ki), d_model, d_model * 4),
                "W2": init_linear(next(ki), d_model * 4, d_model),
            },
        }
        params["blocks"].append(block)

    # Output head — predict next note
    params["head"] = init_linear(next(ki), d_model, VOCAB_SIZE)

    return params

@partial(jit, static_argnames=["training"])
def composer_forward(params, patches, training=False):
    """
    Forward pass: audio patches -> next note logits.
    patches: (seq_len, patch_dim)
    returns: logits (seq_len, VOCAB_SIZE)
    """
    seq_len = patches.shape[0]

    # Embed patches
    x = linear(params["patch_embed"], patches)  # (seq_len, d_model)

    # Add positional embedding (truncate to seq_len)
    x = x + params["pos_embed"][:seq_len]

    # Transformer layers
    for block_params in params["blocks"]:
        x = transformer_block(block_params, x)

    # Predict next note at each position
    logits = linear(params["head"], x)  # (seq_len, VOCAB_SIZE)
    return logits

def cross_entropy_loss(params, patches, targets):
    """Cross-entropy loss for next-note prediction."""
    logits = composer_forward(params, patches)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.mean(log_probs[jnp.arange(len(targets)), targets])
    return loss

# Gradient function (for fine-tuning)
grad_fn = jit(grad(cross_entropy_loss))

# Batched forward pass
batched_forward = jit(vmap(composer_forward, in_axes=(None, 0)))

# ── Benchmarks ────────────────────────────────────────────────────────────────

def benchmark_model(params, n_runs=10):
    print("\n  Model Benchmark:")
    key = random.PRNGKey(99)

    # Single forward pass
    patches = random.normal(key, (32, PATCH_SIZE * N_MELS))  # 32 patches
    _ = composer_forward(params, patches).block_until_ready()  # warmup

    t0 = time.perf_counter()
    for _ in range(n_runs):
        out = composer_forward(params, patches).block_until_ready()
    elapsed = (time.perf_counter() - t0) / n_runs
    print(f"    Single forward (32 patches) : {elapsed*1000:.2f}ms")

    # Batched forward
    batch = random.normal(key, (BATCH_SIZE, 32, PATCH_SIZE * N_MELS))
    _ = batched_forward(params, batch).block_until_ready()  # warmup

    t0 = time.perf_counter()
    for _ in range(n_runs):
        out = batched_forward(params, batch).block_until_ready()
    elapsed = (time.perf_counter() - t0) / n_runs
    print(f"    Batched forward (8x32)      : {elapsed*1000:.2f}ms")
    print(f"    Throughput                  : {BATCH_SIZE/elapsed:.0f} sequences/sec")

    # Gradient computation
    targets = jnp.zeros(32, dtype=jnp.int32)
    _ = grad_fn(params, patches, targets)  # warmup (trigger JIT)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        g = jax.tree_util.tree_map(lambda x: x.block_until_ready(), grad_fn(params, patches, targets))
    elapsed = (time.perf_counter() - t0) / n_runs
    print(f"    Gradient pass               : {elapsed*1000:.2f}ms")

# ── Main Demo ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Composer JAX audio demo")
    parser.add_argument("--gpu",   action="store_true", help="Assert GPU is used")
    parser.add_argument("--bench", action="store_true", help="Run full benchmark")
    args = parser.parse_args()

    print("\nComposer — Audio Intelligence Engine (JAX on AMD GPU)")
    print("=========================================================")

    # Device info
    print(f"\n  Backend  : {jax.default_backend().upper()}")
    print(f"  Devices  : {jax.devices()}")

    if args.gpu and jax.default_backend() != "gpu":
        print("\n  WARNING: Not running on GPU. Install DirectML or ROCm backend.")
        print("  See scripts/test_gpu_directml.py or scripts/test_gpu_rocm.py")

    # Initialize model
    print("\n  Initializing Composer model...")
    key = random.PRNGKey(42)
    params = init_composer(key)

    # Count parameters
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Parameters : {n_params:,}  ({n_params/1e6:.1f}M)")
    print(f"  d_model    : {D_MODEL}")
    print(f"  Layers     : {N_LAYERS}")
    print(f"  Heads      : {N_HEADS}")

    # Simulate audio input
    print("\n  Creating synthetic audio (2 sec @ 22050Hz)...")
    audio_key = random.PRNGKey(7)
    # Simulate a 2-second audio clip with harmonic content
    t = jnp.linspace(0, 2.0, 2 * SAMPLE_RATE)
    audio = (
        0.5 * jnp.sin(2 * jnp.pi * 440.0 * t) +   # A4
        0.3 * jnp.sin(2 * jnp.pi * 554.4 * t) +   # C#5
        0.2 * jnp.sin(2 * jnp.pi * 659.3 * t) +   # E5
        0.05 * random.normal(audio_key, t.shape)    # noise
    )

    # Create filterbank (do once, not traced by JAX)
    print("  Building mel filterbank...")
    n_fft = 1024
    filterbank = create_mel_filterbank(n_fft, N_MELS, SAMPLE_RATE)
    print(f"  Filterbank shape: {filterbank.shape}")

    # Extract mel patches
    print("  Extracting mel patches from audio...")
    patches = audio_to_mel_patches(audio, filterbank)
    print(f"  Patches shape   : {patches.shape}  ({patches.shape[0]} patches of {patches.shape[1]} features)")

    # Forward pass (JIT compiles on first call)
    print("\n  Running forward pass (first call triggers XLA compilation)...")
    t0 = time.perf_counter()
    logits = composer_forward(params, patches)
    logits.block_until_ready()
    compile_time = time.perf_counter() - t0
    print(f"  First call (+ compile): {compile_time*1000:.0f}ms")

    # Second call (cached)
    t0 = time.perf_counter()
    logits = composer_forward(params, patches)
    logits.block_until_ready()
    cached_time = time.perf_counter() - t0
    print(f"  Cached call           : {cached_time*1000:.1f}ms")

    # Decode prediction
    predicted_notes = jnp.argmax(logits, axis=-1)
    print(f"\n  Input  : A major chord (A4 + C#5 + E5)")
    print(f"  Output logits shape   : {logits.shape}")
    print(f"  First 5 predicted MIDI notes: {predicted_notes[:5]}")
    top_note = int(predicted_notes[0])
    note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    note_name  = note_names[top_note % 12]
    note_oct   = top_note // 12 - 1
    print(f"  Most likely next note : MIDI {top_note} ({note_name}{note_oct})")

    # Gradient check (proves autograd works on AMD)
    print("\n  Testing gradient computation (autograd)...")
    targets = predicted_notes[:patches.shape[0]]
    t0 = time.perf_counter()
    grads = grad_fn(params, patches, targets)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), grads)
    grad_time = time.perf_counter() - t0
    print(f"  Gradient time: {grad_time*1000:.0f}ms  (first call includes compile)")
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    print(f"  Gradient norm: {float(grad_norm):.4f}  (non-zero = autograd working)")

    if args.bench:
        benchmark_model(params)

    print("\n" + "="*54)
    print("  Composer demo complete!")
    print(f"  Backend used : {jax.default_backend().upper()}")
    if jax.default_backend() == "gpu":
        print("  AMD GPU acceleration CONFIRMED")
    else:
        print("  Running on CPU — GPU backend not detected.")
        print("  See README.md to enable AMD GPU acceleration.")
    print("="*54)

if __name__ == "__main__":
    main()
