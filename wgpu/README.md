# tapvanity-wgpu

Cross-platform GPU taproot (`bc1p…`) vanity miner in Rust + [wgpu](https://wgpu.rs)
(WGSL compute). Runs on Windows (DX12/Vulkan), Linux (Vulkan), and macOS
(Metal) — NVIDIA / AMD / Intel / Apple GPUs. The WGSL kernel in
`src/tap_vanity.wgsl` is a port of the verified Metal shader in
[`../metal/shaders/tap_vanity.metal`](../metal/shaders/tap_vanity.metal).

## Usage

```
cargo run --release -- --prefix cafe          # standard (internal key, tr())
cargo run --release -- --fast --suffix xyz    # fast / rawtr (output key)
cargo run --release -- --prefix ab --suffix yz   # combined
cargo run --release -- --prefix cafe --estimate  # difficulty/ETA, then exit
cargo run --release -- --self-test               # GPU known-vector check
```

Knobs: `--threadgroups N` (default 256), `--iters N` (per thread per launch,
default 64). Larger values amortize the per-launch readback and raise
throughput at the cost of coarser stop granularity.

## Modes

- **standard** — mines the BIP-341 internal key; result is a `tr()` key.
- **`--fast`** — mines the output key directly (rawtr descriptor); ~3× faster.
  See the top-level README's "rawtr caveat".

Every find is verified in-process against the `bitcoin` crate before it is
printed (address re-derived from the recovered key; the program exits non-zero
on any mismatch). Startup runs a GPU self-test against the `k=1` known vector
and refuses to mine on failure.

## Tests

```
cargo test --release        # GPU self-test + end-to-end mines (both modes,
                            # both parities), all cross-checked in-process
```

On headless CI runners without a GPU, set `TAPVANITY_SKIP_GPU=1` to soft-skip
the GPU tests (the pure-CPU scalar test still runs).

## Performance

Measured on an Apple M3 Max (Metal backend), sustained:

- standard: ~1.5 MK/s
- fast/rawtr: ~4 MK/s

This is well below the hand-tuned native `metal/` miner (~15 / ~180 MK/s) — the
wgpu host reads the result buffer back synchronously after every launch, which
is the dominant cost, and naga's generated code is less optimized than the
hand-written Metal. On discrete Windows/Linux GPUs the relative numbers differ;
raise `--iters` to push throughput. The goal of this target is portability and
easy distribution, not peak speed.

## Implementation notes

- WGSL has no 64-bit integers, so field elements are 8×`u32` limbs and widening
  multiplies are emulated with 16-bit half-splits (`mul32` → `(lo, hi)`).
- The **fast** path is a separate entry point (`search_fast`) from **standard**
  (`search`). The naga/MSL backend miscompiled a shared implementation
  (passing a struct-member array to the big-return `bech32m_values` yielded
  zeroed high limbs, causing spurious matches); isolating the two paths into
  separate kernels fixed it. The fast kernel reports no parity — the host
  recovers the y-parity from the reconstructed key via the `bitcoin` crate.
- `atomicOr`-based claim (not `atomicCompareExchange`, which the Metal backend
  did not implement in the wgpu version used).
