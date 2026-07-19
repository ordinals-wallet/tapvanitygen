# tapvanity-metal

Bitcoin mainnet **P2TR (taproot, `bc1p…`) vanity address miner** for Apple
Silicon GPUs (Metal). Swift host + Metal compute kernel, modeled on
`octra-vanity-metal`. Exact target semantics match the CPU reference miner
`tapvanitygen` (rust `bitcoin` crate): BIP-341 key-path with no script tree,
i.e. output key `Q = P + TapTweak(P.x)·G`, address = bech32m(`bc`, 1, `Q.x`).

## Build & test

```
make            # builds build/tapvanity_metal + build/default.metallib
make test       # GPU known-vector self-test, mines bc1pqq…, cross-checks
                # the found key against the rust reference (verify/)
```

`make test` requires cargo (builds `verify/` — a tiny rust bin using the
same `bitcoin` crate version as tapvanitygen).

## Usage

```
build/tapvanity_metal --prefix <pat>       # match bc1p<pat>…
build/tapvanity_metal --suffix <pat>       # match …<pat>  (checksum chars included)
build/tapvanity_metal --prefix <p> --suffix <s>    # combined (single check per key)
build/tapvanity_metal --fast --prefix <pat>        # FAST/rawtr mode (see below)
build/tapvanity_metal --prefix <pat> --estimate    # difficulty/ETA, then exit
build/tapvanity_metal --self-test                  # known-vector check, exit
```

Performance knobs: `--threadgroups N` (default 1024), `--threads N` (256),
`--iters N` (16, rounded to a multiple of 16). Defaults are near-optimal on
M3 Max (~15 MK/s standard, ~180 MK/s fast, sustained).

### FAST / rawtr mode (`--fast`)

Standard mode mines the BIP-341 **internal** key: per candidate it computes
the TapTweak tagged hash and a full `t·G` scalar multiplication. `--fast`
skips all of that by mining the **output** key directly: it walks
`Q_i = Q_0 + i·G` and matches `bech32m(Q_i.x)` — ~12× faster.

The trade-off: the found secret is the *output* key, so the address
corresponds to a Bitcoin Core `rawtr(KEY)` descriptor, **not** a standard
`tr(KEY)` descriptor. The miner prints the descriptor line ready for
`importdescriptors`. **Wallet-compat caveat:** most wallets (and BIP-386
`tr()` descriptors) expect an internal key and apply the TapTweak
themselves — they will NOT derive this address from this key. Only use
`--fast` output with software that supports `rawtr()` (Bitcoin Core ≥ 24).
Funds are exactly as spendable/secure either way; it's purely a descriptor
/ import-format difference. Note this key provably commits to no script
path only in the sense that none was generated; unlike a BIP-341 tweaked
key, third parties cannot tell it has no script commitment.

On a hit it prints the address, the **internal** private key (hex + WIF),
the TapTweak, and the **tweaked output secret** (the key that actually
signs key-path spends). If `verify/` is built, it automatically re-derives
the address with the rust reference and prints `VERIFY OK`.

Every startup runs a GPU self-test against the known vector `k = 1 →
bc1pmfr3p9j00pfxjh0zmgp99y8zftmd3s5pmedqhyptwy6lm87hf5sspknck9` and refuses
to run if it fails.

## How it works

- Host draws one random 256-bit base secret `k0` (SecRandomCopyBytes),
  reduced mod n. A setup kernel gives thread `i` the start point
  `P_i = (k0 + i·2^32)·G` via a precomputed 8-bit window table
  (`table[b][v] = v·256^b·G`, 32×256 affine points, built on-GPU at launch).
- Each search iteration steps `P += G` (mixed Jacobian+affine add — cheap).
- Per candidate: affine `P.x` + y-parity, TapTweak tagged hash (SHA-256
  midstate for the 64-byte tag prefix is baked in — one compression per
  candidate), `Q = P_even + t·G` (window table), bech32m encode of `Q.x`,
  pattern compare. The two field inversions per candidate (affine P, affine
  Q) are amortized with Montgomery batch inversion over batches of 8
  candidates.
- On a hit the kernel reports (thread id, global iteration, parity); host
  reconstructs `k = k0 + tid·2^32 + iter (mod n)`, negates mod n if the
  internal point had odd y, and computes the output secret `k + t mod n`.

Field arithmetic is secp256k1 (`p = 2^256 − 2^32 − 977`) on 8×32-bit limbs
with 64-bit accumulation and fold reduction (`2^256 ≡ 0x1000003D1`).
Inversion is the libsecp256k1 addition chain.

## Difficulty

The bech32 charset is 32 characters: `qpzry9x8gf2tvdw0s3jn54khce6mua7l`.
Note it **excludes `1`, `b`, `i`, `o`**, and bech32m addresses are
**lowercase-only**. Each pattern char after `bc1p` costs a factor of 32:

| pattern len | expected keys 32^n | standard @ 15 MK/s | fast @ 180 MK/s |
|---:|---:|---:|---:|
| 1 | 32 | instant | instant |
| 2 | 1,024 | instant | instant |
| 3 | 32,768 | instant | instant |
| 4 | 1.05 M | 0.07 s | instant |
| 5 | 33.6 M | 2.2 s | 0.2 s |
| 6 | 1.07 G | 72 s | 6 s |
| 7 | 34.4 G | 38 min | 3.2 min |
| 8 | 1.10 T | 20 h | 1.7 h |
| 9 | 35.2 T | 27 days | 2.3 days |
| 10 | 1.13 P | 2.4 years | 72 days |

(Mean of a geometric distribution; 50% at 0.69×, 90% at 2.3× the mean.)
Caveat: the 52nd data char after `bc1p` (the last one before the checksum)
carries only 1 bit of key material (`q` or `s`), so patterns reaching that
position are cheaper/more constrained than 32^n suggests.

## Correctness gates

- Startup GPU self-test against the k=1 known vector.
- `verify/` rust bin (same `bitcoin` crate as the reference miner) re-derives
  address + tweaked output secret from the printed internal key (or, in fast
  mode, the address from the output key via
  `TweakedPublicKey::dangerous_assume_tweaked` / `Address::p2tr_tweaked`);
  the miner runs it automatically after every find.
- `make test` mines live patterns end-to-end in both modes (prefix, suffix,
  combined) and fails on any mismatch of address or tweaked secret.

## Measured performance

Apple M3 Max, sustained (Montgomery batch inversion, batch=16):

- standard mode: **~15 MK/s** (~33× the tapvanitygen CPU baseline of
  ~0.46 MK/s) — dominated by the per-candidate t·G window multiply.
- fast/rawtr mode: **~180 MK/s** (~390× CPU baseline) — one mixed point
  add + amortized inversion per candidate.

## Limitations

- Prefix/suffix exact match only, combinable (no case-insensitive — taproot
  addresses are lowercase-only anyway; no "anywhere" or regex modes).
- Single found result per run (first hit wins, then exit).
- A dedicated squaring routine (faster than generic mul) is disabled: it
  passes isolated GPU A/B tests (including in-place and 100-deep chains)
  but miscompiles inside the inversion chain even with
  `__attribute__((noinline))`. See note in the shader.
- Iteration space per thread is capped at 2^32 (the stride); the miner
  exits and asks for a restart if it ever gets near that (only relevant for
  ~year-long runs).
- No GPU duty-cycle throttle / auto-tune (the octra miner has these); use
  `--threadgroups/--iters` to shrink launches if the machine needs to stay
  responsive.
