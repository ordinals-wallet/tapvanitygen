# tapvanity

GPU **Bitcoin taproot (P2TR / `bc1p…`) vanity address miners**. One repo, three
targets that share the same verified secp256k1 + BIP-341 + bech32m pipeline:

| Target | Path | Runs on | Speed (M3 Max) |
|---|---|---|---|
| **Metal** (fastest) | [`metal/`](metal/) | macOS, Apple Silicon | ~15 MK/s standard, ~180 MK/s fast |
| **wgpu** (portable) | [`wgpu/`](wgpu/) | Windows / Linux / macOS, any Vulkan/DX12/Metal GPU | ~1.5 MK/s standard, ~4 MK/s fast |
| **Browser** (zero-install) | [`web/`](web/) | Chrome/Edge (WebGPU), fast mode only | ~6 MK/s (Chrome, M3 Max) |

All three mine mainnet **P2TR key-path** addresses (BIP-341, no script tree):
output key `Q = P + TapTweak(P.x)·G`, address `= bech32m("bc", 1, Q.x)`. Every
found key is verified against the `bitcoin` crate (native targets) or an
in-browser secp256k1 (web) before it is shown.

## Which one should I use?

- **Mac, want maximum speed** → `metal/` (hand-tuned Swift + Metal).
- **Windows / Linux / AMD / Intel / NVIDIA, or you just want a binary** → `wgpu/`.
- **No install, try it in a tab** → `web/` (WebGPU).

## Two mining modes (native targets)

- **Standard** (default): mines the BIP-341 **internal** key. The result is a
  normal `tr(KEY)` taproot key that any BIP-386 wallet can import.
- **Fast / rawtr** (`--fast`): mines the **output** key directly (no TapTweak,
  no per-candidate `t·G`) — much faster. The result corresponds to a Bitcoin
  Core `rawtr(KEY)` descriptor, **not** a `tr()` descriptor. See the
  [rawtr caveat](#rawtr-caveat).

The web target is **fast/rawtr only** (simplest and fastest for a browser).

## Quickstart

### Metal (macOS, Apple Silicon)

```
cd metal
make                       # builds build/tapvanity_metal
./build/tapvanity_metal --prefix cafe            # standard
./build/tapvanity_metal --fast --suffix xyz      # fast / rawtr
make test                  # GPU self-test + end-to-end mine + rust cross-check
```

### wgpu (Windows / Linux / macOS)

```
cd wgpu
cargo run --release -- --prefix cafe             # standard
cargo run --release -- --fast --suffix xyz       # fast / rawtr
cargo run --release -- --self-test               # GPU known-vector check
cargo test --release                             # end-to-end, both modes
```

Prebuilt binaries for windows-x86_64, linux-x86_64, and macos-aarch64 are
attached to each tagged GitHub Release (see `.github/workflows/release.yml`).

- **Windows**: run the `.exe`; uses DX12 or Vulkan (NVIDIA/AMD/Intel).
- **Linux**: needs a Vulkan driver (`vulkan-loader` + your GPU's ICD).
- **macOS**: uses the Metal backend (the native `metal/` build is faster).

### Browser

Open `web/index.html` (served over http — WebGPU needs a secure context or
`localhost`). Requires a WebGPU-capable browser (Chrome/Edge 113+; Safari
Technology Preview; Firefox behind a flag). See [`web/`](web/) and the
[Browser](#browser) section.

## Patterns and charset

`--prefix <pat>` matches right after `bc1p`; `--suffix <pat>` matches the end
(may reach into the 6 checksum chars); they can be combined. Matching is exact.

The bech32 charset is **32 characters**: `qpzry9x8gf2tvdw0s3jn54khce6mua7l`.
It **excludes `1`, `b`, `i`, `o`**, and addresses are **lowercase-only**.
Each pattern character after `bc1p` multiplies difficulty by 32:

| pattern len | expected keys `32^n` | metal fast @180 MK/s | wgpu fast @4 MK/s |
|---:|---:|---:|---:|
| 1 | 32 | instant | instant |
| 2 | 1,024 | instant | instant |
| 3 | 32,768 | instant | instant |
| 4 | 1.05 M | instant | 0.3 s |
| 5 | 33.6 M | 0.2 s | 8 s |
| 6 | 1.07 G | 6 s | 4.5 min |
| 7 | 34.4 G | 3.2 min | 2.4 h |
| 8 | 1.10 T | 1.7 h | 3.2 days |
| 9 | 35.2 T | 2.3 days | 100 days |
| 10 | 1.13 P | 72 days | 9 years |

(Mean of a geometric distribution; 50% at 0.69×, 90% at 2.3× the mean.) Note:
the 52nd data char (last before the checksum) carries only 1 bit of key
material, so patterns reaching that position are more constrained than `32^n`.

## rawtr caveat

Fast mode (and the entire web target) produces a key for a Bitcoin Core
**`rawtr(KEY)`** descriptor — the witness program is the key's x-only pubkey
with **no** BIP-341 tweak applied. This means:

- Import it into Bitcoin Core (≥ 24) as `rawtr(<WIF>)`. The tools print the
  descriptor line for you.
- **Most wallets and BIP-386 `tr()` descriptors expect an *internal* key** and
  apply the TapTweak themselves — they will **not** derive this address from
  this key. If you need a `tr()`-compatible key, use **standard** mode.
- Funds are equally spendable/secure either way; it is purely an import-format
  (descriptor) difference.

## Security notes

- **Your private key is the address.** The tools print the private key (hex +
  WIF) and, in the browser, show it on screen. Anyone who sees it controls any
  funds sent to the address. Copy it somewhere safe, don't screen-share it,
  don't paste it into websites.
- **Keys are generated locally.** Nothing is transmitted anywhere — the native
  binaries make no network calls, and the web page makes no runtime network
  requests (everything is vendored). k0 randomness comes from the OS CSPRNG
  (`SecRandomCopyBytes` / `getrandom` / `crypto.getRandomValues`).
- **Verify before funding.** Every target re-derives the address from the found
  key with an independent implementation and refuses to show unverified
  results. You can also independently check any printed key/address with your
  own wallet before sending funds.
- These miners search a random keyspace; they cannot target someone else's
  existing address.

## Correctness

Each target runs a **GPU self-test against the `k=1` known vector**
(`bc1pmfr3p9j00pfxjh0zmgp99y8zftmd3s5pmedqhyptwy6lm87hf5sspknck9`) at startup
and refuses to mine if it fails. Native targets then cross-check every find
against the `bitcoin` crate; the web target cross-checks against a vendored
BigInt secp256k1. `metal/` additionally has `make test` (end-to-end mine +
Rust reference), and `wgpu/` has `cargo test` (end-to-end, both modes, both
parities). See each subdirectory's README for details.

## Browser

The `web/` target is a framework-free static page (one HTML + one JS host + the
shared WGSL + a vendored BigInt secp256k1) that runs the same `search_fast`
kernel via the browser's native WebGPU API. It mines fast/rawtr only, verifies
every find in-browser (recomputes the pubkey → x-only → bech32m from the found
key and only displays matches) before display, and shows live keys/sec.

Measured **~6 MK/s in Chrome on an M3 Max** — slower than the native binaries
(browser GPU compute + the Tint WGSL compiler), but zero-install. Actual speed
depends on the browser and GPU. WebGPU support: Chrome/Edge 113+ (stable),
Safari Technology Preview, Firefox behind `dom.webgpu.enabled`. The page shows
a clear "WebGPU not supported" message with guidance when unavailable, and a
prominent "keys are generated locally and never transmitted" notice. Because it
is a single self-contained page with no runtime network calls, it drops cleanly
into any static host.

## Repository layout

```
tapvanity/
├── metal/     Swift + Metal miner (macOS, fastest) + Rust reference verifier
├── wgpu/      Rust + wgpu miner (cross-platform) — WGSL kernel, in-process verify
├── web/       Browser WebGPU miner (same WGSL, JS host, vendored secp256k1)
├── .github/workflows/   ci.yml (tests), release.yml (tagged binary builds)
└── LICENSE    MIT
```

## License

MIT — see [LICENSE](LICENSE).
