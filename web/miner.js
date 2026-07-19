// miner.js — WebGPU host for the tap_vanity.wgsl kernel (fast/rawtr mode).
// Mirrors the Rust wgpu host (wgpu/src/lib.rs): same bind group layout,
// buffer structs, and dispatch order (init_table -> setup -> search_fast loop).
//
// WebKit/iOS resilience:
//   * Each search launch is sized ADAPTIVELY toward a wall-time budget
//     (~80ms) instead of a fixed desktop size. A giant single dispatch trips
//     WebKit's GPU watchdog, which loses the device and rejects every pending
//     mapAsync with "map async was not successful" — the exact iOS bug.
//   * requestDevice asks for no exotic limits; the 256-wide workgroup is
//     verified against adapter limits up front.
//   * device.lost + uncapturederror are wired to diagnostics; a lost device is
//     recovered by Miner.mine() by recreating at half the workgroup count.

"use strict";

const CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

function patternTo5bit(s) {
  const out = [];
  for (const ch of s) {
    const i = CHARSET.indexOf(ch);
    if (i < 0) throw new Error(`invalid bech32 char '${ch}' (valid: ${CHARSET}; lowercase, no 1/b/i/o)`);
    out.push(i);
  }
  return out;
}

class Miner {
  // cfg: { prefix:[5bit], suffix:[5bit], threadgroups, wgslUrl,
  //        frameBudgetMs?, startIters?, maxIters?, onDiag? }
  static async create(cfg) {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter available");

    // The kernel is hard-coded to @workgroup_size(256). Every spec-compliant
    // device guarantees 256, but check so we fail loud instead of at pipeline
    // creation if some device reports lower.
    const lim = adapter.limits;
    if (lim.maxComputeInvocationsPerWorkgroup < 256 || lim.maxComputeWorkgroupSizeX < 256) {
      throw new Error("this GPU can't run a 256-wide compute kernel (limits too low)");
    }

    // No exotic limits requested — defaults are plenty (largest buffer is a
    // few MB) and asking for more can fail on WebKit.
    const device = await adapter.requestDevice({ label: "tapvanity" });

    const m = new Miner();
    m.device = device;
    m._lost = null;
    m._onDiag = cfg.onDiag || (() => {});

    // Surface GPU trouble instead of letting it vanish into a rejected promise.
    device.lost.then((info) => {
      m._lost = info;
      m._onDiag(`device lost: ${info.reason || "unknown"} ${info.message || ""}`.trim());
    });
    try {
      device.addEventListener("uncapturederror", (ev) => {
        m._onDiag(`gpu error: ${ev.error.message}`);
        // eslint-disable-next-line no-console
        console.error("[tapvanity] uncaptured GPU error:", ev.error);
      });
    } catch (_) { /* older impls: no addEventListener on device */ }

    const maxDim = lim.maxComputeWorkgroupsPerDimension || 65535;
    m.threadgroups = Math.max(1, Math.min(cfg.threadgroups || 64, maxDim));
    m.totalThreads = m.threadgroups * 256;
    m.frameBudgetMs = cfg.frameBudgetMs || 80;
    m.maxIters = cfg.maxIters || 1024;
    m._iters = Math.max(1, cfg.startIters || 2); // iters for the NEXT launch
    m.iters = m._iters;                          // iters used by the LAST launch
    m._iterBase = 0;
    m.prefixLen = cfg.prefix.length;
    m.suffixLen = cfg.suffix.length;
    m.adapterInfo = adapter.info
      ? `${adapter.info.vendor || ""} ${adapter.info.architecture || ""} ${adapter.info.description || ""}`.trim()
      : "unknown";

    const wgslSrc = await (await fetch(cfg.wgslUrl || "tap_vanity.wgsl")).text();
    const module = device.createShaderModule({ code: wgslSrc });

    // k0: random 256-bit, reduced mod n
    const rnd = new Uint8Array(32);
    crypto.getRandomValues(rnd);
    m.k0 = SECP.mod(SECP.bytesToBig(rnd), SECP.N);

    // ----- buffers (layout mirrors lib.rs) -----
    const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
    m.tableBuf = device.createBuffer({ label: "table", size: 32 * 256 * 16 * 4, usage: S });
    m.stateBuf = device.createBuffer({ label: "state", size: m.totalThreads * 24 * 4, usage: S });

    // Cfg: prefix_len, suffix_len, fast, total_threads, prefix[32], suffix[32]
    const cfgArr = new Uint32Array(68);
    cfgArr[0] = m.prefixLen;
    cfgArr[1] = m.suffixLen;
    cfgArr[2] = 1; // fast
    cfgArr[3] = m.totalThreads;
    cfg.prefix.forEach((v, i) => (cfgArr[4 + i] = v));
    cfg.suffix.forEach((v, i) => (cfgArr[36 + i] = v));
    m.cfgBuf = device.createBuffer({ label: "cfg", size: cfgArr.byteLength, usage: S | CD });
    device.queue.writeBuffer(m.cfgBuf, 0, cfgArr);

    // Params: k0[8] (LE u32 limbs), iter_base, iters, _p0, _p1
    const params = new Uint32Array(12);
    let k = m.k0;
    for (let i = 0; i < 8; i++) { params[i] = Number(k & 0xffffffffn); k >>= 32n; }
    params[8] = 0;         // iter_base
    params[9] = m._iters;  // iters
    m.paramsBuf = device.createBuffer({ label: "params", size: params.byteLength, usage: S | CD });
    device.queue.writeBuffer(m.paramsBuf, 0, params);
    m._paramScratch = new Uint32Array(2); // [iter_base, iters], written at offset 32

    // Found: flag, tid, iter, parity, tweak[8], qx[8] = 20 u32
    m.FOUND_SIZE = 20 * 4;
    m.foundBuf = device.createBuffer({ label: "found", size: m.FOUND_SIZE, usage: S | CD | CS });
    m.foundRead = device.createBuffer({ label: "found_read", size: m.FOUND_SIZE, usage: GPUBufferUsage.MAP_READ | CD });

    // ----- bind group: 5 storage buffers, group(0) -----
    const bgl = device.createBindGroupLayout({
      entries: [0, 1, 2, 3, 4].map((i) => ({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: (i === 1 || i === 4) ? "read-only-storage" : "storage" },
      })),
    });
    m.bindGroup = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: m.tableBuf } },
        { binding: 1, resource: { buffer: m.cfgBuf } },
        { binding: 2, resource: { buffer: m.foundBuf } },
        { binding: 3, resource: { buffer: m.stateBuf } },
        { binding: 4, resource: { buffer: m.paramsBuf } },
      ],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const mk = (entryPoint) => device.createComputePipeline({ label: entryPoint, layout, compute: { module, entryPoint } });
    m.initPipeline = mk("init_table");
    m.setupPipeline = mk("setup");
    m.searchPipeline = mk("search_fast");
    return m;
  }

  dispatch(pipeline, groups) {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(groups);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  // Build window table and per-thread start points. Call once.
  async prepare() {
    this.dispatch(this.initPipeline, (32 * 256) / 256); // 32 workgroups
    await this.device.queue.onSubmittedWorkDone();
    this.dispatch(this.setupPipeline, this.threadgroups);
    await this.device.queue.onSubmittedWorkDone();
  }

  // One adaptive search launch. Measures wall time and re-sizes the NEXT
  // launch toward frameBudgetMs so no single dispatch runs long enough to trip
  // a GPU watchdog. Returns { hit, keys } where keys is this launch's count.
  async searchStep() {
    const iters = this._iters;
    this._paramScratch[0] = this._iterBase >>> 0;
    this._paramScratch[1] = iters >>> 0;
    this.device.queue.writeBuffer(this.paramsBuf, 32, this._paramScratch);

    const t = performance.now();
    this.dispatch(this.searchPipeline, this.threadgroups);
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.foundBuf, 0, this.foundRead, 0, this.FOUND_SIZE);
    this.device.queue.submit([enc.finish()]);
    try {
      await this.foundRead.mapAsync(GPUMapMode.READ);
    } catch (e) {
      // WebKit rejects with "map async was not successful" when the device was
      // lost mid-dispatch (watchdog). Tag it so the driver can recover smaller.
      const err = new Error(
        `GPU dispatch failed (device lost / watchdog): ${e && e.message ? e.message : e}`,
      );
      err.deviceLost = true;
      throw err;
    }
    const dt = performance.now() - t;
    const found = new Uint32Array(this.foundRead.getMappedRange().slice(0));
    this.foundRead.unmap();

    const keys = this.totalThreads * iters;
    this._iterBase += iters;
    this.iters = iters; // record for keysPerLaunch()/display

    // Adapt for next launch: nudge toward the budget, clamped to avoid wild
    // swings and to keep any one dispatch bounded.
    if (dt > 0) {
      const scale = Math.max(0.5, Math.min(2, this.frameBudgetMs / dt));
      this._iters = Math.max(1, Math.min(this.maxIters, Math.round(iters * scale) || 1));
    }

    if (found[0] === 0) {
      // iter lives in one 32-bit lane per thread; keep it well inside range.
      if (this._iterBase >= 0x7fff0000) {
        const err = new Error("search space exhausted for this session — start again");
        err.exhausted = true;
        throw err;
      }
      return { hit: null, keys };
    }
    return { hit: this.reconstruct(found), keys };
  }

  // Back-compat single-launch API (fixed iters); prefer searchStep()/mine().
  async searchOnce(iterBase) {
    if (typeof iterBase === "number") this._iterBase = iterBase >>> 0;
    const { hit } = await this.searchStep();
    return hit;
  }

  reconstruct(found) {
    const tid = BigInt(found[1]);
    const iter = BigInt(found[2]);
    const kRaw = SECP.mod(this.k0 + (tid << 32n) + iter, SECP.N);
    const pub = SECP.pubkey(kRaw);
    const k = pub.parityOdd ? SECP.mod(SECP.N - kRaw, SECP.N) : kRaw;
    let gpuX = 0n;
    for (let i = 7; i >= 0; i--) gpuX = (gpuX << 32n) | BigInt(found[12 + i]);
    return { key: k, gpuX, tid: found[1], iter: found[2] };
  }

  keysPerLaunch() { return this.totalThreads * this.iters; }

  destroy() {
    try { this.device.destroy(); } catch (_) { /* ignore */ }
  }

  // High-level driver: create + prepare + adaptive search loop, with automatic
  // recovery from a lost device by rebuilding at a smaller workgroup count.
  // opts: { prefix?, suffix, wgslUrl, threadgroups?, frameBudgetMs?,
  //         onReady?({adapter,threadgroups}), onProgress?({tried,rate,elapsed,iters}),
  //         onDiag?(msg), shouldStop?() }
  // Returns { found:{address,wif,descriptor}, tried, rate, elapsed } or
  //         { stopped:true }.
  static async mine(opts) {
    const prefix5 = patternTo5bit(opts.prefix || "");
    const suffix5 = patternTo5bit(opts.suffix || "");
    let threadgroups = opts.threadgroups || 64;
    const t0 = performance.now();
    let totalTried = 0;
    const diag = opts.onDiag || (() => {});

    for (let attempt = 0; ; attempt++) {
      let miner = null;
      try {
        miner = await Miner.create({
          prefix: prefix5,
          suffix: suffix5,
          threadgroups,
          wgslUrl: opts.wgslUrl,
          frameBudgetMs: opts.frameBudgetMs,
          onDiag: diag,
        });
        opts.onReady?.({ adapter: miner.adapterInfo, threadgroups });
        await miner.prepare();

        for (;;) {
          if (opts.shouldStop?.()) return { stopped: true };
          const { hit, keys } = await miner.searchStep();
          totalTried += keys;
          const elapsed = (performance.now() - t0) / 1000;
          const rate = elapsed > 0 ? totalTried / elapsed : 0;
          opts.onProgress?.({ tried: totalTried, rate, elapsed, iters: miner.iters });
          if (hit) {
            const res = verifyHit(hit, opts.prefix || "", opts.suffix || "");
            const el = (performance.now() - t0) / 1000;
            return { found: res, tried: totalTried, rate: el > 0 ? totalTried / el : 0, elapsed: el };
          }
        }
      } catch (e) {
        const msg = String((e && e.message) || e);
        const recoverable =
          (e && e.deviceLost) || /device.*lost|map async|destroyed/i.test(msg);
        if (recoverable && threadgroups > 8 && attempt < 5) {
          threadgroups = Math.max(8, Math.floor(threadgroups / 2));
          diag(`GPU hiccup — retrying smaller (${threadgroups} workgroups)`);
          await new Promise((r) => setTimeout(r, 250));
          continue;
        }
        throw e;
      } finally {
        try { miner?.destroy(); } catch (_) { /* ignore */ }
      }
    }
  }
}

// Verify a reconstructed key against the pattern; returns full result or throws.
function verifyHit(hit, prefixStr, suffixStr) {
  const pub = SECP.pubkey(hit.key); // must be even-y now
  if (pub.parityOdd) throw new Error("verification failed: normalized key still has odd-y pubkey");
  if (pub.xBig !== hit.gpuX) throw new Error("verification failed: CPU pubkey.x != GPU-reported x (browser WGSL compiler bug?)");
  const address = SECP.bech32mP2TR(pub.x);
  const body = address.slice(4); // after "bc1p"
  if (prefixStr && !body.startsWith(prefixStr)) throw new Error(`verification failed: address ${address} does not start with bc1p${prefixStr}`);
  if (suffixStr && !address.endsWith(suffixStr)) throw new Error(`verification failed: address ${address} does not end with ${suffixStr}`);
  const wif = SECP.wif(hit.key);
  return { address, wif, descriptor: `rawtr(${wif})` };
}
