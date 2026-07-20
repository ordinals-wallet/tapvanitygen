// miner.js — WebGPU host for tap_vanity.wgsl (standard + fast modes).
// Mirrors the Rust wgpu host layout; adds WebKit/iOS resilience, runtime-
// selectable optimizations, per-tier self-test gating, and structured logging.
//
// Resilience model (three suspect WGSL compilers: Tint/Chrome, naga/Firefox,
// WebKit/Safari — the standard path has miscompiled on all three at times):
//   * Adaptive launch sizing toward a wall-time budget so no dispatch trips a
//     GPU watchdog (device-lost → mapAsync rejects).
//   * Two kernel TIERS: "fast" (search_opt = batched inversion, optional
//     dedicated fe_sqr) and "safe" (search = unbatched, fe_mul squaring, the
//     shape proven on every compiler). Each optimization is individually
//     disableable; safe = all off.
//   * Every tier is gated by a k=1 known-vector self-test BEFORE real mining,
//     and every find is re-verified; a failure at either point escalates to
//     the next (safer) tier with a friendly message.
//   * Structured [cheekyminer] logger records everything needed to diagnose a
//     field failure from a pasted log — NEVER private-key material.

"use strict";

const CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
const KNOWN_K1_ADDR =
  "bc1pmfr3p9j00pfxjh0zmgp99y8zftmd3s5pmedqhyptwy6lm87hf5sspknck9";
// (legacy) batched-kernel alignment; the batched kernel was reverted.
const OPT_BATCH = 4;
// 16-bit window table: 16 windows x 65536 entries x 16 u32 = 64 MB.
const WIDE_TABLE_BYTES = 16 * 65536 * 16 * 4;

// ------------------------------------------------------------- structured logger
// A ring buffer + console mirror. Key material is never passed in, by design,
// so dump() is always safe to paste publicly.
const CheekyLog = (() => {
  const buf = [];
  const MAX = 600;
  function line(level, event, data) {
    const ts = (performance.now() / 1000).toFixed(2);
    let s = `[cheekyminer] ${ts}s ${level} ${event}`;
    if (data !== undefined) {
      s += " " + (typeof data === "string" ? data : safeJson(data));
    }
    buf.push(s);
    if (buf.length > MAX) buf.shift();
    const fn = level === "ERR" ? console.error : level === "WARN" ? console.warn : console.log;
    try { fn(s); } catch (_) { /* ignore */ }
    return s;
  }
  function safeJson(o) {
    try { return JSON.stringify(o); } catch (_) { return String(o); }
  }
  return {
    info: (e, d) => line("INFO", e, d),
    warn: (e, d) => line("WARN", e, d),
    err: (e, d) => line("ERR", e, d),
    dump: () => buf.join("\n"),
    clear: () => { buf.length = 0; },
  };
})();

function detectCompiler() {
  const ua = (typeof navigator !== "undefined" && navigator.userAgent) || "";
  if (/Firefox\//.test(ua)) return "naga (Firefox)";
  if (/Edg\//.test(ua)) return "Tint (Edge)";
  if (/Chrome\/|Chromium\//.test(ua)) return "Tint (Chrome)";
  if (/Safari\//.test(ua) && /Version\//.test(ua)) return "WebKit (Safari)";
  if (/AppleWebKit\//.test(ua)) return "WebKit";
  return "unknown";
}

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
  // cfg: { prefix:[5bit], suffix:[5bit], mode, kernel, fastSqr, threadgroups,
  //        wgslUrl, frameBudgetMs?, startIters?, maxIters?, k0?, log? }
  static async create(cfg) {
    const log = cfg.log || CheekyLog;
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter available");

    const lim = adapter.limits;
    if (lim.maxComputeInvocationsPerWorkgroup < 256 || lim.maxComputeWorkgroupSizeX < 256) {
      throw new Error("this GPU can't run a 256-wide compute kernel (limits too low)");
    }
    const device = await adapter.requestDevice({ label: "tapvanity" });

    const m = new Miner();
    m.device = device;
    m._lost = null;
    m.log = log;
    m._onDiag = cfg.onDiag || (() => {});

    device.lost.then((info) => {
      m._lost = info;
      log.err("device-lost-event", { reason: info.reason || "unknown", message: (info.message || "").slice(0, 100) });
      m._onDiag(`device lost: ${info.reason || "unknown"}`.trim());
    });
    try {
      device.addEventListener("uncapturederror", (ev) => {
        log.err("uncaptured-error", { message: String(ev.error && ev.error.message).slice(0, 140) });
        m._onDiag(`gpu error: ${ev.error.message}`);
      });
    } catch (_) { /* older impls */ }

    m.mode = cfg.mode === "fast" ? "fast" : "standard";
    m.kernel = cfg.kernel || (m.mode === "fast" ? "search_fast" : "search");
    m.fastSqr = !!cfg.fastSqr;
    m.wide = m.kernel === "search_wide";
    if (m.wide && lim.maxStorageBufferBindingSize < WIDE_TABLE_BYTES) {
      device.destroy();
      const e = new Error(`device can't bind the 64MB wide table (maxStorageBufferBindingSize=${lim.maxStorageBufferBindingSize})`);
      e.wideUnsupported = true;
      throw e;
    }
    m.batchSize = m.kernel === "search_opt" ? OPT_BATCH : 1;
    m._batchAlign = m.batchSize;

    const maxDim = lim.maxComputeWorkgroupsPerDimension || 65535;
    m.threadgroups = Math.max(1, Math.min(cfg.threadgroups || 64, maxDim));
    m.totalThreads = m.threadgroups * 256;
    m.frameBudgetMs = cfg.frameBudgetMs || 80;
    m.maxIters = cfg.maxIters || 1024;
    m._iters = Math.max(m._batchAlign, Math.round((cfg.startIters || m._batchAlign) / m._batchAlign) * m._batchAlign);
    m.iters = m._iters;
    m._iterBase = 0;
    m.prefixLen = cfg.prefix.length;
    m.suffixLen = cfg.suffix.length;
    m.adapterInfo = adapter.info
      ? `${adapter.info.vendor || ""} ${adapter.info.architecture || ""} ${adapter.info.description || ""}`.trim() || "adapter"
      : "adapter";
    m.limitsInfo = {
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxBufferSize: lim.maxBufferSize,
      maxComputeInvocationsPerWorkgroup: lim.maxComputeInvocationsPerWorkgroup,
    };

    const wgslSrc = await (await fetch(cfg.wgslUrl || "tap_vanity.wgsl")).text();
    const module = device.createShaderModule({ code: wgslSrc });

    // k0
    if (cfg.k0 !== undefined) {
      m.k0 = SECP.mod(BigInt(cfg.k0), SECP.N);
      if (m.k0 === 0n) m.k0 = 1n;
    } else {
      const rnd = new Uint8Array(32);
      crypto.getRandomValues(rnd);
      m.k0 = SECP.mod(SECP.bytesToBig(rnd), SECP.N);
      if (m.k0 === 0n) m.k0 = 1n;
    }

    const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
    m.tableBuf = device.createBuffer({ label: "table", size: 32 * 256 * 16 * 4, usage: S });
    m.stateBuf = device.createBuffer({ label: "state", size: m.totalThreads * 24 * 4, usage: S });

    const cfgArr = new Uint32Array(68);
    cfgArr[0] = m.prefixLen;
    cfgArr[1] = m.suffixLen;
    cfgArr[2] = m.mode === "fast" ? 1 : 0;
    cfgArr[3] = m.totalThreads;
    cfg.prefix.forEach((v, i) => (cfgArr[4 + i] = v));
    cfg.suffix.forEach((v, i) => (cfgArr[36 + i] = v));
    m.cfgBuf = device.createBuffer({ label: "cfg", size: cfgArr.byteLength, usage: S | CD });
    device.queue.writeBuffer(m.cfgBuf, 0, cfgArr);

    const params = new Uint32Array(12);
    let k = m.k0;
    for (let i = 0; i < 8; i++) { params[i] = Number(k & 0xffffffffn); k >>= 32n; }
    params[8] = 0;
    params[9] = m._iters;
    m.paramsBuf = device.createBuffer({ label: "params", size: params.byteLength, usage: S | CD });
    device.queue.writeBuffer(m.paramsBuf, 0, params);
    m._paramScratch = new Uint32Array(2);

    m.FOUND_SIZE = 20 * 4;
    m.foundBuf = device.createBuffer({ label: "found", size: m.FOUND_SIZE, usage: S | CD | CS });
    m.foundRead = device.createBuffer({ label: "found_read", size: m.FOUND_SIZE, usage: GPUBufferUsage.MAP_READ | CD });

    // Wide config binds the 64 MB 16-bit table at binding 5; layout entries a
    // given pipeline doesn't use are permitted, so one layout serves them all.
    const bindings = m.wide ? [0, 1, 2, 3, 4, 5] : [0, 1, 2, 3, 4];
    if (m.wide) {
      m.wtableBuf = device.createBuffer({ label: "wtable", size: WIDE_TABLE_BYTES, usage: S });
    }
    const bgl = device.createBindGroupLayout({
      entries: bindings.map((i) => ({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: (i === 1 || i === 4) ? "read-only-storage" : "storage" },
      })),
    });
    const bgEntries = [
      { binding: 0, resource: { buffer: m.tableBuf } },
      { binding: 1, resource: { buffer: m.cfgBuf } },
      { binding: 2, resource: { buffer: m.foundBuf } },
      { binding: 3, resource: { buffer: m.stateBuf } },
      { binding: 4, resource: { buffer: m.paramsBuf } },
    ];
    if (m.wide) bgEntries.push({ binding: 5, resource: { buffer: m.wtableBuf } });
    m.bindGroup = device.createBindGroup({ layout: bgl, entries: bgEntries });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    // init_table / setup always use the proven (fe_mul) squaring; the dedicated
    // squaring override only ever touches the search pipeline, so a fastSqr
    // miscompile can't corrupt the window table (which the self-test trusts).
    m.initPipeline = device.createComputePipeline({ label: "init_table", layout, compute: { module, entryPoint: "init_table" } });
    m.setupPipeline = device.createComputePipeline({ label: "setup", layout, compute: { module, entryPoint: "setup" } });
    if (m.wide) {
      m.initWidePipeline = device.createComputePipeline({
        label: "init_table_wide", layout, compute: { module, entryPoint: "init_table_wide" },
      });
    }
    const searchConstants = (m.kernel === "search" || m.kernel === "search_wide")
      ? { USE_FAST_SQR: m.fastSqr ? 1 : 0 }
      : {};
    m.searchPipeline = device.createComputePipeline({
      label: m.kernel, layout, compute: { module, entryPoint: m.kernel, constants: searchConstants },
    });
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

  async prepare() {
    const t = performance.now();
    this.dispatch(this.initPipeline, (32 * 256) / 256);
    await this.device.queue.onSubmittedWorkDone();
    if (this.wide) {
      // 16*65536 entries, built from the (already-complete) 8-bit table.
      // One entry = at most 2 mixed adds + 1 affine inversion; a single
      // 4096-workgroup dispatch is ~1M entries of bounded work.
      this.dispatch(this.initWidePipeline, (16 * 65536) / 256);
      await this.device.queue.onSubmittedWorkDone();
    }
    this.dispatch(this.setupPipeline, this.threadgroups);
    await this.device.queue.onSubmittedWorkDone();
    this.prepMs = +(performance.now() - t).toFixed(1);
  }

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
      const err = new Error(`GPU dispatch failed (device lost / watchdog): ${e && e.message ? e.message : e}`);
      err.deviceLost = true;
      throw err;
    }
    const dt = performance.now() - t;
    const found = new Uint32Array(this.foundRead.getMappedRange().slice(0));
    this.foundRead.unmap();

    const keys = this.totalThreads * iters;
    this._iterBase += iters;
    this.iters = iters;

    // adapt next launch toward the wall-time budget, aligned to batch size
    if (dt > 0) {
      const scale = Math.max(0.5, Math.min(2, this.frameBudgetMs / dt));
      let next = Math.round((iters * scale) / this._batchAlign) * this._batchAlign;
      next = Math.max(this._batchAlign, Math.min(this.maxIters, next || this._batchAlign));
      this._iters = next;
    }

    if (found[0] === 0) {
      if (this._iterBase >= 0x7fff0000) {
        const err = new Error("search space exhausted for this session — start again");
        err.exhausted = true;
        throw err;
      }
      return { hit: null, keys, dt };
    }
    return { hit: this.reconstruct(found), keys, dt };
  }

  async searchOnce(iterBase) {
    if (typeof iterBase === "number") this._iterBase = iterBase >>> 0;
    const { hit } = await this.searchStep();
    return hit;
  }

  reconstruct(found) {
    const tid = BigInt(found[1]);
    const iter = BigInt(found[2]);
    const kRaw = SECP.mod(this.k0 + (tid << 32n) + iter, SECP.N);
    let gpuX = 0n;
    for (let i = 7; i >= 0; i--) gpuX = (gpuX << 32n) | BigInt(found[12 + i]);

    if (this.mode === "fast") {
      const pub = SECP.pubkey(kRaw);
      const k = pub.parityOdd ? SECP.mod(SECP.N - kRaw, SECP.N) : kRaw;
      return { mode: "fast", key: k, gpuX, tid: found[1], iter: found[2] };
    }
    const parity = found[3] & 1;
    const internal = parity ? SECP.mod(SECP.N - kRaw, SECP.N) : kRaw;
    let gpuTweak = 0n;
    for (let i = 7; i >= 0; i--) gpuTweak = (gpuTweak << 32n) | BigInt(found[4 + i]);
    return { mode: "standard", internal, gpuTweak, gpuX, parity, tid: found[1], iter: found[2] };
  }

  keysPerLaunch() { return this.totalThreads * this.iters; }

  destroy() { try { this.device.destroy(); } catch (_) { /* ignore */ } }

  // k=1 known-vector check for a given tier. Returns {ok, reason}. Mines with
  // k0=1 and the known address prefix so a compiler miscompile is caught in the
  // very first launch (instant), before any real mining begins.
  static async selfCheck(cfg) {
    const log = cfg.log || CheekyLog;
    let miner = null;
    try {
      miner = await Miner.create({
        prefix: patternTo5bit(KNOWN_K1_ADDR.slice(4, 12)), // "mfr3p9j0"
        suffix: [], mode: "standard", kernel: cfg.kernel, fastSqr: cfg.fastSqr,
        threadgroups: 1, wgslUrl: cfg.wgslUrl, k0: 1n,
        startIters: cfg.kernel === "search_opt" ? OPT_BATCH : 4, log,
      });
      await miner.prepare();
      // a couple of launches in case the batch needs to sweep past iter 0
      for (let i = 0; i < 3; i++) {
        const { hit } = await miner.searchStep();
        if (hit) {
          const res = verifyHit(hit, "", "", log);
          if (res.address !== KNOWN_K1_ADDR) return { ok: false, reason: `k1 address ${res.address}` };
          return { ok: true };
        }
      }
      return { ok: false, reason: "k=1 vector not reproduced" };
    } catch (e) {
      return { ok: false, reason: String((e && e.message) || e) };
    } finally {
      try { miner && miner.destroy(); } catch (_) { /* ignore */ }
    }
  }

  // Short timed benchmark of one tier config. Mines a fixed astronomically-
  // unlikely pattern (so nothing real can be found mid-benchmark) for a couple
  // of warmup launches plus ~1.2s of measurement; returns { mkps, prepMs }.
  static async benchTier(tier, opts, log) {
    let miner = null;
    try {
      miner = await Miner.create({
        prefix: patternTo5bit("qqqqqqqq"), suffix: [], mode: "standard",
        kernel: tier.kernel, fastSqr: tier.fastSqr,
        threadgroups: opts.threadgroups || 64, wgslUrl: opts.wgslUrl,
        frameBudgetMs: opts.frameBudgetMs, log,
      });
      await miner.prepare();
      for (let i = 0; i < 3; i++) await miner.searchStep(); // warmup + calibration
      let keys = 0;
      const t0 = performance.now();
      while (performance.now() - t0 < 1200) {
        const r = await miner.searchStep();
        keys += r.keys;
      }
      const dt = (performance.now() - t0) / 1000;
      const mkps = +(keys / dt / 1e6).toFixed(3);
      log.info("bench-tier", { tier: tier.name, kernel: tier.kernel, fastSqr: tier.fastSqr, mkps, prepMs: miner.prepMs });
      return { mkps, prepMs: miner.prepMs };
    } catch (e) {
      log.warn("bench-tier-fail", { tier: tier.name, reason: String((e && e.message) || e).slice(0, 100) });
      return { mkps: 0, prepMs: 0 };
    } finally {
      try { miner && miner.destroy(); } catch (_) { /* ignore */ }
    }
  }

  // High-level driver. opts: { prefix?, suffix, mode?, fastSqr?, forceSafe?,
  //   wgslUrl, threadgroups?, frameBudgetMs?, onReady?, onProgress?, onDiag?,
  //   shouldStop?, log? }
  // Returns { found, tried, rate, elapsed, tier } or { stopped:true }.
  static async mine(opts) {
    const log = opts.log || CheekyLog;
    const compiler = detectCompiler();
    const prefix5 = patternTo5bit(opts.prefix || "");
    const suffix5 = patternTo5bit(opts.suffix || "");
    const mode = opts.mode === "fast" ? "fast" : "standard";
    const diag = (msg) => opts.onDiag && opts.onDiag(msg);

    // Kernel tiers, most-optimized first; each is gated by the k=1 self-test so
    // a compiler that miscompiles a tier is detected before real mining and
    // escalates to the next (safer) one.
    //
    // Optimizations: (1) 16-bit "wide" window table (64 MB, capability-gated
    // AND runtime-benchmarked — on bandwidth-starved mobile GPUs a big table
    // can be SLOWER even when it fits, so wide only wins its slot by measuring
    // faster than the 8-bit table on THIS device); (2) dedicated fe_sqr on the
    // proven `search` shape. Safe = all off. (A batched Montgomery-inversion
    // kernel was tried but miscompiled on Tint/Chrome — reverted.)
    const optFastSqr = opts.fastSqr !== false; // default: try dedicated squaring
    const tryWide = mode !== "fast" && opts.wide !== false && !opts.forceSafe;
    let tiers = mode === "fast"
      ? [{ name: "fast", kernel: "search_fast", fastSqr: false }]
      : [
          { name: "fast", kernel: "search", fastSqr: optFastSqr },
          { name: "safe", kernel: "search", fastSqr: false },
        ];

    if (tryWide) {
      // Wide is admitted to the tier list only if it (a) passes the k=1
      // self-test and (b) measures faster than the narrow fast tier here and
      // now. Both measured rates are logged so field logs answer what phones do.
      const wideTier = { name: "wide", kernel: "search_wide", fastSqr: optFastSqr };
      log.info("wide-selftest-begin", { kernel: "search_wide", fastSqr: optFastSqr });
      const wsc = await Miner.selfCheck({ kernel: "search_wide", fastSqr: optFastSqr, wgslUrl: opts.wgslUrl, log });
      if (!wsc.ok) {
        log.warn("wide-rejected", { stage: "selftest", reason: wsc.reason, compiler });
      } else {
        log.info("wide-selftest-ok", {});
        const narrowTier = tiers[0];
        const [wideBench, narrowBench] = [
          await Miner.benchTier(wideTier, opts, log),
          await Miner.benchTier(narrowTier, opts, log),
        ];
        const decision =
          wideBench.mkps > 0 && wideBench.mkps > narrowBench.mkps * 1.03 ? "wide" : "narrow";
        log.info("table-decision", {
          chosen: decision,
          wideMkps: wideBench.mkps, narrowMkps: narrowBench.mkps,
          widePrepMs: wideBench.prepMs, narrowPrepMs: narrowBench.prepMs,
          compiler,
        });
        if (decision === "wide") tiers = [wideTier, ...tiers];
      }
    }
    let tierIdx = opts.forceSafe ? tiers.length - 1 : 0;

    const t0 = performance.now();
    let totalTried = 0;
    let lastRateLog = -1e9;

    log.info("startup", {
      compiler, ua: ((navigator && navigator.userAgent) || "").slice(0, 140),
      mode, target: { prefix: opts.prefix || "", suffix: opts.suffix || "" },
    });

    tierLoop:
    for (; tierIdx < tiers.length; tierIdx++) {
      const tier = tiers[tierIdx];

      // ---- per-tier self-test gate ----
      log.info("selftest-begin", { tier: tier.name, kernel: tier.kernel, fastSqr: tier.fastSqr });
      const sc = await Miner.selfCheck({ kernel: tier.kernel, fastSqr: tier.fastSqr, wgslUrl: opts.wgslUrl, log });
      if (!sc.ok) {
        log.err("selftest-fail", { tier: tier.name, reason: sc.reason, compiler });
        if (tierIdx < tiers.length - 1) { diag("self-test failed — switching to safe mode"); continue tierLoop; }
        throw new Error(`GPU self-test failed even in safe mode (${compiler}): ${sc.reason}`);
      }
      log.info("selftest-ok", { tier: tier.name });

      // ---- mine this tier, with device-loss shrink recovery ----
      let threadgroups = opts.threadgroups || 64;
      for (let attempt = 0; ; attempt++) {
        let miner = null;
        try {
          miner = await Miner.create({
            prefix: prefix5, suffix: suffix5, mode, kernel: tier.kernel, fastSqr: tier.fastSqr,
            threadgroups, wgslUrl: opts.wgslUrl, frameBudgetMs: opts.frameBudgetMs, log,
            onDiag: opts.onDiag,
          });
          log.info("miner-ready", {
            tier: tier.name, kernel: tier.kernel, fastSqr: tier.fastSqr, batch: miner.batchSize,
            threadgroups, adapter: miner.adapterInfo, limits: miner.limitsInfo, compiler,
          });
          opts.onReady && opts.onReady({
            adapter: miner.adapterInfo, threadgroups, compiler, tier: tier.name,
            kernel: tier.kernel, fastSqr: tier.fastSqr, batch: miner.batchSize,
          });
          await miner.prepare();
          log.info("prepared", { tier: tier.name, prepMs: miner.prepMs });

          for (;;) {
            if (opts.shouldStop && opts.shouldStop()) {
              log.info("stopped", { tried: Math.round(totalTried) });
              return { stopped: true };
            }
            const { hit, keys, dt } = await miner.searchStep();
            totalTried += keys;
            const elapsed = (performance.now() - t0) / 1000;
            const rate = elapsed > 0 ? totalTried / elapsed : 0;
            opts.onProgress && opts.onProgress({ tried: totalTried, rate, elapsed, iters: miner.iters });
            if (elapsed - lastRateLog >= 5) {
              lastRateLog = elapsed;
              log.info("rate", {
                tried: Math.round(totalTried), mkps: +(rate / 1e6).toFixed(3),
                launchKeys: keys, iters: miner.iters, dtMs: +dt.toFixed(1), tier: tier.name,
              });
            }
            if (hit) {
              try {
                const res = verifyHit(hit, opts.prefix || "", opts.suffix || "", log);
                const el = (performance.now() - t0) / 1000;
                log.info("found", {
                  address: res.address, parity: hit.parity, tried: Math.round(totalTried),
                  mkps: +(totalTried / Math.max(el, 1e-9) / 1e6).toFixed(3),
                  elapsedS: +el.toFixed(1), tier: tier.name,
                });
                return { found: res, tried: totalTried, rate: totalTried / Math.max(el, 1e-9), elapsed: el, tier: tier.name };
              } catch (ve) {
                log.err("verify-fail-runtime", { reason: String(ve.message || ve).slice(0, 120), tier: tier.name, compiler });
                if (tierIdx < tiers.length - 1) {
                  diag("hiccup — switching to safe mode");
                  continue tierLoop; // finally destroys miner, tierLoop increments
                }
                diag("mining failed verification even in safe mode");
                throw ve;
              }
            }
          }
        } catch (e) {
          const msg = String((e && e.message) || e);
          const recoverable = (e && e.deviceLost) || /device.*lost|map async|destroyed/i.test(msg);
          if (recoverable && threadgroups > 8 && attempt < 6) {
            threadgroups = Math.max(8, Math.floor(threadgroups / 2));
            log.warn("device-lost", { msg: msg.slice(0, 80), retryThreadgroups: threadgroups, tier: tier.name });
            diag(`GPU hiccup — retrying smaller (${threadgroups} workgroups)`);
            await new Promise((r) => setTimeout(r, 250));
            continue; // inner attempt loop
          }
          log.err("fatal", { msg: msg.slice(0, 140), tier: tier.name });
          throw e;
        } finally {
          try { miner && miner.destroy(); } catch (_) { /* ignore */ }
        }
      }
    }
    throw new Error("no viable kernel tier");
  }
}

// Verify a reconstructed key; returns the display result or throws (tagged
// .verifyFail). Logs each sub-check pass/fail. NEVER logs key material.
function verifyHit(hit, prefixStr, suffixStr, log) {
  log = log || CheekyLog;
  const fail = (reason) => { const e = new Error("verification failed: " + reason); e.verifyFail = true; throw e; };
  const checkPattern = (address, label) => {
    const body = address.slice(4);
    if (prefixStr && !body.startsWith(prefixStr)) { log.err("vcheck", { check: label + "-prefix", ok: false }); fail(`${address} !startswith bc1p${prefixStr}`); }
    if (suffixStr && !address.endsWith(suffixStr)) { log.err("vcheck", { check: label + "-suffix", ok: false }); fail(`${address} !endswith ${suffixStr}`); }
    log.info("vcheck", { check: label + "-pattern", ok: true });
  };

  if (hit.mode === "fast") {
    const pub = SECP.pubkey(hit.key);
    if (pub.parityOdd) { log.err("vcheck", { check: "fast-eveny", ok: false }); fail("normalized key still odd-y"); }
    if (pub.xBig !== hit.gpuX) { log.err("vcheck", { check: "fast-gpux", ok: false }); fail("CPU pubkey.x != GPU x (compiler bug?)"); }
    log.info("vcheck", { check: "fast-gpux", ok: true });
    const address = SECP.bech32mP2TR(pub.x);
    checkPattern(address, "fast");
    const wif = SECP.wif(hit.key);
    return { mode: "fast", address, wif, descriptor: `rawtr(${wif})` };
  }

  // standard: prove internal+TapTweak == GPU Q.x, cross-check tweak, and
  // independently simulate the wallet (BIP-86) import path.
  const t = SECP.taprootFromInternalPriv(hit.internal);
  if (t.qxBig !== hit.gpuX) { log.err("vcheck", { check: "std-gpu-qx", ok: false }); fail("CPU output-key.x != GPU-reported Q.x (compiler bug?)"); }
  log.info("vcheck", { check: "std-gpu-qx", ok: true });
  if (t.tweak !== hit.gpuTweak) { log.err("vcheck", { check: "std-tweak", ok: false }); fail("CPU TapTweak != GPU-reported tweak"); }
  log.info("vcheck", { check: "std-tweak", ok: true });
  checkPattern(t.address, "std");
  const walletAddr = SECP.taprootFromInternalPriv(t.internalEven).address;
  if (walletAddr !== t.address) { log.err("vcheck", { check: "std-wallet-sim", ok: false }); fail("wallet BIP-86 path address mismatch"); }
  log.info("vcheck", { check: "std-wallet-sim", ok: true });
  const wif = SECP.wif(t.internalEven);
  return { mode: "standard", address: t.address, wif, tweakedSecret: t.outputSecret.toString(16).padStart(64, "0") };
}
