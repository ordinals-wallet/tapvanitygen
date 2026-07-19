// miner.js — WebGPU host for the tap_vanity.wgsl kernel (fast/rawtr mode).
// Mirrors the Rust wgpu host (wgpu/src/lib.rs): same bind group layout,
// buffer structs, and dispatch order (init_table -> setup -> search_fast loop).

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
  // cfg: { prefix: [5bit...], suffix: [5bit...], threadgroups, iters }
  static async create(cfg) {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter available");
    const device = await adapter.requestDevice();

    const wgslSrc = await (await fetch("tap_vanity.wgsl")).text();
    const module = device.createShaderModule({ code: wgslSrc });

    const m = new Miner();
    m.device = device;
    m.threadgroups = cfg.threadgroups;
    m.totalThreads = cfg.threadgroups * 256;
    m.iters = cfg.iters;
    m.prefixLen = cfg.prefix.length;
    m.suffixLen = cfg.suffix.length;
    m.adapterInfo = adapter.info ? `${adapter.info.vendor || ""} ${adapter.info.architecture || ""} ${adapter.info.description || ""}`.trim() : "unknown";

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
    cfg.prefix.forEach((v, i) => cfgArr[4 + i] = v);
    cfg.suffix.forEach((v, i) => cfgArr[36 + i] = v);
    m.cfgBuf = device.createBuffer({ label: "cfg", size: cfgArr.byteLength, usage: S | CD });
    device.queue.writeBuffer(m.cfgBuf, 0, cfgArr);

    // Params: k0[8] (LE u32 limbs), iter_base, iters, _p0, _p1
    const params = new Uint32Array(12);
    let k = m.k0;
    for (let i = 0; i < 8; i++) { params[i] = Number(k & 0xffffffffn); k >>= 32n; }
    params[8] = 0; // iter_base
    params[9] = m.iters;
    m.paramsBuf = device.createBuffer({ label: "params", size: params.byteLength, usage: S | CD });
    device.queue.writeBuffer(m.paramsBuf, 0, params);

    // Found: flag, tid, iter, parity, tweak[8], qx[8] = 20 u32
    m.FOUND_SIZE = 20 * 4;
    m.foundBuf = device.createBuffer({ label: "found", size: m.FOUND_SIZE, usage: S | CD | CS });
    m.foundRead = device.createBuffer({ label: "found_read", size: m.FOUND_SIZE, usage: GPUBufferUsage.MAP_READ | CD });

    // ----- bind group: 5 storage buffers, group(0) -----
    const bgl = device.createBindGroupLayout({
      entries: [0, 1, 2, 3, 4].map(i => ({
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

  // One search_fast launch at iterBase. Returns reconstructed hit or null.
  async searchOnce(iterBase) {
    // update params.iter_base at byte offset 32
    this.device.queue.writeBuffer(this.paramsBuf, 32, new Uint32Array([iterBase >>> 0]));
    this.dispatch(this.searchPipeline, this.threadgroups);
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.foundBuf, 0, this.foundRead, 0, this.FOUND_SIZE);
    this.device.queue.submit([enc.finish()]);
    await this.foundRead.mapAsync(GPUMapMode.READ);
    const found = new Uint32Array(this.foundRead.getMappedRange().slice(0));
    this.foundRead.unmap();
    if (found[0] === 0) return null;
    return this.reconstruct(found);
  }

  reconstruct(found) {
    const tid = BigInt(found[1]);
    const iter = BigInt(found[2]);
    // k_raw = k0 + tid*2^32 + iter (mod n)
    const kRaw = SECP.mod(this.k0 + (tid << 32n) + iter, SECP.N);
    // fast mode: GPU doesn't report parity; normalize to even-y key
    const pub = SECP.pubkey(kRaw);
    const k = pub.parityOdd ? SECP.mod(SECP.N - kRaw, SECP.N) : kRaw;
    // GPU-reported qx (LE u32 limbs) -> hex, for cross-check
    let gpuX = 0n;
    for (let i = 7; i >= 0; i--) gpuX = (gpuX << 32n) | BigInt(found[12 + i]);
    return { key: k, gpuX, tid: found[1], iter: found[2] };
  }

  keysPerLaunch() { return this.totalThreads * this.iters; }
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
