//! tapvanity-wgpu — cross-platform GPU taproot (P2TR) vanity miner core.
//!
//! Reuses the WGSL kernel in `src/tap_vanity.wgsl` (a port of the verified
//! Metal shader in `../metal/shaders/tap_vanity.metal`). This module owns the
//! wgpu device/pipeline/buffer plumbing, host-side scalar reconstruction, and
//! in-process verification against the `bitcoin` crate.

use bitcoin::key::{Keypair, Secp256k1, TapTweak, TweakedPublicKey};
use bitcoin::secp256k1::SecretKey;
use bitcoin::{Address, Network, XOnlyPublicKey};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;

pub const BECH32_CHARSET: &[u8] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";

// secp256k1 order n, big-endian.
const N_BE: [u8; 32] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
    0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B, 0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41,
];

// ------------------------------------------------------------- scalar math (mod n)
// U256 as 4 x u64 little-endian.
pub type U256 = [u64; 4];

pub fn u256_from_be(b: &[u8; 32]) -> U256 {
    let mut r = [0u64; 4];
    for i in 0..4 {
        let mut v = 0u64;
        for j in 0..8 {
            v = (v << 8) | b[24 - 8 * i + j] as u64;
        }
        r[i] = v;
    }
    r
}
pub fn u256_to_be(a: &U256) -> [u8; 32] {
    let mut b = [0u8; 32];
    for i in 0..4 {
        let mut v = a[i];
        for j in 0..8 {
            b[31 - 8 * i - j] = (v & 0xff) as u8;
            v >>= 8;
        }
    }
    b
}
fn n_u256() -> U256 {
    u256_from_be(&N_BE)
}
fn u256_cmp(a: &U256, b: &U256) -> std::cmp::Ordering {
    for i in (0..4).rev() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    std::cmp::Ordering::Equal
}
fn u256_add(a: &U256, b: &U256) -> (U256, bool) {
    let mut r = [0u64; 4];
    let mut carry = 0u128;
    for i in 0..4 {
        let s = a[i] as u128 + b[i] as u128 + carry;
        r[i] = s as u64;
        carry = s >> 64;
    }
    (r, carry != 0)
}
fn u256_sub(a: &U256, b: &U256) -> (U256, bool) {
    let mut r = [0u64; 4];
    let mut borrow = 0i128;
    for i in 0..4 {
        let d = a[i] as i128 - b[i] as i128 - borrow;
        if d < 0 {
            r[i] = (d + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            r[i] = d as u64;
            borrow = 0;
        }
    }
    (r, borrow != 0)
}
pub fn mod_n(a: &U256) -> U256 {
    let n = n_u256();
    if u256_cmp(a, &n) != std::cmp::Ordering::Less {
        u256_sub(a, &n).0
    } else {
        *a
    }
}
pub fn add_mod_n(a: &U256, b: &U256) -> U256 {
    let n = n_u256();
    let (s, carry) = u256_add(a, b);
    if carry || u256_cmp(&s, &n) != std::cmp::Ordering::Less {
        u256_sub(&s, &n).0
    } else {
        s
    }
}
pub fn neg_mod_n(a: &U256) -> U256 {
    u256_sub(&n_u256(), a).0
}
pub fn hex_be(a: &U256) -> String {
    u256_to_be(a).iter().map(|b| format!("{:02x}", b)).collect()
}

// ------------------------------------------------------------- WIF (compressed, mainnet)
const B58: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
fn base58check(payload: &[u8]) -> String {
    use bitcoin::hashes::{sha256, Hash};
    let h1 = sha256::Hash::hash(payload);
    let h2 = sha256::Hash::hash(&h1[..]);
    let mut data = payload.to_vec();
    data.extend_from_slice(&h2[..4]);
    let zeros = data.iter().take_while(|&&b| b == 0).count();
    let mut num = data.clone();
    let mut out = Vec::new();
    let mut start = 0;
    while start < num.len() {
        let mut rem = 0u32;
        for i in start..num.len() {
            let cur = rem * 256 + num[i] as u32;
            num[i] = (cur / 58) as u8;
            rem = cur % 58;
        }
        out.push(B58[rem as usize]);
        while start < num.len() && num[start] == 0 {
            start += 1;
        }
    }
    let mut s: Vec<u8> = std::iter::repeat(b'1').take(zeros).collect();
    out.reverse();
    s.extend_from_slice(&out);
    String::from_utf8(s).unwrap()
}
pub fn wif(key: &U256) -> String {
    let mut payload = vec![0x80u8];
    payload.extend_from_slice(&u256_to_be(key));
    payload.push(0x01);
    base58check(&payload)
}

// ------------------------------------------------------------- bech32m (host, for GPU cross-check)
fn bech32_polymod(values: &[u8]) -> u32 {
    const GEN: [u32; 5] = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3];
    let mut chk = 1u32;
    for &v in values {
        let top = chk >> 25;
        chk = ((chk & 0x1ffffff) << 5) ^ v as u32;
        for i in 0..5 {
            if (top >> i) & 1 != 0 {
                chk ^= GEN[i];
            }
        }
    }
    chk
}
/// bech32m address for a 32-byte x-only key (witness v1), mainnet "bc".
pub fn bech32m_p2tr(xonly_be: &[u8; 32]) -> String {
    // convertbits 8->5
    let mut data = vec![1u8]; // witness version 1
    let mut acc = 0u32;
    let mut bits = 0u32;
    for &b in xonly_be {
        acc = (acc << 8) | b as u32;
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            data.push(((acc >> bits) & 31) as u8);
        }
    }
    if bits > 0 {
        data.push(((acc << (5 - bits)) & 31) as u8);
    }
    // hrp "bc" expand = [3,3,0,2,3]
    let hrp_exp = [3u8, 3, 0, 2, 3];
    let mut poly_in = hrp_exp.to_vec();
    poly_in.extend_from_slice(&data);
    poly_in.extend_from_slice(&[0u8; 6]);
    let polymod = bech32_polymod(&poly_in) ^ 0x2bc830a3;
    let mut checksum = Vec::new();
    for i in 0..6 {
        checksum.push(((polymod >> (5 * (5 - i))) & 31) as u8);
    }
    let mut s = String::from("bc1");
    for &v in data.iter().chain(checksum.iter()) {
        s.push(BECH32_CHARSET[v as usize] as char);
    }
    s
}

// ------------------------------------------------------------- verification (bitcoin crate)
pub struct Derived {
    pub address: String,
    pub output_secret: U256,
}
/// Standard mode: `key` is the internal key; apply BIP-341 tweak.
pub fn derive_standard(key: &U256) -> Derived {
    let secp = Secp256k1::new();
    let secret = SecretKey::from_slice(&u256_to_be(key)).expect("valid key");
    let kp = Keypair::from_secret_key(&secp, &secret);
    let (xonly, _) = XOnlyPublicKey::from_keypair(&kp);
    let address = Address::p2tr(&secp, xonly, None, Network::Bitcoin).to_string();
    let tweaked = kp.tap_tweak(&secp, None);
    let out = tweaked.to_inner().secret_key();
    Derived {
        address,
        output_secret: u256_from_be(&out.secret_bytes()),
    }
}
/// Return the even-y variant of `key`: if pubkey(key).y is odd, return n-key,
/// else key. Used to normalize a fast/rawtr output key to the x-only signing
/// convention (address is the same either way).
pub fn rawtr_even_key(key: &U256) -> U256 {
    let secp = Secp256k1::new();
    let secret = SecretKey::from_slice(&u256_to_be(key)).expect("valid key");
    let kp = Keypair::from_secret_key(&secp, &secret);
    let (_xonly, parity) = XOnlyPublicKey::from_keypair(&kp);
    match parity {
        bitcoin::key::Parity::Odd => neg_mod_n(key),
        bitcoin::key::Parity::Even => *key,
    }
}

/// Fast/rawtr mode: `key` is the output key itself; no tweak.
pub fn derive_rawtr(key: &U256) -> Derived {
    let secp = Secp256k1::new();
    let secret = SecretKey::from_slice(&u256_to_be(key)).expect("valid key");
    let kp = Keypair::from_secret_key(&secp, &secret);
    let (xonly, _) = XOnlyPublicKey::from_keypair(&kp);
    let tweaked = TweakedPublicKey::dangerous_assume_tweaked(xonly);
    let address = Address::p2tr_tweaked(tweaked, Network::Bitcoin).to_string();
    Derived {
        address,
        output_secret: *key,
    }
}

// ------------------------------------------------------------- GPU buffers layout
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Cfg {
    prefix_len: u32,
    suffix_len: u32,
    fast: u32,
    total_threads: u32,
    prefix: [u32; 32],
    suffix: [u32; 32],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    k0: [u32; 8],
    iter_base: u32,
    iters: u32,
    _p0: u32,
    _p1: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct Found {
    flag: u32,
    tid: u32,
    iter: u32,
    parity: u32,
    tweak: [u32; 8],
    qx: [u32; 8],
}

pub struct Config {
    pub prefix: Vec<u8>, // 5-bit values
    pub suffix: Vec<u8>,
    pub fast: bool,
    pub threadgroups: u32,
    pub threads: u32, // must be 256 (workgroup size in WGSL)
    pub iters: u32,
}

pub struct Hit {
    pub address: String,
    pub mode: &'static str, // "standard" | "fast"
    pub secret: U256,       // internal key (standard) or output key (fast)
    pub tweaked: Option<U256>,
}

#[allow(dead_code)]
pub struct Miner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group: wgpu::BindGroup,
    init_pipeline: wgpu::ComputePipeline,
    setup_pipeline: wgpu::ComputePipeline,
    search_pipeline: wgpu::ComputePipeline,
    cfg_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    found_buf: wgpu::Buffer,
    found_read: wgpu::Buffer,
    total_threads: u32,
    threadgroups: u32,
    iters: u32,
    fast: bool,
    prefix_len: u32,
    suffix_len: u32,
    k0: U256,
    pub adapter_name: String,
    pub backend: String,
}

pub fn pattern_to_5bit(s: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for ch in s.chars() {
        match BECH32_CHARSET.iter().position(|&c| c as char == ch) {
            Some(i) => out.push(i as u8),
            None => {
                return Err(format!(
                    "invalid bech32 char '{}' (valid: {}; lowercase, no 1/b/i/o)",
                    ch,
                    std::str::from_utf8(BECH32_CHARSET).unwrap()
                ))
            }
        }
    }
    Ok(out)
}

impl Miner {
    pub fn new(cfg: &Config, k0: U256) -> Result<Self, String> {
        pollster::block_on(Self::new_async(cfg, k0))
    }

    async fn new_async(cfg: &Config, k0: U256) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or("no suitable GPU adapter found")?;
        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend = format!("{:?}", info.backend);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("tapvanity"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("request_device failed: {e}"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tap_vanity"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("tap_vanity.wgsl"))),
        });

        let total_threads = cfg.threadgroups * cfg.threads;

        // buffers
        let table_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("table"),
            size: (32 * 256 * 16 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("state"),
            size: (total_threads as u64) * 24 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let mut cfg_data = Cfg {
            prefix_len: cfg.prefix.len() as u32,
            suffix_len: cfg.suffix.len() as u32,
            fast: if cfg.fast { 1 } else { 0 },
            total_threads,
            prefix: [0; 32],
            suffix: [0; 32],
        };
        for (i, &b) in cfg.prefix.iter().enumerate() {
            cfg_data.prefix[i] = b as u32;
        }
        for (i, &b) in cfg.suffix.iter().enumerate() {
            cfg_data.suffix[i] = b as u32;
        }
        let cfg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cfg"),
            size: std::mem::size_of::<Cfg>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&cfg_buf, 0, bytemuck::bytes_of(&cfg_data));

        let mut k0_limbs = [0u32; 8];
        for i in 0..4 {
            k0_limbs[2 * i] = k0[i] as u32;
            k0_limbs[2 * i + 1] = (k0[i] >> 32) as u32;
        }
        let params0 = Params {
            k0: k0_limbs,
            iter_base: 0,
            iters: cfg.iters,
            _p0: 0,
            _p1: 0,
        };
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params0));

        let found_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("found"),
            size: std::mem::size_of::<Found>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let found_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("found_read"),
            size: std::mem::size_of::<Found>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // bind group layout: 5 storage buffers
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..5)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if i == 1 || i == 4 {
                        wgpu::BufferBindingType::Storage { read_only: true }
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: false }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &entries,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: table_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cfg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: found_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let mkpipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &shader,
                entry_point: entry,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let init_pipeline = mkpipe("init_table");
        let setup_pipeline = mkpipe("setup");
        let search_pipeline = if cfg.fast { mkpipe("search_fast") } else { mkpipe("search") };

        Ok(Miner {
            device,
            queue,
            bind_group,
            init_pipeline,
            setup_pipeline,
            search_pipeline,
            cfg_buf: cfg_buf,
            params_buf,
            found_buf,
            found_read,
            total_threads,
            threadgroups: cfg.threadgroups,
            iters: cfg.iters,
            fast: cfg.fast,
            prefix_len: cfg.prefix.len() as u32,
            suffix_len: cfg.suffix.len() as u32,
            k0,
            adapter_name,
            backend,
        })
    }

    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, groups: u32) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    /// Build the window table and seed per-thread start points. Call once.
    pub fn prepare(&self) {
        self.dispatch(&self.init_pipeline, 32 * 256 / 256);
        self.device.poll(wgpu::Maintain::Wait);
        self.dispatch(&self.setup_pipeline, self.threadgroups);
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Run one search launch at iter_base. Returns Some(Found-derived Hit) if found.
    pub fn search_once(&self, iter_base: u32) -> Option<Hit> {
        // update params.iter_base
        self.queue
            .write_buffer(&self.params_buf, 32, bytemuck::bytes_of(&iter_base));
        self.dispatch(&self.search_pipeline, self.threadgroups);
        // read found
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(
            &self.found_buf,
            0,
            &self.found_read,
            0,
            std::mem::size_of::<Found>() as u64,
        );
        self.queue.submit(Some(enc.finish()));

        let slice = self.found_read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let found: Found = *bytemuck::from_bytes(&data);
        drop(data);
        self.found_read.unmap();

        if found.flag == 0 {
            return None;
        }
        Some(self.reconstruct(&found))
    }

    fn reconstruct(&self, found: &Found) -> Hit {
        // k = k0 + tid*2^32 + iter (mod n)
        let mut offset: U256 = [0; 4];
        offset[0] = (found.iter as u64) | ((found.tid as u64) << 32);
        offset[1] = (found.tid as u64) >> 32;
        let k_raw = add_mod_n(&self.k0, &mod_n(&offset));
        if self.fast {
            // The fast kernel does not report parity. Determine it from the
            // reconstructed key: if its pubkey has odd y, the even-y signing
            // key (rawtr output key) is n - k_raw. Address is identical either
            // way (depends only on x).
            let k = rawtr_even_key(&k_raw);
            let d = derive_rawtr(&k);
            Hit {
                address: d.address,
                mode: "fast",
                secret: k,
                tweaked: None,
            }
        } else {
            let mut k = k_raw;
            if found.parity == 1 {
                k = neg_mod_n(&k);
            }
            let d = derive_standard(&k);
            Hit {
                address: d.address,
                mode: "standard",
                secret: k,
                tweaked: Some(d.output_secret),
            }
        }
    }

    pub fn keys_per_launch(&self) -> u64 {
        self.total_threads as u64 * self.iters as u64
    }

    pub fn iters(&self) -> u32 { self.iters }
}

/// GPU self-test against the k=1 known vector. Mines with k0=1 and a prefix
/// taken from the known standard-mode address; a hit proves the GPU's field
/// math + TapTweak + t*G + bech32m all agree with the reference. Returns Ok
/// only when the GPU-found key reconstructs to k=1 and the bitcoin-crate
/// derivation matches the full known address.
pub fn gpu_self_test() -> Result<(), String> {
    const KNOWN: &str = "bc1pmfr3p9j00pfxjh0zmgp99y8zftmd3s5pmedqhyptwy6lm87hf5sspknck9";
    // reference check first
    let ref_addr = derive_standard(&[1, 0, 0, 0]).address;
    if ref_addr != KNOWN {
        return Err(format!("reference k=1 mismatch: {ref_addr}"));
    }
    let prefix = pattern_to_5bit(&KNOWN[4..12]).unwrap(); // "mfr3p9j0"
    let cfg = Config {
        prefix,
        suffix: vec![],
        fast: false,
        threadgroups: 1,
        threads: 256,
        iters: 1,
    };
    let miner = Miner::new(&cfg, [1, 0, 0, 0])?;
    miner.prepare();
    match miner.search_once(0) {
        Some(hit) => {
            if hit.secret != [1, 0, 0, 0] {
                return Err(format!(
                    "GPU found a hit but key != 1 (got {})",
                    hex_be(&hit.secret)
                ));
            }
            if hit.address != KNOWN {
                return Err(format!("GPU address mismatch: {}", hit.address));
            }
            Ok(())
        }
        None => Err("GPU did not reproduce the k=1 known vector (field math bug)".into()),
    }
}
