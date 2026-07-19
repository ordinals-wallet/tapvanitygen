//! End-to-end GPU tests. These require a working GPU adapter. On headless CI
//! runners without a GPU, set TAPVANITY_SKIP_GPU=1 to soft-skip.

use tapvanity_wgpu::*;

fn gpu_available() -> bool {
    std::env::var("TAPVANITY_SKIP_GPU").is_err()
}

fn addr_matches(addr: &str, prefix: &str, suffix: &str) -> bool {
    let after = &addr[4..]; // strip bc1p
    after.starts_with(prefix) && addr.ends_with(suffix)
}

fn mine(fast: bool, prefix: &str, suffix: &str) -> Hit {
    let cfg = Config {
        prefix: pattern_to_5bit(prefix).unwrap(),
        suffix: pattern_to_5bit(suffix).unwrap(),
        fast,
        threadgroups: 256,
        threads: 256,
        iters: 64,
    };
    // deterministic-ish seed for reproducibility of test timing (still random)
    use rand::RngCore;
    let mut b = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut b);
    let k0 = mod_n(&u256_from_be(&b));
    let miner = Miner::new(&cfg, k0).expect("gpu init");
    miner.prepare();
    let mut base = 0u32;
    loop {
        if let Some(h) = miner.search_once(base) {
            return h;
        }
        base = base.wrapping_add(miner.iters());
        assert!(base < 0xFFF0_0000, "pattern not found in stride window");
    }
}

#[test]
fn self_test_known_vector() {
    if !gpu_available() {
        eprintln!("skipping (no GPU)");
        return;
    }
    gpu_self_test().expect("k=1 GPU self-test");
}

#[test]
fn standard_prefix_verifies() {
    if !gpu_available() {
        return;
    }
    let h = mine(false, "qq", "");
    assert_eq!(h.mode, "standard");
    let d = derive_standard(&h.secret);
    assert_eq!(d.address, h.address, "in-process derivation mismatch");
    assert!(addr_matches(&h.address, "qq", ""), "pattern: {}", h.address);
}

#[test]
fn standard_combined_verifies() {
    if !gpu_available() {
        return;
    }
    let h = mine(false, "t", "q");
    let d = derive_standard(&h.secret);
    assert_eq!(d.address, h.address);
    assert!(addr_matches(&h.address, "t", "q"), "pattern: {}", h.address);
}

#[test]
fn fast_rawtr_prefix_verifies() {
    if !gpu_available() {
        return;
    }
    let h = mine(true, "qq", "");
    assert_eq!(h.mode, "fast");
    let d = derive_rawtr(&h.secret);
    assert_eq!(d.address, h.address, "rawtr derivation mismatch");
    assert!(addr_matches(&h.address, "qq", ""), "pattern: {}", h.address);
}

#[test]
fn fast_combined_verifies() {
    if !gpu_available() {
        return;
    }
    let h = mine(true, "t", "q");
    let d = derive_rawtr(&h.secret);
    assert_eq!(d.address, h.address);
    assert!(addr_matches(&h.address, "t", "q"), "pattern: {}", h.address);
}

#[test]
fn scalar_roundtrip() {
    // pure-CPU sanity: WIF/hex reconstruction and mod-n arithmetic
    let k: U256 = [0x1234, 0x5678, 0x9abc, 0xdef0];
    let neg = neg_mod_n(&k);
    let back = neg_mod_n(&neg);
    assert_eq!(back, k);
    assert_eq!(hex_be(&[1, 0, 0, 0]).len(), 64);
}
