//! tapvanity-wgpu CLI — cross-platform GPU taproot vanity miner.

use std::time::Instant;
use tapvanity_wgpu::*;

fn usage() {
    eprintln!(
        r#"
  tapvanity-wgpu — Bitcoin P2TR (bc1p) vanity miner (Rust + wgpu)

    --prefix <pat>      match  bc1p<pat>...
    --suffix <pat>      match  ...<pat>   (may reach into the checksum chars)
                        (--prefix and --suffix may be combined)
    --fast              FAST/rawtr mode: mine the OUTPUT key directly
                        (no TapTweak; result is a rawtr() descriptor key)
    --estimate          benchmark + difficulty/ETA, then exit
    --self-test         GPU known-vector check, then exit

  Performance:
    --threadgroups <N>  workgroups (default 256)
    --iters <N>         iterations per thread per launch (default 64)

  Charset (bech32): qpzry9x8gf2tvdw0s3jn54khce6mua7l
  (no '1', 'b', 'i', 'o'; lowercase only)
"#
    );
}

fn rand_k0() -> U256 {
    use rand::RngCore;
    let mut b = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut b);
    let mut k = mod_n(&u256_from_be(&b));
    if k == [0, 0, 0, 0] {
        k[0] = 1;
    }
    k
}

fn main() {
    let mut prefix: Option<String> = None;
    let mut suffix: Option<String> = None;
    let mut fast = false;
    let mut estimate = false;
    let mut selftest = false;
    let mut threadgroups = 256u32;
    let mut iters = 64u32;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => {
                usage();
                return;
            }
            "--prefix" => prefix = args.next(),
            "--suffix" => suffix = args.next(),
            "--fast" => fast = true,
            "--estimate" => estimate = true,
            "--self-test" => selftest = true,
            "--threadgroups" => threadgroups = args.next().and_then(|s| s.parse().ok()).unwrap_or(threadgroups),
            "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(iters),
            other => {
                eprintln!("Unknown argument: {other}");
                usage();
                std::process::exit(2);
            }
        }
    }
    iters = iters.max(16);

    println!("\n  tapvanity-wgpu — P2TR vanity miner");

    print!("  GPU self-test (k=1 known vector)... ");
    match gpu_self_test() {
        Ok(()) => println!("OK"),
        Err(e) => {
            println!("FAILED\n  {e}");
            std::process::exit(1);
        }
    }
    if selftest {
        println!("  Self-test passed.");
        return;
    }

    if prefix.is_none() && suffix.is_none() {
        usage();
        std::process::exit(2);
    }
    let prefix5 = match prefix.as_deref().map(pattern_to_5bit) {
        Some(Ok(v)) => v,
        Some(Err(e)) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
        None => vec![],
    };
    let suffix5 = match suffix.as_deref().map(pattern_to_5bit) {
        Some(Ok(v)) => v,
        Some(Err(e)) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
        None => vec![],
    };
    if prefix5.len() > 32 || suffix5.len() > 32 {
        eprintln!("pattern too long (max 32)");
        std::process::exit(2);
    }

    let cfg = Config {
        prefix: prefix5.clone(),
        suffix: suffix5.clone(),
        fast,
        threadgroups,
        threads: 256,
        iters,
    };
    let k0 = rand_k0();
    let miner = Miner::new(&cfg, k0).unwrap_or_else(|e| {
        eprintln!("GPU init failed: {e}");
        std::process::exit(1);
    });
    println!("  Adapter: {} ({})", miner.adapter_name, miner.backend);
    print!("  Building window table + seeding threads... ");
    let t0 = Instant::now();
    miner.prepare();
    println!("{:.2}s", t0.elapsed().as_secs_f64());

    let per_launch = miner.keys_per_launch();
    // benchmark one launch
    let tb = Instant::now();
    let _ = miner.search_once(0);
    let bench = per_launch as f64 / tb.elapsed().as_secs_f64();
    println!(
        "  Rate: {:.2} MK/s ({} groups x 256 threads x {} iters)",
        bench / 1e6,
        threadgroups,
        iters
    );

    let patlen = prefix5.len() + suffix5.len();
    let expected = 32f64.powi(patlen as i32);
    if estimate {
        let mean = expected / bench;
        println!(
            "  Difficulty: 32^{} = {:.4} keys; mean {:.4}s; 50%/90%/99% {:.3}s / {:.3}s / {:.3}s",
            patlen,
            expected,
            mean,
            mean * 2f64.ln(),
            mean * 10f64.ln(),
            mean * 100f64.ln()
        );
        return;
    }

    let mode = if fast { "FAST/rawtr" } else { "standard" };
    println!(
        "  Target: bc1p{}...{}  [{}]  (~{:.3} keys expected)\n",
        prefix.as_deref().unwrap_or(""),
        suffix.as_deref().unwrap_or(""),
        mode,
        expected
    );

    let t_search = Instant::now();
    let mut iter_base: u32 = miner.iters(); // benchmark launch already ran iter 0..iters
    let mut total = per_launch;
    let mut last = Instant::now();
    let hit;
    loop {
        if let Some(h) = miner.search_once(iter_base) {
            hit = h;
            break;
        }
        iter_base = iter_base.wrapping_add(miner.iters());
        total += per_launch;
        if iter_base >= 0xFFFF_0000 {
            eprintln!("\n  iteration space near stride boundary; restart for a fresh k0.");
            std::process::exit(1);
        }
        if last.elapsed().as_secs_f64() >= 1.0 {
            let el = t_search.elapsed().as_secs_f64();
            print!(
                "\r  Tried {} keys  {:.0}s  {:.2} MK/s   ",
                total,
                el,
                total as f64 / el / 1e6
            );
            use std::io::Write;
            let _ = std::io::stdout().flush();
            last = Instant::now();
        }
    }

    println!("\n");
    if fast {
        // in-process verify (rawtr)
        let d = derive_rawtr(&hit.secret);
        let ok = d.address == hit.address;
        println!("  ============ FOUND (FAST / rawtr) ============");
        println!("  Address              {}", hit.address);
        println!("  Output secret hex    {}", hex_be(&hit.secret));
        println!("  Output secret WIF    {}", wif(&hit.secret));
        println!("  Descriptor           rawtr({})", wif(&hit.secret));
        println!("  NOTE: rawtr() key-path key (Bitcoin Core rawtr descriptor).");
        println!("        NOT a BIP-386 tr() internal key; wallets expecting tr()");
        println!("        will not derive this address from this key.");
        println!("  VERIFY {}", if ok { "OK (noble/bitcoin reference matches)" } else { "MISMATCH!" });
        if !ok {
            std::process::exit(1);
        }
    } else {
        let d = derive_standard(&hit.secret);
        let ok = d.address == hit.address;
        println!("  ================= FOUND =================");
        println!("  Address              {}", hit.address);
        println!("  Internal privkey hex {}", hex_be(&hit.secret));
        println!("  Internal privkey WIF {}", wif(&hit.secret));
        println!("  Output secret (Q)    {}", hex_be(&d.output_secret));
        println!("  VERIFY {}", if ok { "OK (bitcoin reference matches)" } else { "MISMATCH!" });
        if !ok {
            std::process::exit(1);
        }
    }
    let el = t_search.elapsed().as_secs_f64();
    println!(
        "  Tried ~{} keys in {:.1}s ({:.2} MK/s)\n",
        total,
        el,
        total as f64 / el / 1e6
    );
}
