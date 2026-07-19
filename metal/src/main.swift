// tapvanity_metal — Bitcoin mainnet P2TR (taproot) vanity address miner, Metal.
//
// Search strategy: one random 256-bit base secret k0; thread i starts at
// P_i = (k0 + i*2^32)*G (computed by a GPU setup kernel via a precomputed
// window table); each GPU iteration steps P += G. On a hit the kernel reports
// (tid, global iteration, parity of y(P)); the host reconstructs
// k = k0 + tid*2^32 + iter (mod n), negating mod n if y(P) was odd, and the
// tweaked output secret k' + t mod n.

import Foundation
import Metal
import QuartzCore
import CryptoKit

let BECH32_CHARSET = Array("qpzry9x8gf2tvdw0s3jn54khce6mua7l")

// secp256k1 order n, little-endian UInt64 limbs
let N_LIMBS: [UInt64] = [0xBFD25E8CD0364141, 0xBAAEDCE6AF48A03B,
                         0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFF]

// MARK: - 256-bit scalar helpers (little-endian UInt64[4])

typealias U256 = [UInt64]  // always 4 limbs, LE

func u256FromBEBytes(_ b: [UInt8]) -> U256 {
    precondition(b.count == 32)
    var r = [UInt64](repeating: 0, count: 4)
    for i in 0..<4 {  // limb i (LE) = bytes 31-8i-7 .. 31-8i
        var v: UInt64 = 0
        for j in 0..<8 { v = (v << 8) | UInt64(b[24 - 8*i + j]) }
        r[i] = v
    }
    return r
}

func u256ToBEBytes(_ a: U256) -> [UInt8] {
    var b = [UInt8](repeating: 0, count: 32)
    for i in 0..<4 {
        var v = a[i]
        for j in 0..<8 { b[31 - 8*i - j] = UInt8(v & 0xff); v >>= 8 }
    }
    return b
}

func u256Cmp(_ a: U256, _ b: U256) -> Int {
    for i in stride(from: 3, through: 0, by: -1) {
        if a[i] < b[i] { return -1 }
        if a[i] > b[i] { return 1 }
    }
    return 0
}

func u256Add(_ a: U256, _ b: U256) -> (U256, Bool) {
    var r = [UInt64](repeating: 0, count: 4)
    var carry: UInt64 = 0
    for i in 0..<4 {
        let (s1, o1) = a[i].addingReportingOverflow(b[i])
        let (s2, o2) = s1.addingReportingOverflow(carry)
        r[i] = s2
        carry = (o1 ? 1 : 0) + (o2 ? 1 : 0)
    }
    return (r, carry != 0)
}

func u256Sub(_ a: U256, _ b: U256) -> (U256, Bool) {
    var r = [UInt64](repeating: 0, count: 4)
    var borrow: UInt64 = 0
    for i in 0..<4 {
        let (d1, o1) = a[i].subtractingReportingOverflow(b[i])
        let (d2, o2) = d1.subtractingReportingOverflow(borrow)
        r[i] = d2
        borrow = (o1 ? 1 : 0) + (o2 ? 1 : 0)
    }
    return (r, borrow != 0)
}

/// (a + b) mod n, assuming a,b < n
func addModN(_ a: U256, _ b: U256) -> U256 {
    let (s, carry) = u256Add(a, b)
    if carry || u256Cmp(s, N_LIMBS) >= 0 {
        return u256Sub(s, N_LIMBS).0
    }
    return s
}

/// n - a  (a != 0, a < n)
func negModN(_ a: U256) -> U256 {
    return u256Sub(N_LIMBS, a).0
}

func modN(_ a: U256) -> U256 {
    if u256Cmp(a, N_LIMBS) >= 0 { return u256Sub(a, N_LIMBS).0 }
    return a
}

func hexBE(_ a: U256) -> String {
    u256ToBEBytes(a).map { String(format: "%02x", $0) }.joined()
}

// MARK: - WIF (compressed, mainnet)

let B58_ALPHABET = Array("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")

func base58Check(_ payload: [UInt8]) -> String {
    let checksum = Array(SHA256.hash(data: Data(SHA256.hash(data: Data(payload)))).prefix(4))
    var num = payload + checksum
    // big-int base58 via repeated division on byte array
    var zeros = 0
    for b in num { if b == 0 { zeros += 1 } else { break } }
    var out = [Character]()
    var start = 0
    while start < num.count {
        var rem = 0
        var allZero = true
        for i in start..<num.count {
            let cur = rem * 256 + Int(num[i])
            num[i] = UInt8(cur / 58)
            rem = cur % 58
            if num[i] != 0 { allZero = allZero && (i < start) }
        }
        out.append(B58_ALPHABET[rem])
        while start < num.count && num[start] == 0 { start += 1 }
        _ = allZero
    }
    return String(repeating: "1", count: zeros) + String(out.reversed())
}

func wif(_ key: U256) -> String {
    var payload: [UInt8] = [0x80]
    payload += u256ToBEBytes(key)
    payload.append(0x01)  // compressed
    return base58Check(payload)
}

// MARK: - CLI

func usage() {
    print("""

      tapvanity_metal — Bitcoin P2TR (bc1p) vanity address miner (Metal)

        --prefix <pat>      match  bc1p<pat>...
        --suffix <pat>      match  ...<pat>  (may reach into the checksum chars)
                            (--prefix and --suffix may be combined)
        --fast              FAST/rawtr mode: mine the OUTPUT key directly
                            (no TapTweak; result is a rawtr() descriptor key,
                            NOT importable as a normal tr() taproot wallet key)
        --estimate          benchmark, print difficulty/ETA for the pattern, exit
        --self-test         GPU vs known-vector sanity check, exit

      Performance:
        --threadgroups <N>  threadgroup count            (default 1024)
        --threads <N>       threads per threadgroup      (default 256)
        --iters <N>         iterations per thread/launch (default 16)

      Charset (bech32): qpzry9x8gf2tvdw0s3jn54khce6mua7l
      (no '1', 'b', 'i', 'o'; lowercase only)
    """)
}

struct CLI {
    var prefix: String? = nil
    var suffix: String? = nil
    var estimate = false
    var selfTest = false
    var fast = false
    var threadgroups = 1024
    var threads = 256
    var iters = 16
}

func parseArgs() -> CLI {
    var c = CLI()
    var args = Array(CommandLine.arguments.dropFirst())
    func need(_ flag: String) -> String {
        guard !args.isEmpty else { print("Missing value for \(flag)"); exit(2) }
        return args.removeFirst()
    }
    while !args.isEmpty {
        let a = args.removeFirst()
        switch a {
        case "-h", "--help": usage(); exit(0)
        case "--prefix":   c.prefix = need(a)
        case "--suffix":   c.suffix = need(a)
        case "--estimate": c.estimate = true
        case "--fast":     c.fast = true
        case "--self-test": c.selfTest = true
        case "--threadgroups": c.threadgroups = Int(need(a)) ?? c.threadgroups
        case "--threads":  c.threads = Int(need(a)) ?? c.threads
        case "--iters":    c.iters = Int(need(a)) ?? c.iters
        default: print("Unknown argument: \(a)"); usage(); exit(2)
        }
    }
    return c
}

func patternTo5Bit(_ s: String) -> [UInt8] {
    var out = [UInt8]()
    for ch in s {
        guard let idx = BECH32_CHARSET.firstIndex(of: ch) else {
            print("Invalid bech32 character '\(ch)' in pattern '\(s)'.")
            print("Valid: \(String(BECH32_CHARSET))  (lowercase only; no 1/b/i/o)")
            exit(2)
        }
        out.append(UInt8(idx))
    }
    return out
}

var cli = parseArgs()
// kernel processes candidates in batches of BATCH (16)
cli.iters = max(16, (cli.iters / 16) * 16)
if cli.prefix == nil && cli.suffix == nil && !cli.selfTest {
    usage(); exit(2)
}
let prefix5 = cli.prefix.map(patternTo5Bit) ?? []
let suffix5 = cli.suffix.map(patternTo5Bit) ?? []
if prefix5.count > 32 || suffix5.count > 32 {
    print("Pattern too long (max 32 chars)."); exit(2)
}
// The address after 'bc1p' has 52 data chars + 6 checksum chars = 58.
if prefix5.count > 58 || suffix5.count > 58 { print("Pattern longer than address."); exit(2) }
// Data char #52 (0-based, right after bc1p it's index 51 of the prefix span)
// only takes values 'q' or 's' (4 pad bits); we don't special-case it, the
// kernel match is exact either way.

// MARK: - Metal setup

guard let device = MTLCreateSystemDefaultDevice() else { print("No Metal device."); exit(1) }

func locateMetallib() -> URL {
    let fm = FileManager.default
    let exeDir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
    let beside = exeDir.appendingPathComponent("default.metallib")
    if fm.fileExists(atPath: beside.path) { return beside }
    let underBuild = URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent("build/default.metallib")
    if fm.fileExists(atPath: underBuild.path) { return underBuild }
    return beside
}

let library: MTLLibrary
do { library = try device.makeLibrary(URL: locateMetallib()) }
catch { print("Could not load metallib: \(error)"); exit(1) }

guard
    let initFn   = library.makeFunction(name: "init_table_kernel"),
    let setupFn  = library.makeFunction(name: "setup_kernel"),
    let searchFn = library.makeFunction(name: "search_kernel"),
    let debugFn  = library.makeFunction(name: "debug_derive")
else { print("Missing kernel."); exit(1) }

let initPSO   = try! device.makeComputePipelineState(function: initFn)
let setupPSO  = try! device.makeComputePipelineState(function: setupFn)
let searchPSO = try! device.makeComputePipelineState(function: searchFn)
let debugPSO  = try! device.makeComputePipelineState(function: debugFn)
guard let queue = device.makeCommandQueue() else { print("No command queue."); exit(1) }

print()
print("  tapvanity_metal — P2TR vanity miner")
print("  Device : \(device.name)")

// MARK: - Buffers

let TABLE_ENTRIES = 32 * 256
let AFFINE_SIZE = 64
let tableBuf = device.makeBuffer(length: TABLE_ENTRIES * AFFINE_SIZE, options: .storageModePrivate)!

let totalThreads = cli.threadgroups * cli.threads
let stateBuf = device.makeBuffer(length: totalThreads * 96, options: .storageModePrivate)!
let k0Buf    = device.makeBuffer(length: 32, options: .storageModeShared)!
let cfgBuf   = device.makeBuffer(length: 4 + 4 + 4 + 4 + 32 + 32, options: .storageModeShared)!
let FOUND_SIZE = 4 + 4 + 8 + 4 + 4 + 32 + 64  // 120
let outBuf   = device.makeBuffer(length: FOUND_SIZE, options: .storageModeShared)!
memset(outBuf.contents(), 0, FOUND_SIZE)

// cfg
do {
    let p = cfgBuf.contents()
    var plen = Int32(prefix5.count), slen = Int32(suffix5.count)
    var fast = Int32(cli.fast ? 1 : 0)
    memcpy(p, &plen, 4)
    memcpy(p + 4, &slen, 4)
    memcpy(p + 8, &fast, 4)
    _ = prefix5.withUnsafeBytes { memcpy(p + 16, $0.baseAddress!, prefix5.count) }
    _ = suffix5.withUnsafeBytes { memcpy(p + 48, $0.baseAddress!, suffix5.count) }
}

// MARK: - Precompute window table

let tStart0 = Date()
do {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(initPSO)
    enc.setBuffer(tableBuf, offset: 0, index: 0)
    enc.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
}
print(String(format: "  Window table (32x256) ready in %.2fs", -tStart0.timeIntervalSinceNow))

// MARK: - GPU self-test against a known vector

// k = 1: internal pubkey = G.x, well-known taproot address.
func gpuDeriveAddress(scalarLE: [UInt32]) -> String {
    let sBuf = device.makeBuffer(length: 32, options: .storageModeShared)!
    _ = scalarLE.withUnsafeBytes { memcpy(sBuf.contents(), $0.baseAddress!, 32) }
    let aBuf = device.makeBuffer(length: 64, options: .storageModeShared)!
    let xBuf = device.makeBuffer(length: 96, options: .storageModeShared)!
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(debugPSO)
    enc.setBuffer(sBuf, offset: 0, index: 0)
    enc.setBuffer(aBuf, offset: 0, index: 1)
    enc.setBuffer(xBuf, offset: 0, index: 2)
    enc.setBuffer(tableBuf, offset: 0, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    return String(cString: aBuf.contents().assumingMemoryBound(to: CChar.self))
}

// Known vector: secret key 1 -> internal key = G.x
// P2TR address for internal key G.x (BIP-341 key-path, no script tree):
let KNOWN_K1_ADDR = "bc1pmfr3p9j00pfxjh0zmgp99y8zftmd3s5pmedqhyptwy6lm87hf5sspknck9"
do {
    let got = gpuDeriveAddress(scalarLE: [1,0,0,0,0,0,0,0])
    if got != KNOWN_K1_ADDR {
        print("  GPU SELF-TEST FAILED for k=1:")
        print("    got      \(got)")
        print("    expected \(KNOWN_K1_ADDR)")
        exit(1)
    }
    print("  GPU self-test (k=1 known vector): OK")
}
if cli.selfTest {
    print("  Self-test passed."); exit(0)
}

// MARK: - k0 and setup

func randomK0() -> U256 {
    var bytes = [UInt8](repeating: 0, count: 32)
    let res = SecRandomCopyBytes(kSecRandomDefault, 32, &bytes)
    if res != errSecSuccess { fatalError("SecRandomCopyBytes failed") }
    var k = u256FromBEBytes(bytes)
    k = modN(k)
    if k.allSatisfy({ $0 == 0 }) { k[0] = 1 }
    return k
}

let k0 = randomK0()
do {
    // write k0 as 8 LE u32 limbs
    var limbs = [UInt32](repeating: 0, count: 8)
    for i in 0..<4 {
        limbs[2*i]   = UInt32(truncatingIfNeeded: k0[i])
        limbs[2*i+1] = UInt32(truncatingIfNeeded: k0[i] >> 32)
    }
    _ = limbs.withUnsafeBytes { memcpy(k0Buf.contents(), $0.baseAddress!, 32) }

    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(setupPSO)
    enc.setBuffer(k0Buf, offset: 0, index: 0)
    enc.setBuffer(stateBuf, offset: 0, index: 1)
    enc.setBuffer(tableBuf, offset: 0, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: cli.threadgroups, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: cli.threads, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
}
print("  Setup: \(totalThreads) threads seeded (stride 2^32)")

// MARK: - Search loop

nonisolated(unsafe) var g_abort: Int32 = 0
let abortHandler: @convention(c) (Int32) -> Void = { _ in g_abort = 1 }
signal(SIGINT, abortHandler)
signal(SIGTERM, abortHandler)

func launch(iterBase: UInt64, iters: Int) -> Double {
    var ib = iterBase
    var it = Int32(iters)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(searchPSO)
    enc.setBytes(&ib, length: 8, index: 0)
    enc.setBytes(&it, length: 4, index: 1)
    enc.setBuffer(cfgBuf, offset: 0, index: 2)
    enc.setBuffer(outBuf, offset: 0, index: 3)
    enc.setBuffer(stateBuf, offset: 0, index: 4)
    enc.setBuffer(tableBuf, offset: 0, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: cli.threadgroups, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: cli.threads, height: 1, depth: 1))
    enc.endEncoding()
    let t0 = CACurrentMediaTime()
    cb.commit(); cb.waitUntilCompleted()
    return CACurrentMediaTime() - t0
}

let perLaunch = UInt64(totalThreads) * UInt64(cli.iters)

// benchmark
let benchSec = launch(iterBase: 0, iters: cli.iters)
var iterBase: UInt64 = UInt64(cli.iters) // benchmark launch did real work; keep it
let benchKps = Double(perLaunch) / benchSec
print(String(format: "  Rate  : %.2f MK/s (%d tg x %d th x %d iters)", benchKps / 1e6,
             cli.threadgroups, cli.threads, cli.iters))

let patLen = prefix5.count + suffix5.count
let expectedKeys = pow(32.0, Double(patLen))
if cli.estimate {
    let mean = expectedKeys / benchKps
    print(String(format: """
        Pattern:        prefix '%@' suffix '%@'  (%d chars)
        Difficulty:     32^%d = %.4g expected keys
        Mean time:      %.4g s
        50%% / 90%% / 99%%: %.4g s / %.4g s / %.4g s
      """, cli.prefix ?? "", cli.suffix ?? "", patLen, patLen, expectedKeys,
      mean, mean * log(2.0), mean * log(10.0), mean * log(100.0)))
    exit(0)
}

print("  Target: bc1p\(cli.prefix ?? "")...\(cli.suffix ?? "")  (~\(String(format: "%.3g", expectedKeys)) keys expected)")
print()

var totalKeys: UInt64 = perLaunch
let tSearch = Date()
var lastPrint = Date(timeIntervalSince1970: 0)

func foundFlag() -> UInt32 { outBuf.contents().load(as: UInt32.self) }

while foundFlag() == 0 {
    if g_abort != 0 { print("\n  Aborted."); exit(130) }
    let sec = launch(iterBase: iterBase, iters: cli.iters)
    iterBase += UInt64(cli.iters)
    totalKeys += perLaunch
    if UInt64(cli.iters) >= (1 << 31) || iterBase >= 0xFFFF0000 {
        print("\n  Iteration space near stride boundary; restart with a new k0.")
        exit(1)
    }
    let now = Date()
    if now.timeIntervalSince(lastPrint) >= 2.0 {
        let elapsed = now.timeIntervalSince(tSearch)
        let avg = Double(totalKeys) / max(elapsed, 1e-9)
        let inst = Double(perLaunch) / max(sec, 1e-9)
        print(String(format: "\r  Tried %llu keys  %5.0fs  avg %6.2f MK/s  inst %6.2f MK/s",
                     totalKeys, elapsed, avg / 1e6, inst / 1e6), terminator: "")
        fflush(stdout)
        lastPrint = now
    }
}

// MARK: - Reconstruct and report

let raw = outBuf.contents()
let tid  = raw.load(fromByteOffset: 4, as: UInt32.self)
let iter = raw.load(fromByteOffset: 8, as: UInt64.self)
let parity = raw.load(fromByteOffset: 16, as: UInt32.self)
var tweakBytes = [UInt8](repeating: 0, count: 32)
memcpy(&tweakBytes, raw + 24, 32)
let addr = String(cString: (raw + 56).assumingMemoryBound(to: CChar.self))

// k = k0 + tid*2^32 + iter  (mod n)
var offset: U256 = [0, 0, 0, 0]
// tid*2^32 + iter as a 128-bit value spread over limbs 0..1
let lo = UInt64(iter & 0xFFFFFFFF) | (UInt64(tid) << 32)   // careful: iter < 2^32 enforced
// iter fits in 32 bits (checked above); tid*2^32 occupies bits 32..63+
let iterLow = iter & 0xFFFF_FFFF
offset[0] = iterLow | (UInt64(tid) << 32)
offset[1] = UInt64(tid) >> 32
_ = lo
var k = addModN(k0, modN(offset))
if parity == 1 { k = negModN(k) }   // even-y convention for the walked point

let elapsed = Date().timeIntervalSince(tSearch)
print("\n")
if cli.fast {
    // FAST/rawtr mode: k IS the output secret. No TapTweak was applied —
    // the address commits to the raw key (rawtr descriptor), not tr().
    print("  ============ FOUND (FAST / rawtr) ============")
    print("  Address              \(addr)")
    print("  Output secret hex    \(hexBE(k))")
    print("  Output secret WIF    \(wif(k))")
    print("  Descriptor           rawtr(\(wif(k)))")
    print("  NOTE: FAST mode key-path key. Import as a rawtr() descriptor")
    print("        (Bitcoin Core). It is NOT a tr() internal key; most")
    print("        wallets that expect BIP-386 tr() will not derive this")
    print("        address from it.")
} else {
    let tweak = u256FromBEBytes(tweakBytes)
    let outSecret = addModN(k, modN(tweak))
    print("  ================= FOUND =================")
    print("  Address              \(addr)")
    print("  Internal privkey hex \(hexBE(k))")
    print("  Internal privkey WIF \(wif(k))")
    print("  Tweak (TapTweak)     \(tweakBytes.map { String(format: "%02x", $0) }.joined())")
    print("  Output secret (Q)    \(hexBE(addModN(k, modN(u256FromBEBytes(tweakBytes)))))")
    _ = outSecret
}
print(String(format: "  Tried ~%llu keys in %.1fs (%.2f MK/s)", totalKeys, elapsed,
             Double(totalKeys) / max(elapsed, 1e-9) / 1e6))
print()
// machine-readable lines for the test harness
print("MODE \(cli.fast ? "fast" : "standard")")
print("ADDR \(addr)")
print("PRIV \(hexBE(k))")
print("WIF \(wif(k))")
if !cli.fast {
    print("TWEAKED \(hexBE(addModN(k, modN(u256FromBEBytes(tweakBytes)))))")
}

// Optional immediate CPU verification via the Rust reference, if built.
let exeDir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
let verifier = exeDir.deletingLastPathComponent()
    .appendingPathComponent("verify/target/release/tapverify").path
if FileManager.default.isExecutableFile(atPath: verifier) {
    let proc = Process()
    proc.executableURL = URL(fileURLWithPath: verifier)
    proc.arguments = cli.fast ? [hexBE(k), "rawtr"] : [hexBE(k)]
    let pipe = Pipe()
    proc.standardOutput = pipe
    try? proc.run()
    proc.waitUntilExit()
    let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    let derived = out.split(separator: "\n").first(where: { $0.hasPrefix("address ") })?
        .dropFirst("address ".count)
    if let d = derived {
        if String(d) == addr {
            print("VERIFY OK (rust reference derives the same address)")
        } else {
            print("VERIFY MISMATCH! rust says \(d)")
            exit(1)
        }
    }
} else {
    print("VERIFY SKIPPED (build verify/ with cargo for automatic cross-check)")
}
