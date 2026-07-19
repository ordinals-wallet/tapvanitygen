// tap_vanity.wgsl — cross-platform P2TR (taproot) vanity miner kernel.
//
// Port of the verified Metal shader (../metal/shaders/tap_vanity.metal) to
// WGSL. WGSL has no 64-bit integers, so field elements are 8 x u32 limbs and
// all widening multiplies are emulated via 16-bit half-splits:
//   mul32(a,b) -> (lo,hi)  where a*b = hi*2^32 + lo.
//
// Two entry points share this file (selected by the Rust host via naga/wgpu
// pipeline creation with the same module):
//   init_table   — builds the 32x256 window table  (grid 32*256)
//   setup        — per-thread start point P_i = (k0 + i*2^32)*G
//   search       — steps P += G, checks pattern (standard or fast/rawtr)
//
// secp256k1: p = 2^256 - 2^32 - 977.  Fold constant 2^256 ≡ 0x1000003D1.

// ------------------------------------------------------------------ widening mul
// returns vec2(lo, hi) = a*b
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xffffu;
    let ah = a >> 16u;
    let bl = b & 0xffffu;
    let bh = b >> 16u;
    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;
    // mid = lh + hl (may carry into bit 32)
    let mid = lh + hl;
    let midCarry = select(0u, 0x10000u, mid < lh); // carry out of 32-bit add, *2^16
    // lo = ll + (mid << 16)
    let midLo = mid << 16u;
    var lo = ll + midLo;
    var carry = select(0u, 1u, lo < ll);
    // hi = hh + (mid >> 16) + midCarry + carry
    let hi = hh + (mid >> 16u) + midCarry + carry;
    return vec2<u32>(lo, hi);
}

// add with carry: returns (sum, carryOut) for a+b+cin
fn addc(a: u32, b: u32, cin: u32) -> vec2<u32> {
    let s1 = a + b;
    let c1 = select(0u, 1u, s1 < a);
    let s2 = s1 + cin;
    let c2 = select(0u, 1u, s2 < s1);
    return vec2<u32>(s2, c1 + c2);
}

// subtract with borrow: returns (diff, borrowOut) for a-b-bin
fn subb(a: u32, b: u32, bin: u32) -> vec2<u32> {
    let d1 = a - b;
    let b1 = select(0u, 1u, a < b);
    let d2 = d1 - bin;
    let b2 = select(0u, 1u, d1 < bin);
    return vec2<u32>(d2, b1 + b2);
}

// ------------------------------------------------------------------ field element
// fe = array<u32,8>, little-endian limbs, value < 2^256 (may be >= p)

const FOLD_LO: u32 = 977u; // 2^256 mod p = 2^32 + 977

const P0: u32 = 0xFFFFFC2Fu; const P1: u32 = 0xFFFFFFFEu;
// P2..P7 are all 0xffffffff

// generator G
const GX = array<u32,8>(0x16F81798u,0x59F2815Bu,0x2DCE28D9u,0x029BFCDBu,
                        0xCE870B07u,0x55A06295u,0xF9DCBBACu,0x79BE667Eu);
const GY = array<u32,8>(0xFB10D4B8u,0x9C47D08Fu,0xA6855419u,0xFD17B448u,
                        0x0E1108A8u,0x5DA4FBFCu,0x26A3C465u,0x483ADA77u);

// curve order n
const N = array<u32,8>(0xD0364141u,0xBFD25E8Cu,0xAF48A03Bu,0xBAAEDCE6u,
                       0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu);

alias Fe = array<u32,8>;

fn fe_zero() -> Fe { return Fe(0u,0u,0u,0u,0u,0u,0u,0u); }

// r = a + b (folds carry-out)
fn fe_add(pa: Fe, pb: Fe) -> Fe {
    var a = pa; var b = pb;
    var r: Fe;
    var c = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let s = addc(a[i], b[i], c);
        r[i] = s.x; c = s.y;
    }
    // fold carry c (0 or 1): add c*(2^32 + 977) at limb0/limb1
    var cc = c;
    loop {
        if (cc == 0u) { break; }
        let add0 = mul32(cc, FOLD_LO); // cc*977 (< 2^32, hi=0 for cc small)
        var carry = 0u;
        let t0 = addc(r[0], add0.x, 0u); r[0] = t0.x; carry = t0.y + add0.y;
        let t1 = addc(r[1], cc, carry); r[1] = t1.x; carry = t1.y;
        for (var i = 2u; i < 8u; i = i + 1u) {
            if (carry == 0u) { break; }
            let t = addc(r[i], 0u, carry); r[i] = t.x; carry = t.y;
        }
        cc = carry;
    }
    return r;
}

// r = a - b (folds borrow-out by subtracting the fold constant)
fn fe_sub(pa: Fe, pb: Fe) -> Fe {
    var a = pa; var b = pb;
    var r: Fe;
    var bor = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let s = subb(a[i], b[i], bor);
        r[i] = s.x; bor = s.y;
    }
    var bb = bor;
    loop {
        if (bb == 0u) { break; }
        // subtract 0x1000003D1 (once per wrap)
        var borrow = 0u;
        let t0 = subb(r[0], FOLD_LO, 0u); r[0] = t0.x; borrow = t0.y;
        let t1 = subb(r[1], 1u, borrow); r[1] = t1.x; borrow = t1.y;
        for (var i = 2u; i < 8u; i = i + 1u) {
            if (borrow == 0u) { break; }
            let t = subb(r[i], 0u, borrow); r[i] = t.x; borrow = t.y;
        }
        bb = borrow;
    }
    return r;
}

// reduce a 512-bit product t[16] into r[8] (< 2^256, non-canonical)
fn fe_reduce512(pt: array<u32,16>) -> Fe {
    var t = pt;
    // s = t_lo + t_hi*977 + (t_hi << 32)
    var s: Fe;
    var c = 0u;
    // limb 0: t[0] + t[8]*977
    {
        let m = mul32(t[8], FOLD_LO);
        let a0 = addc(t[0], m.x, 0u);
        s[0] = a0.x;
        c = a0.y + m.y;
    }
    for (var i = 1u; i < 8u; i = i + 1u) {
        let m = mul32(t[8u + i], FOLD_LO);
        // s[i] = t[i] + m.lo + t[7+i] + carry-chain
        var acc = addc(t[i], m.x, 0u);
        var carry = acc.y;
        let a2 = addc(acc.x, t[7u + i], 0u);
        acc.x = a2.x; carry = carry + a2.y;
        let a3 = addc(acc.x, c, 0u);
        s[i] = a3.x; carry = carry + a3.y + m.y;
        c = carry;
    }
    // c now holds overflow above 2^256 (plus t[15] contribution already folded
    // via t[7+i] when i==8? no — t[15] used at i=7 as t[14]; t[15] pending)
    // Add t[15] into c.
    let cAdd = addc(c, t[15], 0u);
    c = cAdd.x; // ignore c.y: bounded
    // fold c*(2^32 + 977) into s
    var r: Fe;
    let d = mul32(c, FOLD_LO); // c*977
    var carry = 0u;
    let r0 = addc(s[0], d.x, 0u); r[0] = r0.x; carry = r0.y + d.y;
    let r1 = addc(s[1], c, carry); r[1] = r1.x; carry = r1.y;
    for (var i = 2u; i < 8u; i = i + 1u) {
        let t2 = addc(s[i], 0u, carry); r[i] = t2.x; carry = t2.y;
    }
    // final possible carry -> one more fold (cannot recarry)
    if (carry != 0u) {
        let e = mul32(carry, FOLD_LO);
        var cy = 0u;
        let e0 = addc(r[0], e.x, 0u); r[0] = e0.x; cy = e0.y + e.y;
        let e1 = addc(r[1], carry, cy); r[1] = e1.x; cy = e1.y;
        for (var i = 2u; i < 8u; i = i + 1u) {
            if (cy == 0u) { break; }
            let tt = addc(r[i], 0u, cy); r[i] = tt.x; cy = tt.y;
        }
    }
    return r;
}

fn fe_mul(pa: Fe, pb: Fe) -> Fe {
    var a = pa; var b = pb;
    var t: array<u32,16>;
    for (var i = 0u; i < 16u; i = i + 1u) { t[i] = 0u; }
    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let m = mul32(a[i], b[j]);
            // t[i+j] += m.lo + carry ; carry = m.hi + carryout
            let s1 = addc(t[i + j], m.x, 0u);
            let s2 = addc(s1.x, carry, 0u);
            t[i + j] = s2.x;
            carry = m.y + s1.y + s2.y;
        }
        t[i + 8u] = carry;
    }
    return fe_reduce512(t);
}

fn fe_sqr(a: Fe) -> Fe { return fe_mul(a, a); }

fn fe_is_ge_p(a: Fe) -> bool {
    // compare a >= p
    if (a[7] != 0xffffffffu) { return a[7] > 0xffffffffu; }
    if (a[6] != 0xffffffffu) { return a[6] > 0xffffffffu; }
    if (a[5] != 0xffffffffu) { return a[5] > 0xffffffffu; }
    if (a[4] != 0xffffffffu) { return a[4] > 0xffffffffu; }
    if (a[3] != 0xffffffffu) { return a[3] > 0xffffffffu; }
    if (a[2] != 0xffffffffu) { return a[2] > 0xffffffffu; }
    if (a[1] != P1) { return a[1] > P1; }
    return a[0] >= P0;
}

fn fe_normalize(pa: Fe) -> Fe {
    if (!fe_is_ge_p(pa)) { return pa; }
    var a = pa;
    var r: Fe;
    var bor = 0u;
    let s0 = subb(a[0], P0, 0u); r[0] = s0.x; bor = s0.y;
    let s1 = subb(a[1], P1, bor); r[1] = s1.x; bor = s1.y;
    for (var i = 2u; i < 8u; i = i + 1u) {
        let s = subb(a[i], 0xffffffffu, bor); r[i] = s.x; bor = s.y;
    }
    return r;
}

fn fe_is_zero_norm(a: Fe) -> bool {
    return (a[0]|a[1]|a[2]|a[3]|a[4]|a[5]|a[6]|a[7]) == 0u;
}

// p - a  (a normalized, nonzero)
fn fe_neg_norm(pa: Fe) -> Fe {
    var a = pa;
    var r: Fe;
    var bor = 0u;
    let s0 = subb(P0, a[0], 0u); r[0] = s0.x; bor = s0.y;
    let s1 = subb(P1, a[1], bor); r[1] = s1.x; bor = s1.y;
    for (var i = 2u; i < 8u; i = i + 1u) {
        let s = subb(0xffffffffu, a[i], bor); r[i] = s.x; bor = s.y;
    }
    return r;
}

// inversion via libsecp256k1 addition chain (a^(p-2))
fn fe_inv(a: Fe) -> Fe {
    var x2 = fe_mul(fe_sqr(a), a);
    var x3 = fe_mul(fe_sqr(x2), a);
    var x6 = x3;
    for (var j = 0; j < 3; j = j + 1) { x6 = fe_sqr(x6); }
    x6 = fe_mul(x6, x3);
    var x9 = x6;
    for (var j = 0; j < 3; j = j + 1) { x9 = fe_sqr(x9); }
    x9 = fe_mul(x9, x3);
    var x11 = x9;
    for (var j = 0; j < 2; j = j + 1) { x11 = fe_sqr(x11); }
    x11 = fe_mul(x11, x2);
    var x22 = x11;
    for (var j = 0; j < 11; j = j + 1) { x22 = fe_sqr(x22); }
    x22 = fe_mul(x22, x11);
    var x44 = x22;
    for (var j = 0; j < 22; j = j + 1) { x44 = fe_sqr(x44); }
    x44 = fe_mul(x44, x22);
    var x88 = x44;
    for (var j = 0; j < 44; j = j + 1) { x88 = fe_sqr(x88); }
    x88 = fe_mul(x88, x44);
    var x176 = x88;
    for (var j = 0; j < 88; j = j + 1) { x176 = fe_sqr(x176); }
    x176 = fe_mul(x176, x88);
    var x220 = x176;
    for (var j = 0; j < 44; j = j + 1) { x220 = fe_sqr(x220); }
    x220 = fe_mul(x220, x44);
    var x223 = x220;
    for (var j = 0; j < 3; j = j + 1) { x223 = fe_sqr(x223); }
    x223 = fe_mul(x223, x3);
    var t1 = x223;
    for (var j = 0; j < 23; j = j + 1) { t1 = fe_sqr(t1); }
    t1 = fe_mul(t1, x22);
    for (var j = 0; j < 5; j = j + 1) { t1 = fe_sqr(t1); }
    t1 = fe_mul(t1, a);
    for (var j = 0; j < 3; j = j + 1) { t1 = fe_sqr(t1); }
    t1 = fe_mul(t1, x2);
    for (var j = 0; j < 2; j = j + 1) { t1 = fe_sqr(t1); }
    return fe_mul(t1, a);
}

// ------------------------------------------------------------------ point ops
// Jacobian: X,Y,Z. Mixed add P(jac) += Q(affine). inf tracks infinity.

struct Jac { X: Fe, Y: Fe, Z: Fe, inf: u32 };

fn jac_double(qx: Fe, qy: Fe) -> Jac {
    // double of affine (qx,qy,1)
    let A = fe_sqr(qx);
    let B = fe_sqr(qy);
    let C = fe_sqr(B);
    var D = fe_add(qx, B); D = fe_sqr(D);
    D = fe_sub(D, A); D = fe_sub(D, C); D = fe_add(D, D);
    var E = fe_add(A, A); E = fe_add(E, A);
    let F = fe_sqr(E);
    var X = fe_sub(F, D); X = fe_sub(X, D);
    var t = fe_sub(D, X);
    var Y = fe_mul(E, t);
    var C8 = fe_add(C, C); C8 = fe_add(C8, C8); C8 = fe_add(C8, C8);
    Y = fe_sub(Y, C8);
    let Z = fe_add(qy, qy);
    return Jac(X, Y, Z, 0u);
}

fn jac_madd(p: Jac, qx: Fe, qy: Fe) -> Jac {
    if (p.inf != 0u) {
        return Jac(qx, qy, Fe(1u,0u,0u,0u,0u,0u,0u,0u), 0u);
    }
    let z1z1 = fe_sqr(p.Z);
    let u2 = fe_mul(qx, z1z1);
    var s2 = fe_mul(p.Z, z1z1);
    s2 = fe_mul(s2, qy);
    let h = fe_sub(u2, p.X);
    let rr = fe_sub(s2, p.Y);

    let hn = fe_normalize(h);
    if (fe_is_zero_norm(hn)) {
        let rn = fe_normalize(rr);
        if (fe_is_zero_norm(rn)) {
            return jac_double(qx, qy);
        }
        return Jac(p.X, p.Y, p.Z, 1u); // P == -Q -> infinity
    }
    let h2 = fe_sqr(h);
    let h3 = fe_mul(h2, h);
    let v = fe_mul(p.X, h2);
    var x3 = fe_sqr(rr);
    x3 = fe_sub(x3, h3);
    x3 = fe_sub(x3, v);
    x3 = fe_sub(x3, v);
    let z3 = fe_mul(p.Z, h);
    var y3 = fe_sub(v, x3);
    y3 = fe_mul(y3, rr);
    let yh3 = fe_mul(p.Y, h3);
    y3 = fe_sub(y3, yh3);
    return Jac(x3, y3, z3, 0u);
}

fn jac_to_affine_x(j: Jac) -> Fe {
    let zi = fe_inv(j.Z);
    let zi2 = fe_sqr(zi);
    return fe_normalize(fe_mul(j.X, zi2));
}


// full affine (x,y) both normalized
struct Aff { x: Fe, y: Fe };
fn jac_to_affine(j: Jac) -> Aff {
    let zi = fe_inv(j.Z);
    let zi2 = fe_sqr(zi);
    let zi3 = fe_mul(zi2, zi);
    return Aff(fe_normalize(fe_mul(j.X, zi2)), fe_normalize(fe_mul(j.Y, zi3)));
}

// window table lives in a storage buffer as 16 u32 per entry (x[8], y[8])
@group(0) @binding(0) var<storage, read_write> table: array<u32>;

fn table_x(idx: u32) -> Fe {
    var r: Fe;
    let base = idx * 16u;
    for (var i = 0u; i < 8u; i = i + 1u) { r[i] = table[base + i]; }
    return r;
}
fn table_y(idx: u32) -> Fe {
    var r: Fe;
    let base = idx * 16u + 8u;
    for (var i = 0u; i < 8u; i = i + 1u) { r[i] = table[base + i]; }
    return r;
}

// scalar*G via 8-bit window table: table[b*256+v] = (v*256^b)*G
fn scalarmult_base(ps: Fe) -> Jac {
    var s = ps;
    var p = Jac(fe_zero(), fe_zero(), fe_zero(), 1u);
    for (var byte = 0u; byte < 32u; byte = byte + 1u) {
        let limb = s[byte / 4u];
        let j = (limb >> ((byte % 4u) * 8u)) & 0xffu;
        if (j == 0u) { continue; }
        let idx = byte * 256u + j;
        p = jac_madd(p, table_x(idx), table_y(idx));
    }
    return p;
}

// ------------------------------------------------------------------ SHA-256
const SK = array<u32,64>(
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u);

// midstate after compressing SHA256("TapTweak")||SHA256("TapTweak")
const TWEAK_MID = array<u32,8>(
    0xd129a2f3u,0x701c655du,0x6583b6c3u,0xb9419727u,
    0x95f4e232u,0x94fd54f4u,0xa2ae8d85u,0x47ca590bu);

fn rotr(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }

// TapTweak of a normalized x coordinate (px LE limbs) -> tweak scalar (LE limbs)
fn taptweak(ppx: Fe) -> Fe {
    var px = ppx;
    var tm = TWEAK_MID; var sk = SK;
    var h: array<u32,8>;
    for (var i = 0u; i < 8u; i = i + 1u) { h[i] = tm[i]; }
    var w: array<u32,16>;
    // message block: px big-endian words, 0x80, zeros, bitlen 768
    for (var i = 0u; i < 8u; i = i + 1u) { w[i] = px[7u - i]; }
    w[8] = 0x80000000u;
    for (var i = 9u; i < 15u; i = i + 1u) { w[i] = 0u; }
    w[15] = 768u;

    var a=h[0]; var b=h[1]; var c=h[2]; var d=h[3];
    var e=h[4]; var f=h[5]; var g=h[6]; var hv=h[7];
    for (var i = 0u; i < 64u; i = i + 1u) {
        if (i >= 16u) {
            let v15 = w[(i + 1u) & 15u];
            let v2  = w[(i + 14u) & 15u];
            let s0 = rotr(v15,7u) ^ rotr(v15,18u) ^ (v15 >> 3u);
            let s1 = rotr(v2,17u) ^ rotr(v2,19u) ^ (v2 >> 10u);
            w[i & 15u] = w[i & 15u] + s0 + w[(i + 9u) & 15u] + s1;
        }
        let wi = w[i & 15u];
        let S1 = rotr(e,6u) ^ rotr(e,11u) ^ rotr(e,25u);
        let ch = (e & f) ^ (~e & g);
        let t1 = hv + S1 + ch + sk[i] + wi;
        let S0 = rotr(a,2u) ^ rotr(a,13u) ^ rotr(a,22u);
        let mj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = S0 + mj;
        hv=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]=h[0]+a; h[1]=h[1]+b; h[2]=h[2]+c; h[3]=h[3]+d;
    h[4]=h[4]+e; h[5]=h[5]+f; h[6]=h[6]+g; h[7]=h[7]+hv;
    // digest BE bytes -> LE limbs: limb i = h[7-i]
    var t: Fe;
    for (var i = 0u; i < 8u; i = i + 1u) { t[i] = h[7u - i]; }
    return t;
}

// ------------------------------------------------------------------ bech32m
const BECH32_GEN = array<u32,5>(0x3b6a57b2u,0x26508e6du,0x1ea119fau,0x3d4233ddu,0x2a1462b3u);
const HRP_EXP = array<u32,5>(3u,3u,0u,2u,3u);
const BECH32M_CONST: u32 = 0x2bc830a3u;

// produce 59 5-bit values after "bc1": v[0]=witver(1), v[1..52]=data, v[53..58]=checksum
fn bech32m_values(pqx: Fe) -> array<u32,59> {
    var qx = pqx;
    var hexp = HRP_EXP; var gen = BECH32_GEN;
    // big-endian bytes of qx
    var bytes: array<u32,32>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let limb = qx[7u - i];
        bytes[i*4u]   = (limb >> 24u) & 0xffu;
        bytes[i*4u+1u] = (limb >> 16u) & 0xffu;
        bytes[i*4u+2u] = (limb >> 8u) & 0xffu;
        bytes[i*4u+3u] = limb & 0xffu;
    }
    var v: array<u32,59>;
    v[0] = 1u;
    var acc = 0u;
    var bits = 0u;
    var out = 1u;
    for (var i = 0u; i < 32u; i = i + 1u) {
        acc = (acc << 8u) | bytes[i];
        bits = bits + 8u;
        loop {
            if (bits < 5u) { break; }
            bits = bits - 5u;
            v[out] = (acc >> bits) & 31u;
            out = out + 1u;
        }
    }
    if (bits > 0u) { v[out] = (acc << (5u - bits)) & 31u; out = out + 1u; }
    // checksum
    var chk = 1u;
    for (var i = 0u; i < 5u; i = i + 1u) {
        let top = chk >> 25u;
        chk = ((chk & 0x1ffffffu) << 5u) ^ hexp[i];
        for (var k = 0u; k < 5u; k = k + 1u) { if (((top >> k) & 1u) != 0u) { chk = chk ^ gen[k]; } }
    }
    for (var i = 0u; i < 53u; i = i + 1u) {
        let top = chk >> 25u;
        chk = ((chk & 0x1ffffffu) << 5u) ^ v[i];
        for (var k = 0u; k < 5u; k = k + 1u) { if (((top >> k) & 1u) != 0u) { chk = chk ^ gen[k]; } }
    }
    for (var i = 0u; i < 6u; i = i + 1u) {
        let top = chk >> 25u;
        chk = ((chk & 0x1ffffffu) << 5u);
        for (var k = 0u; k < 5u; k = k + 1u) { if (((top >> k) & 1u) != 0u) { chk = chk ^ gen[k]; } }
    }
    chk = chk ^ BECH32M_CONST;
    for (var i = 0u; i < 6u; i = i + 1u) {
        v[53u + i] = (chk >> (5u * (5u - i))) & 31u;
    }
    return v;
}

// ------------------------------------------------------------------ config / IO
struct Cfg {
    prefix_len: u32,
    suffix_len: u32,
    fast: u32,
    total_threads: u32,
    prefix: array<u32,32>,   // 5-bit values (one per u32 for simple layout)
    suffix: array<u32,32>,
};
@group(0) @binding(1) var<storage, read> cfg: Cfg;

// found output
struct Found {
    flag: atomic<u32>,
    tid: u32,
    iter: u32,
    parity: u32,
    tweak: array<u32,8>,   // LE limbs
    qx: array<u32,8>,      // LE limbs of Q.x (address x-only)
};
@group(0) @binding(2) var<storage, read_write> found: Found;

// per-thread persistent state (Jacobian point)
@group(0) @binding(3) var<storage, read_write> state: array<u32>; // 24 u32 per thread: X,Y,Z

// scalars / params
struct Params {
    k0: array<u32,8>,
    iter_base: u32,
    iters: u32,
    _p0: u32,
    _p1: u32,
};
@group(0) @binding(4) var<storage, read> params: Params;

fn load_state(tid: u32) -> Jac {
    var j: Jac;
    let base = tid * 24u;
    for (var i = 0u; i < 8u; i = i + 1u) { j.X[i] = state[base + i]; }
    for (var i = 0u; i < 8u; i = i + 1u) { j.Y[i] = state[base + 8u + i]; }
    for (var i = 0u; i < 8u; i = i + 1u) { j.Z[i] = state[base + 16u + i]; }
    j.inf = 0u;
    return j;
}
fn store_state(tid: u32, pj: Jac) {
    var j = pj;
    let base = tid * 24u;
    for (var i = 0u; i < 8u; i = i + 1u) { state[base + i] = j.X[i]; }
    for (var i = 0u; i < 8u; i = i + 1u) { state[base + 8u + i] = j.Y[i]; }
    for (var i = 0u; i < 8u; i = i + 1u) { state[base + 16u + i] = j.Z[i]; }
}

// ------------------------------------------------------------------ kernels

@compute @workgroup_size(256)
fn init_table(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= 32u * 256u) { return; }
    let b = id / 256u;      // byte position 0..31
    let jv = id % 256u;     // byte value 0..255
    let idx = id;
    if (jv == 0u) {
        let base = idx * 16u;
        for (var i = 0u; i < 16u; i = i + 1u) { table[base + i] = 0u; }
        return;
    }
    // compute jv*G by repeated mixed add, then double 8*b times
    var gx: Fe; var gy: Fe;
    var GXv = GX; var GYv = GY;
    for (var i = 0u; i < 8u; i = i + 1u) { gx[i] = GXv[i]; gy[i] = GYv[i]; }
    var p = Jac(fe_zero(), fe_zero(), fe_zero(), 1u);
    for (var k = 0u; k < jv; k = k + 1u) { p = jac_madd(p, gx, gy); }
    // double 8*b times
    let ndoub = 8u * b;
    for (var d = 0u; d < ndoub; d = d + 1u) {
        // Jacobian double of p (not necessarily affine)
        let A = fe_sqr(p.X);
        let B = fe_sqr(p.Y);
        let C = fe_sqr(B);
        var DD = fe_add(p.X, B); DD = fe_sqr(DD);
        DD = fe_sub(DD, A); DD = fe_sub(DD, C); DD = fe_add(DD, DD);
        var E = fe_add(A, A); E = fe_add(E, A);
        let F = fe_sqr(E);
        var X3 = fe_sub(F, DD); X3 = fe_sub(X3, DD);
        var tt = fe_sub(DD, X3);
        var Y3 = fe_mul(E, tt);
        var C8 = fe_add(C, C); C8 = fe_add(C8, C8); C8 = fe_add(C8, C8);
        Y3 = fe_sub(Y3, C8);
        var Z3 = fe_mul(p.Y, p.Z); Z3 = fe_add(Z3, Z3);
        p = Jac(X3, Y3, Z3, 0u);
    }
    var aff = jac_to_affine(p);
    let base = idx * 16u;
    for (var i = 0u; i < 8u; i = i + 1u) { table[base + i] = aff.x[i]; }
    for (var i = 0u; i < 8u; i = i + 1u) { table[base + 8u + i] = aff.y[i]; }
}

fn fe_ge_n(pa: Fe) -> bool {
    var a = pa; var nn = N;
    for (var i = 8u; i > 0u; i = i - 1u) {
        let idx = i - 1u;
        if (a[idx] != nn[idx]) { return a[idx] > nn[idx]; }
    }
    return true; // equal
}
fn fe_sub_n(pa: Fe) -> Fe {
    var a = pa; var nn = N;
    var r: Fe;
    var bor = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let s = subb(a[i], nn[i], bor); r[i] = s.x; bor = s.y;
    }
    return r;
}

@compute @workgroup_size(256)
fn setup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= cfg.total_threads) { return; }
    // s = k0 + tid*2^32  (add tid at limb 1)
    var s: Fe;
    for (var i = 0u; i < 8u; i = i + 1u) { s[i] = params.k0[i]; }
    let a1 = addc(s[1], tid, 0u);
    s[1] = a1.x;
    var carry = a1.y;
    for (var i = 2u; i < 8u; i = i + 1u) {
        if (carry == 0u) { break; }
        let t = addc(s[i], 0u, carry); s[i] = t.x; carry = t.y;
    }
    // conditional subtract n if s >= n (s < 2n)
    if (fe_ge_n(s)) { s = fe_sub_n(s); }
    let j = scalarmult_base(s);
    store_state(tid, j);
}

fn check_pattern(pv: array<u32,59>) -> bool {
    var v = pv;
    let plen = cfg.prefix_len;
    let slen = cfg.suffix_len;
    if (plen == 0u && slen == 0u) { return false; }
    for (var i = 0u; i < plen; i = i + 1u) {
        if (v[1u + i] != cfg.prefix[i]) { return false; }
    }
    for (var i = 0u; i < slen; i = i + 1u) {
        if (v[59u - slen + i] != cfg.suffix[i]) { return false; }
    }
    return true;
}

fn report(tid: u32, it: u32, parity: u32, ptw: Fe, pqx: Fe) {
    var tw = ptw; var qx = pqx;
    // atomicOr-based claim (Metal backend lacks atomicCompareExchange): the
    // thread that flips 0->1 is the unique winner.
    let prev = atomicOr(&found.flag, 1u);
    if (prev == 0u) {
        found.tid = tid;
        found.iter = params.iter_base + it;
        found.parity = parity;
        for (var i = 0u; i < 8u; i = i + 1u) { found.tweak[i] = tw[i]; }
        for (var i = 0u; i < 8u; i = i + 1u) { found.qx[i] = qx[i]; }
    }
}

@compute @workgroup_size(256)
fn search(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= cfg.total_threads) { return; }
    if (atomicLoad(&found.flag) != 0u) { return; }

    var j = load_state(tid);
    var gx: Fe; var gy: Fe;
    var GXv = GX; var GYv = GY;
    for (var i = 0u; i < 8u; i = i + 1u) { gx[i] = GXv[i]; gy[i] = GYv[i]; }

    let iters = params.iters;
    for (var it = 0u; it < iters; it = it + 1u) {
        let aff = jac_to_affine(j);
        let parity = aff.y[0] & 1u;
        let tw = taptweak(aff.x);
        var q = scalarmult_base(tw);
        var pyeven = aff.y;
        if (parity != 0u) { pyeven = fe_neg_norm(aff.y); }
        q = jac_madd(q, aff.x, pyeven);
        let qx = jac_to_affine_x(q);
        let v = bech32m_values(qx);
        if (check_pattern(v)) {
            report(tid, it, parity, tw, qx);
            return;
        }

        // step P += G
        j = jac_madd(j, gx, gy);

        if ((it & 15u) == 15u && atomicLoad(&found.flag) != 0u) { break; }
    }
    store_state(tid, j);
}

// FAST/rawtr search — separate entry point so its codegen is isolated from the
// standard `search` kernel (the naga/MSL backend is sensitive to how the two
// paths share a function). Matches P.x directly and reports no parity; the
// host recovers the y-parity from the reconstructed key via the bitcoin crate.
@compute @workgroup_size(256)
fn search_fast(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= cfg.total_threads) { return; }
    if (atomicLoad(&found.flag) != 0u) { return; }

    var j = load_state(tid);
    var gx: Fe; var gy: Fe;
    var GXv = GX; var GYv = GY;
    for (var i = 0u; i < 8u; i = i + 1u) { gx[i] = GXv[i]; gy[i] = GYv[i]; }

    let iters = params.iters;
    for (var it = 0u; it < iters; it = it + 1u) {
        let px = jac_to_affine_x(j);
        let v = bech32m_values(px);
        if (check_pattern(v)) {
            report(tid, it, 0u, fe_zero(), px);
            return;
        }
        j = jac_madd(j, gx, gy);
        if ((it & 15u) == 15u && atomicLoad(&found.flag) != 0u) { break; }
    }
    store_state(tid, j);
}
