/*
 * tap_vanity.metal — Bitcoin mainnet P2TR (taproot) vanity address miner.
 *
 * Per candidate:
 *   P = k*G (Jacobian, stepped incrementally P += G per iteration)
 *   px, py-parity  (affine via field inversion)
 *   t  = SHA256(SHA256("TapTweak")||SHA256("TapTweak")||px)   [BIP-341 tagged hash]
 *   Q  = P_even + t*G   (t*G via precomputed 8-bit window table)
 *   addr = bech32m("bc", 1, Q.x) ; pattern check after "bc1p"
 *
 * Field arithmetic: secp256k1, p = 2^256 - 2^32 - 977, 8 x 32-bit limbs,
 * schoolbook multiply with 64-bit accumulation, fold-reduction using
 * 2^256 ≡ 0x1000003D1 (mod p). Values are kept non-canonical (< 2^256)
 * between ops and normalized only for serialization / parity / compare.
 */

#include <metal_stdlib>
using namespace metal;

typedef uchar u8;
typedef uint  u32;
typedef ulong u64;

/* ---------------- constants ---------------- */

/* fold constant: 2^256 mod p = 2^32 + 977 = 0x1000003D1 */
#define FOLD_LO 977u

/* p, little-endian limbs */
constant u32 P_LIMBS[8] = {
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

/* curve order n, little-endian limbs */
constant u32 N_LIMBS[8] = {
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

/* generator G, affine, little-endian limbs */
constant u32 GX[8] = {
    0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
    0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu
};
constant u32 GY[8] = {
    0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
    0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u
};

/* SHA-256 round constants */
constant u32 SK[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

/* SHA-256 midstate after compressing SHA256("TapTweak")||SHA256("TapTweak")
 * (one full 64-byte block) starting from the standard IV. */
constant u32 TWEAK_MID[8] = {
    0xd129a2f3u, 0x701c655du, 0x6583b6c3u, 0xb9419727u,
    0x95f4e232u, 0x94fd54f4u, 0xa2ae8d85u, 0x47ca590bu
};

/* bech32 charset + bech32m machinery */
constant char BECH32_CHARSET[33] = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
constant u32  BECH32_GEN[5] = {0x3b6a57b2u,0x26508e6du,0x1ea119fau,0x3d4233ddu,0x2a1462b3u};
/* hrp-expand("bc") = [3,3,0,2,3] */
constant u32  HRP_EXP[5] = {3u,3u,0u,2u,3u};
#define BECH32M_CONST 0x2bc830a3u

/* ---------------- field element ops (mod p) ---------------- */
/* fe = u32[8], little-endian limbs, value < 2^256 (possibly >= p). */

static inline void fe_set(thread u32 *r, thread const u32 *a) {
    for (int i = 0; i < 8; i++) r[i] = a[i];
}
static inline void fe_set_const(thread u32 *r, constant u32 *a) {
    for (int i = 0; i < 8; i++) r[i] = a[i];
}

/* r = a + b (mod-ish: folds carry-out) */
static void fe_add(thread u32 *r, thread const u32 *a, thread const u32 *b) {
    u64 c = 0;
    for (int i = 0; i < 8; i++) {
        c += (u64)a[i] + (u64)b[i];
        r[i] = (u32)c; c >>= 32;
    }
    while (c) {           /* fold 2^256 -> 0x1000003D1 */
        u64 cc = (u64)r[0] + c * (u64)FOLD_LO;
        r[0] = (u32)cc; cc >>= 32;
        cc += (u64)r[1] + c;      /* the 2^32 part of the fold constant */
        r[1] = (u32)cc; cc >>= 32;
        for (int i = 2; i < 8 && cc; i++) {
            cc += (u64)r[i];
            r[i] = (u32)cc; cc >>= 32;
        }
        c = cc;
    }
}

/* r = a - b (mod-ish: folds borrow) */
static void fe_sub(thread u32 *r, thread const u32 *a, thread const u32 *b) {
    long borrow = 0;
    for (int i = 0; i < 8; i++) {
        long d = (long)a[i] - (long)b[i] + borrow;
        r[i] = (u32)d;
        borrow = d >> 32;   /* arithmetic shift: 0 or -1 */
    }
    while (borrow) {  /* value wrapped: subtract 0x1000003D1 */
        long bb;
        long d = (long)r[0] - (long)FOLD_LO;
        r[0] = (u32)d; bb = d >> 32;
        d = (long)r[1] - 1 + bb;
        r[1] = (u32)d; bb = d >> 32;
        for (int i = 2; i < 8 && bb; i++) {
            d = (long)r[i] + bb;
            r[i] = (u32)d; bb = d >> 32;
        }
        borrow = bb; /* 0, or -1 if it wrapped again (extremely rare) — loop folds once more */
    }
}

/* reduce a 512-bit product t[16] into r[8] (< 2^256, non-canonical) */
static void fe_reduce512(thread u32 *r, thread const u32 *t) {
    u32 s[8];
    u64 c;
    /* s = t_lo + t_hi*977 + (t_hi << 32) */
    c = (u64)t[0] + (u64)t[8] * (u64)FOLD_LO;
    s[0] = (u32)c; c >>= 32;
    for (int i = 1; i < 8; i++) {
        c += (u64)t[i] + (u64)t[8 + i] * (u64)FOLD_LO + (u64)t[7 + i];
        s[i] = (u32)c; c >>= 32;
    }
    c += (u64)t[15];              /* c < ~2^34 */
    /* fold c*(2^32 + 977) into s */
    u64 d  = c * (u64)FOLD_LO;    /* < 2^44 */
    u64 cc;
    cc = (u64)s[0] + (d & 0xffffffffu);
    r[0] = (u32)cc; cc >>= 32;
    cc += (u64)s[1] + (d >> 32) + (c & 0xffffffffu);
    r[1] = (u32)cc; cc >>= 32;
    cc += (u64)s[2] + (c >> 32);
    r[2] = (u32)cc; cc >>= 32;
    for (int i = 3; i < 8; i++) {
        cc += (u64)s[i];
        r[i] = (u32)cc; cc >>= 32;
    }
    if (cc) { /* one final fold, cannot re-carry */
        u64 e = (u64)r[0] + cc * (u64)FOLD_LO;
        r[0] = (u32)e; e >>= 32;
        e += (u64)r[1] + cc;
        r[1] = (u32)e; e >>= 32;
        for (int i = 2; i < 8 && e; i++) {
            e += (u64)r[i];
            r[i] = (u32)e; e >>= 32;
        }
    }
}

static void fe_mul(thread u32 *r, thread const u32 *a, thread const u32 *b) {
    u32 t[16];
    for (int i = 0; i < 16; i++) t[i] = 0;
    for (int i = 0; i < 8; i++) {
        u64 carry = 0;
        for (int j = 0; j < 8; j++) {
            u64 cur = (u64)t[i + j] + (u64)a[i] * (u64)b[j] + carry;
            t[i + j] = (u32)cur;
            carry = cur >> 32;
        }
        t[i + 8] = (u32)carry;
    }
    fe_reduce512(r, t);
}

/* NOTE: a dedicated schoolbook squaring (36 vs 64 partial products) was
 * implemented and passes exhaustive isolated GPU A/B tests vs fe_mul, but
 * triggers a Metal -O3 miscompilation when inlined into the deep fe_inv
 * chain (top limbs of intermediate results get zeroed). Until that is
 * chased down, squaring goes through fe_mul, which is verified correct. */
static void fe_sqr(thread u32 *r, thread const u32 *a) { fe_mul(r, a, a); }

/* canonicalize: single conditional subtract of p (value < 2^256 < 2p) */
static void fe_normalize(thread u32 *r) {
    u32 m[8];
    long borrow = 0;
    for (int i = 0; i < 8; i++) {
        long d = (long)r[i] - (long)P_LIMBS[i] + borrow;
        m[i] = (u32)d;
        borrow = d >> 32;
    }
    if (borrow == 0) {
        for (int i = 0; i < 8; i++) r[i] = m[i];
    }
}

static bool fe_is_zero_normalized(thread const u32 *a) {
    u32 acc = 0;
    for (int i = 0; i < 8; i++) acc |= a[i];
    return acc == 0;
}

/* r = a^(p-2) mod p — libsecp256k1 addition chain */
static void fe_inv(thread u32 *r, thread const u32 *a) {
    u32 x2[8], x3[8], x6[8], x9[8], x11[8], x22[8], x44[8], x88[8], x176[8], x220[8], x223[8], t1[8];
    int j;

    fe_sqr(x2, a);      fe_mul(x2, x2, a);
    fe_sqr(x3, x2);     fe_mul(x3, x3, a);
    fe_set(x6, x3);
    for (j = 0; j < 3; j++) fe_sqr(x6, x6);
    fe_mul(x6, x6, x3);
    fe_set(x9, x6);
    for (j = 0; j < 3; j++) fe_sqr(x9, x9);
    fe_mul(x9, x9, x3);
    fe_set(x11, x9);
    for (j = 0; j < 2; j++) fe_sqr(x11, x11);
    fe_mul(x11, x11, x2);
    fe_set(x22, x11);
    for (j = 0; j < 11; j++) fe_sqr(x22, x22);
    fe_mul(x22, x22, x11);
    fe_set(x44, x22);
    for (j = 0; j < 22; j++) fe_sqr(x44, x44);
    fe_mul(x44, x44, x22);
    fe_set(x88, x44);
    for (j = 0; j < 44; j++) fe_sqr(x88, x88);
    fe_mul(x88, x88, x44);
    fe_set(x176, x88);
    for (j = 0; j < 88; j++) fe_sqr(x176, x176);
    fe_mul(x176, x176, x88);
    fe_set(x220, x176);
    for (j = 0; j < 44; j++) fe_sqr(x220, x220);
    fe_mul(x220, x220, x44);
    fe_set(x223, x220);
    for (j = 0; j < 3; j++) fe_sqr(x223, x223);
    fe_mul(x223, x223, x3);

    fe_set(t1, x223);
    for (j = 0; j < 23; j++) fe_sqr(t1, t1);
    fe_mul(t1, t1, x22);
    for (j = 0; j < 5; j++) fe_sqr(t1, t1);
    fe_mul(t1, t1, a);
    for (j = 0; j < 3; j++) fe_sqr(t1, t1);
    fe_mul(t1, t1, x2);
    for (j = 0; j < 2; j++) fe_sqr(t1, t1);
    fe_mul(r, t1, a);
}

/* ---------------- point ops ---------------- */
/* Jacobian point: X, Y, Z (fe each). Affine when Z == 1. */

struct AffinePoint {   /* device table entry, canonical limbs */
    u32 x[8];
    u32 y[8];
};

/* P(jac) += Q(affine given as thread arrays). inf: P is infinity flag. */
static void jac_madd(thread u32 *X, thread u32 *Y, thread u32 *Z, thread bool &inf,
                     thread const u32 *qx, thread const u32 *qy)
{
    if (inf) {
        fe_set(X, qx); fe_set(Y, qy);
        for (int i = 0; i < 8; i++) Z[i] = 0;
        Z[0] = 1;
        inf = false;
        return;
    }
    u32 z1z1[8], u2[8], s2[8], h[8], rr[8], h2[8], h3[8], v[8], t[8];
    fe_sqr(z1z1, Z);
    fe_mul(u2, qx, z1z1);
    fe_mul(s2, Z, z1z1);
    fe_mul(s2, s2, qy);
    fe_sub(h, u2, X);
    fe_sub(rr, s2, Y);

    /* h == 0 cases (doubling / infinity) are cryptographically unreachable
     * for random scalars during mining; the table-init path never hits them
     * either (see host comments). We still guard the degenerate double. */
    u32 hn[8]; fe_set(hn, h); fe_normalize(hn);
    if (fe_is_zero_normalized(hn)) {
        u32 rn[8]; fe_set(rn, rr); fe_normalize(rn);
        if (fe_is_zero_normalized(rn)) {
            /* P == Q: do a Jacobian double of (qx,qy,1) */
            u32 A[8], B[8], C[8], D[8], E[8], F[8];
            fe_sqr(A, qx);
            fe_sqr(B, qy);
            fe_sqr(C, B);
            fe_add(D, qx, B); fe_sqr(D, D);
            fe_sub(D, D, A); fe_sub(D, D, C);
            fe_add(D, D, D);
            fe_add(E, A, A); fe_add(E, E, A);
            fe_sqr(F, E);
            fe_sub(X, F, D); fe_sub(X, X, D);
            fe_sub(t, D, X);
            fe_mul(Y, E, t);
            fe_add(C, C, C); fe_add(C, C, C); fe_add(C, C, C);
            fe_sub(Y, Y, C);
            fe_add(Z, qy, qy);
            return;
        }
        /* P == -Q: result infinity */
        inf = true;
        return;
    }

    fe_sqr(h2, h);
    fe_mul(h3, h2, h);
    fe_mul(v, X, h2);

    fe_sqr(t, rr);
    fe_sub(t, t, h3);
    fe_sub(t, t, v);
    fe_sub(t, t, v);          /* X3 */
    fe_mul(Z, Z, h);          /* Z3 = Z1*h */
    u32 y3[8];
    fe_sub(y3, v, t);
    fe_mul(y3, y3, rr);
    u32 yh3[8];
    fe_mul(yh3, Y, h3);
    fe_sub(Y, y3, yh3);
    fe_set(X, t);
}

/* device-memory variant of the table entry loader */
static void load_affine(device const AffinePoint *e, thread u32 *x, thread u32 *y) {
    for (int i = 0; i < 8; i++) { x[i] = e->x[i]; y[i] = e->y[i]; }
}

/* scalar (8 LE limbs) * G via 8-bit window table: table[i*256+j] = (j * 256^i) * G */
static void scalarmult_base(thread u32 *X, thread u32 *Y, thread u32 *Z, thread bool &inf,
                            thread const u32 *s, device const AffinePoint *table)
{
    inf = true;
    for (int byte = 0; byte < 32; byte++) {
        u32 j = (s[byte / 4] >> ((byte % 4) * 8)) & 0xffu;
        if (j == 0) continue;
        u32 ax[8], ay[8];
        load_affine(&table[byte * 256 + (int)j], ax, ay);
        jac_madd(X, Y, Z, inf, ax, ay);
    }
}

/* affine conversion: px = X/Z^2, py = Y/Z^3, both normalized */
static void jac_to_affine(thread const u32 *X, thread const u32 *Y, thread const u32 *Z,
                          thread u32 *px, thread u32 *py)
{
    u32 zi[8], zi2[8], zi3[8];
    fe_inv(zi, Z);
    fe_sqr(zi2, zi);
    fe_mul(zi3, zi2, zi);
    fe_mul(px, X, zi2);
    fe_mul(py, Y, zi3);
    fe_normalize(px);
    fe_normalize(py);
}

/* ---------------- SHA-256 single compression ---------------- */
static void sha256_compress(thread u32 *h, thread const u32 *w_in) {
    u32 w[16];
    for (int i = 0; i < 16; i++) w[i] = w_in[i];
    u32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hv=h[7];
    for (int i = 0; i < 64; i++) {
        if (i >= 16) {
            u32 v15 = w[(i + 1) & 15];
            u32 v2  = w[(i + 14) & 15];
            u32 s0 = ((v15>>7)|(v15<<25)) ^ ((v15>>18)|(v15<<14)) ^ (v15>>3);
            u32 s1 = ((v2>>17)|(v2<<15)) ^ ((v2>>19)|(v2<<13)) ^ (v2>>10);
            w[i & 15] += s0 + w[(i + 9) & 15] + s1;
        }
        u32 wi = w[i & 15];
        u32 S1 = ((e>>6)|(e<<26)) ^ ((e>>11)|(e<<21)) ^ ((e>>25)|(e<<7));
        u32 ch = (e & f) ^ (~e & g);
        u32 t1 = hv + S1 + ch + SK[i] + wi;
        u32 S0 = ((a>>2)|(a<<30)) ^ ((a>>13)|(a<<19)) ^ ((a>>22)|(a<<10));
        u32 mj = (a & b) ^ (a & c) ^ (b & c);
        u32 t2 = S0 + mj;
        hv=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hv;
}

/* TapTweak tagged hash of a 32-byte x coordinate.
 * px: normalized fe limbs (LE). Output: tweak as scalar limbs (LE u32),
 * i.e. the hash bytes interpreted big-endian. */
static void taptweak(thread const u32 *px, thread u32 *t_le) {
    u32 h[8];
    for (int i = 0; i < 8; i++) h[i] = TWEAK_MID[i];
    /* message block 2: px big-endian (32 bytes), 0x80, zeros, bitlen 768 */
    u32 w[16];
    for (int i = 0; i < 8; i++) w[i] = px[7 - i];    /* limb LE -> byte BE words */
    w[8] = 0x80000000u;
    for (int i = 9; i < 15; i++) w[i] = 0;
    w[15] = 768u;
    sha256_compress(h, w);
    /* digest words h[0..7] are BE bytes 0..31; as a BE integer,
     * LE limb i = h[7-i] */
    for (int i = 0; i < 8; i++) t_le[i] = h[7 - i];
}

/* ---------------- bech32m ---------------- */
/* Build the 5-bit value string after "bc1": v[0]=witver(1='p'),
 * v[1..52] = 5-bit groups of qx (BE bit order, 4 zero pad bits at end),
 * v[53..58] = checksum. Total 59 values. */
static void bech32m_values(thread const u32 *qx /*normalized LE limbs*/, thread u8 *v) {
    /* qx big-endian bytes */
    u8 b[32];
    for (int i = 0; i < 8; i++) {
        u32 limb = qx[7 - i];
        b[i*4]   = (u8)(limb >> 24);
        b[i*4+1] = (u8)(limb >> 16);
        b[i*4+2] = (u8)(limb >> 8);
        b[i*4+3] = (u8)(limb);
    }
    v[0] = 1;
    /* convertbits 8->5, MSB first, pad */
    u32 acc = 0;
    int bits = 0;
    int out = 1;
    for (int i = 0; i < 32; i++) {
        acc = (acc << 8) | b[i];
        bits += 8;
        while (bits >= 5) {
            bits -= 5;
            v[out++] = (u8)((acc >> bits) & 31u);
        }
    }
    if (bits > 0) v[out++] = (u8)((acc << (5 - bits)) & 31u);
    /* out == 53 */
    /* checksum: polymod(hrpexp ++ data ++ [0]*6) ^ BECH32M_CONST */
    u32 chk = 1;
    for (int i = 0; i < 5; i++) {
        u32 top = chk >> 25;
        chk = ((chk & 0x1ffffffu) << 5) ^ HRP_EXP[i];
        for (int k = 0; k < 5; k++) if ((top >> k) & 1u) chk ^= BECH32_GEN[k];
    }
    for (int i = 0; i < 53; i++) {
        u32 top = chk >> 25;
        chk = ((chk & 0x1ffffffu) << 5) ^ (u32)v[i];
        for (int k = 0; k < 5; k++) if ((top >> k) & 1u) chk ^= BECH32_GEN[k];
    }
    for (int i = 0; i < 6; i++) {
        u32 top = chk >> 25;
        chk = ((chk & 0x1ffffffu) << 5);
        for (int k = 0; k < 5; k++) if ((top >> k) & 1u) chk ^= BECH32_GEN[k];
    }
    chk ^= BECH32M_CONST;
    for (int i = 0; i < 6; i++) v[53 + i] = (u8)((chk >> (5 * (5 - i))) & 31u);
}

/* ---------------- config / output ---------------- */

struct VanityCfg {
    int prefix_len;         /* 0 = no prefix constraint */
    int suffix_len;         /* 0 = no suffix constraint */
    int fast;               /* 1 = FAST/rawtr mode: the walked point IS the
                               output key; match bech32m(P.x) directly, no
                               TapTweak hash and no t*G per candidate */
    int _pad;
    u8  prefix[32];         /* 5-bit values */
    u8  suffix[32];         /* 5-bit values */
};

struct FoundOut {
    atomic_uint flag;
    u32   tid;
    u64   iter;             /* global iteration index of the hit */
    u32   parity;           /* 1 if y(P) was odd (internal key negated) */
    u32   _pad;
    u8    tweak[32];        /* t, big-endian bytes */
    char  addr[64];         /* full address string */
};

/* per-thread persistent Jacobian point */
struct ThreadState {
    u32 X[8];
    u32 Y[8];
    u32 Z[8];
};

/* ---------------- kernels ---------------- */

/* table[i*256 + j] = (j * 256^i) * G in affine coords.
 * grid: 32 threadgroups x 256 threads. Entry j == 0 is unused/zero. */
kernel void init_table_kernel(
    device AffinePoint *table [[buffer(0)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint tlocal [[thread_position_in_threadgroup]])
{
    int i = (int)tgid;
    u32 j = tlocal;
    int idx = i * 256 + (int)j;
    if (j == 0) {
        for (int k = 0; k < 8; k++) { table[idx].x[k] = 0; table[idx].y[k] = 0; }
        return;
    }
    /* compute j*G by repeated mixed addition (j <= 255 — trivial for a
     * one-shot init; jac_madd handles the P==Q doubling case that occurs
     * when the accumulator equals G), then double 8*i times. */
    u32 X[8], Y[8], Z[8];
    bool inf = true;
    u32 gx[8], gy[8];
    fe_set_const(gx, GX);
    fe_set_const(gy, GY);
    for (u32 k = 0; k < j; k++) {
        jac_madd(X, Y, Z, inf, gx, gy);
    }
    /* now multiply by 256^i via 8*i doublings */
    for (int d = 0; d < 8 * i; d++) {
        u32 A[8], B[8], C[8], D[8], E[8], F[8], t[8];
        fe_sqr(A, X);
        fe_sqr(B, Y);
        fe_sqr(C, B);
        fe_add(D, X, B); fe_sqr(D, D);
        fe_sub(D, D, A); fe_sub(D, D, C);
        fe_add(D, D, D);
        fe_add(E, A, A); fe_add(E, E, A);
        fe_sqr(F, E);
        u32 X3[8];
        fe_sub(X3, F, D); fe_sub(X3, X3, D);
        fe_sub(t, D, X3);
        u32 Y3[8];
        fe_mul(Y3, E, t);
        fe_add(C, C, C); fe_add(C, C, C); fe_add(C, C, C);
        fe_sub(Y3, Y3, C);
        u32 Z3[8];
        fe_mul(Z3, Y, Z);
        fe_add(Z3, Z3, Z3);
        fe_set(X, X3); fe_set(Y, Y3); fe_set(Z, Z3);
    }
    u32 px[8], py[8];
    jac_to_affine(X, Y, Z, px, py);
    for (int k = 0; k < 8; k++) { table[idx].x[k] = px[k]; table[idx].y[k] = py[k]; }
}

/* setup: thread tid computes P = (k0 + tid*2^32 mod n) * G and stores it. */
kernel void setup_kernel(
    device const u32        *k0       [[buffer(0)]],   /* 8 LE limbs, already < n */
    device ThreadState      *state    [[buffer(1)]],
    device const AffinePoint*table    [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    /* s = k0 + (tid << 32) */
    u32 s[8];
    u64 c = 0;
    for (int i = 0; i < 8; i++) s[i] = k0[i];
    c = (u64)s[1] + (u64)tid;
    s[1] = (u32)c; c >>= 32;
    for (int i = 2; i < 8 && c; i++) {
        c += (u64)s[i];
        s[i] = (u32)c; c >>= 32;
    }
    /* conditional subtract n (sum < 2n given k0 < n and tid*2^32 << n) */
    if (c == 0) {
        /* compare s >= n */
        u32 m[8];
        long borrow = 0;
        for (int i = 0; i < 8; i++) {
            long d = (long)s[i] - (long)N_LIMBS[i] + borrow;
            m[i] = (u32)d; borrow = d >> 32;
        }
        if (borrow == 0) for (int i = 0; i < 8; i++) s[i] = m[i];
    } else {
        /* carried past 2^256 (only possible if k0 near 2^256; k0 < n so no) */
        u32 m[8];
        long borrow = 0;
        for (int i = 0; i < 8; i++) {
            long d = (long)s[i] - (long)N_LIMBS[i] + borrow;
            m[i] = (u32)d; borrow = d >> 32;
        }
        for (int i = 0; i < 8; i++) s[i] = m[i];
    }

    u32 X[8], Y[8], Z[8];
    bool inf = true;
    scalarmult_base(X, Y, Z, inf, s, table);
    device ThreadState *st = &state[tid];
    for (int i = 0; i < 8; i++) { st->X[i] = X[i]; st->Y[i] = Y[i]; st->Z[i] = Z[i]; }
}

/* main search kernel.
 *
 * Batched: candidates are produced BATCH at a time; the two field
 * inversions per candidate (P affine + Q affine) are amortized with
 * Montgomery batch inversion (2 inversions per BATCH candidates).
 * `iters` must be a multiple of BATCH (host enforces). */
#define BATCH 16

static void batch_invert(thread u32 z[BATCH][8], thread u32 zinv[BATCH][8]) {
    u32 m[BATCH][8];
    fe_set(m[0], z[0]);
    for (int i = 1; i < BATCH; i++) fe_mul(m[i], m[i-1], z[i]);
    u32 inv[8];
    fe_inv(inv, m[BATCH-1]);
    for (int i = BATCH - 1; i > 0; i--) {
        fe_mul(zinv[i], inv, m[i-1]);
        fe_mul(inv, inv, z[i]);
    }
    fe_set(zinv[0], inv);
}

kernel void search_kernel(
    constant u64            &iter_base [[buffer(0)]],
    constant int            &iters     [[buffer(1)]],
    constant VanityCfg      *cfg       [[buffer(2)]],
    device   FoundOut       *out       [[buffer(3)]],
    device   ThreadState    *state     [[buffer(4)]],
    device const AffinePoint*table     [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (atomic_load_explicit(&out->flag, memory_order_relaxed) != 0u) return;

    u32 X[8], Y[8], Z[8];
    device ThreadState *st = &state[tid];
    for (int i = 0; i < 8; i++) { X[i] = st->X[i]; Y[i] = st->Y[i]; Z[i] = st->Z[i]; }

    u32 gx[8], gy[8];
    fe_set_const(gx, GX);
    fe_set_const(gy, GY);

    int plen = cfg->prefix_len;
    int slen = cfg->suffix_len;
    int fast = cfg->fast;

    int nbatches = iters / BATCH;
    for (int b = 0; b < nbatches; b++) {
        /* ---- phase A: emit BATCH candidate points, stepping P += G ---- */
        u32 cX[BATCH][8], cY[BATCH][8], cZ[BATCH][8];
        for (int c = 0; c < BATCH; c++) {
            fe_set(cX[c], X); fe_set(cY[c], Y); fe_set(cZ[c], Z);
            bool pinf = false;
            jac_madd(X, Y, Z, pinf, gx, gy);
        }

        /* ---- batch invert candidate Zs ---- */
        u32 zinv[BATCH][8];
        batch_invert(cZ, zinv);

        if (fast) {
            /* ---- FAST/rawtr: the point itself is the output key ---- */
            for (int c = 0; c < BATCH; c++) {
                u32 zi2[8], zi3[8], px[8], py[8];
                fe_sqr(zi2, zinv[c]);
                fe_mul(zi3, zi2, zinv[c]);
                fe_mul(px, cX[c], zi2);
                fe_mul(py, cY[c], zi3);
                fe_normalize(px);
                fe_normalize(py);

                u8 v[59];
                bech32m_values(px, v);

                bool ok = true;
                for (int i = 0; i < plen && ok; i++) {
                    if (v[1 + i] != cfg->prefix[i]) ok = false;
                }
                if (ok && slen > 0) {
                    for (int i = 0; i < slen && ok; i++) {
                        if (v[59 - slen + i] != cfg->suffix[i]) ok = false;
                    }
                }
                if (ok && (plen > 0 || slen > 0)) {
                    uint expected = 0;
                    if (atomic_compare_exchange_weak_explicit(&out->flag, &expected, 1u,
                                                              memory_order_relaxed, memory_order_relaxed)) {
                        out->tid = tid;
                        out->iter = iter_base + (u64)(b * BATCH + c);
                        out->parity = py[0] & 1u;
                        for (int i = 0; i < 32; i++) out->tweak[i] = 0;
                        out->addr[0] = 'b'; out->addr[1] = 'c'; out->addr[2] = '1';
                        for (int i = 0; i < 59; i++) out->addr[3 + i] = BECH32_CHARSET[v[i]];
                        out->addr[62] = '\0';
                    }
                    return;
                }
            }
            if (atomic_load_explicit(&out->flag, memory_order_relaxed) != 0u) break;
            continue;
        }

        /* ---- phase B: per candidate, affine P, tweak, Q jacobian ---- */
        u32 qX[BATCH][8], qZ[BATCH][8];
        u32 tw[BATCH][8];
        u32 par[BATCH];
        for (int c = 0; c < BATCH; c++) {
            u32 zi2[8], zi3[8], px[8], py[8];
            fe_sqr(zi2, zinv[c]);
            fe_mul(zi3, zi2, zinv[c]);
            fe_mul(px, cX[c], zi2);
            fe_mul(py, cY[c], zi3);
            fe_normalize(px);
            fe_normalize(py);
            par[c] = py[0] & 1u;

            taptweak(px, tw[c]);

            u32 QX[8], QY[8], QZ[8];
            bool qinf = true;
            scalarmult_base(QX, QY, QZ, qinf, tw[c], table);
            u32 pyeven[8];
            if (par[c]) {
                long borrow = 0;
                for (int i = 0; i < 8; i++) {
                    long d = (long)P_LIMBS[i] - (long)py[i] + borrow;
                    pyeven[i] = (u32)d; borrow = d >> 32;
                }
            } else {
                fe_set(pyeven, py);
            }
            jac_madd(QX, QY, QZ, qinf, px, pyeven);
            fe_set(qX[c], QX);
            fe_set(qZ[c], QZ);
        }

        /* ---- batch invert Q Zs ---- */
        u32 qzinv[BATCH][8];
        batch_invert(qZ, qzinv);

        /* ---- phase C: affine Q.x, bech32m, pattern check ---- */
        for (int c = 0; c < BATCH; c++) {
            u32 zi2[8], qx[8];
            fe_sqr(zi2, qzinv[c]);
            fe_mul(qx, qX[c], zi2);
            fe_normalize(qx);

            u8 v[59];
            bech32m_values(qx, v);

            bool ok = true;
            for (int i = 0; i < plen && ok; i++) {
                if (v[1 + i] != cfg->prefix[i]) ok = false;
            }
            if (ok && slen > 0) {
                for (int i = 0; i < slen && ok; i++) {
                    if (v[59 - slen + i] != cfg->suffix[i]) ok = false;
                }
            }

            if (ok && (plen > 0 || slen > 0)) {
                uint expected = 0;
                if (atomic_compare_exchange_weak_explicit(&out->flag, &expected, 1u,
                                                          memory_order_relaxed, memory_order_relaxed)) {
                    out->tid = tid;
                    out->iter = iter_base + (u64)(b * BATCH + c);
                    out->parity = par[c];
                    for (int i = 0; i < 8; i++) {
                        u32 limb = tw[c][7 - i];
                        out->tweak[i*4]   = (u8)(limb >> 24);
                        out->tweak[i*4+1] = (u8)(limb >> 16);
                        out->tweak[i*4+2] = (u8)(limb >> 8);
                        out->tweak[i*4+3] = (u8)(limb);
                    }
                    out->addr[0] = 'b'; out->addr[1] = 'c'; out->addr[2] = '1';
                    for (int i = 0; i < 59; i++) out->addr[3 + i] = BECH32_CHARSET[v[i]];
                    out->addr[62] = '\0';
                }
                return;
            }
        }

        if (atomic_load_explicit(&out->flag, memory_order_relaxed) != 0u) break;
    }

    for (int i = 0; i < 8; i++) { st->X[i] = X[i]; st->Y[i] = Y[i]; st->Z[i] = Z[i]; }
}

/* debug: derive address for scalar s (8 LE limbs) — used by self-test */
kernel void debug_derive(
    device const u32         *s_in   [[buffer(0)]],
    device u8                *addr   [[buffer(1)]],   /* 64 bytes */
    device u8                *aux    [[buffer(2)]],   /* 32B px + 32B tweak + 1B parity */
    device const AffinePoint *table  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    u32 s[8];
    for (int i = 0; i < 8; i++) s[i] = s_in[i];
    u32 X[8], Y[8], Z[8];
    bool inf = true;
    scalarmult_base(X, Y, Z, inf, s, table);
    u32 px[8], py[8];
    jac_to_affine(X, Y, Z, px, py);
    u32 parity = py[0] & 1u;
    u32 t[8];
    taptweak(px, t);
    u32 QX[8], QY[8], QZ[8];
    bool qinf = true;
    scalarmult_base(QX, QY, QZ, qinf, t, table);
    u32 pyeven[8];
    if (parity) {
        long borrow = 0;
        for (int i = 0; i < 8; i++) {
            long d = (long)P_LIMBS[i] - (long)py[i] + borrow;
            pyeven[i] = (u32)d; borrow = d >> 32;
        }
    } else {
        fe_set(pyeven, py);
    }
    jac_madd(QX, QY, QZ, qinf, px, pyeven);
    u32 qx[8], qy[8];
    jac_to_affine(QX, QY, QZ, qx, qy);
    u8 v[59];
    bech32m_values(qx, v);
    addr[0]='b'; addr[1]='c'; addr[2]='1';
    for (int i = 0; i < 59; i++) addr[3+i] = (u8)BECH32_CHARSET[v[i]];
    addr[62] = 0;
    for (int i = 0; i < 8; i++) {
        u32 limb = px[7-i];
        aux[i*4]=(u8)(limb>>24); aux[i*4+1]=(u8)(limb>>16); aux[i*4+2]=(u8)(limb>>8); aux[i*4+3]=(u8)limb;
        limb = t[7-i];
        aux[32+i*4]=(u8)(limb>>24); aux[32+i*4+1]=(u8)(limb>>16); aux[32+i*4+2]=(u8)(limb>>8); aux[32+i*4+3]=(u8)limb;
    }
    aux[64] = (u8)parity;
}
