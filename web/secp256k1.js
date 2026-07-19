// secp256k1.js — minimal VENDORED secp256k1 + bech32m + base58check/WIF,
// pure vanilla JS (BigInt). Used ONLY for in-browser verification of GPU
// finds (compute pubkey(k), x-only, y-parity, bech32m address, WIF).
// Not constant-time; do not use for signing on hostile inputs.

"use strict";

const SECP = (() => {
  const P = (1n << 256n) - (1n << 32n) - 977n;
  const N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141n;
  const GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798n;
  const GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8n;

  const mod = (a, m) => ((a % m) + m) % m;

  // modular inverse via extended Euclid
  function inv(a, m) {
    a = mod(a, m);
    let [g, x, _g, _x] = [a, 1n, m, 0n];
    while (g !== 0n) {
      const q = _g / g;
      [g, _g] = [_g - q * g, g];
      [x, _x] = [_x - q * x, x];
    }
    return mod(_x, m);
  }

  // affine point ops (null = infinity)
  function add(p, q) {
    if (!p) return q;
    if (!q) return p;
    if (p[0] === q[0]) {
      if (mod(p[1] + q[1], P) === 0n) return null;
      // double
      const l = mod(3n * p[0] * p[0] * inv(2n * p[1], P), P);
      const x = mod(l * l - 2n * p[0], P);
      return [x, mod(l * (p[0] - x) - p[1], P)];
    }
    const l = mod((q[1] - p[1]) * inv(q[0] - p[0], P), P);
    const x = mod(l * l - p[0] - q[0], P);
    return [x, mod(l * (p[0] - x) - p[1], P)];
  }

  function mulG(k) {
    k = mod(k, N);
    let r = null, a = [GX, GY];
    while (k > 0n) {
      if (k & 1n) r = add(r, a);
      a = add(a, a);
      k >>= 1n;
    }
    return r;
  }

  function bigToBytes32(b) {
    const out = new Uint8Array(32);
    for (let i = 31; i >= 0; i--) { out[i] = Number(b & 0xffn); b >>= 8n; }
    return out;
  }
  function bytesToBig(bytes) {
    let r = 0n;
    for (const b of bytes) r = (r << 8n) | BigInt(b);
    return r;
  }

  // pubkey of private key k (BigInt): { x: Uint8Array(32), parityOdd: bool }
  function pubkey(k) {
    const pt = mulG(k);
    if (!pt) throw new Error("infinity");
    return { x: bigToBytes32(pt[0]), xBig: pt[0], parityOdd: (pt[1] & 1n) === 1n };
  }

  // ---------------- bech32m ----------------
  const CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
  function polymod(values) {
    const GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3];
    let chk = 1;
    for (const v of values) {
      const top = chk >>> 25;
      chk = ((chk & 0x1ffffff) << 5) ^ v;
      for (let i = 0; i < 5; i++) if ((top >>> i) & 1) chk ^= GEN[i];
    }
    return chk >>> 0;
  }
  // mainnet P2TR address for 32-byte x-only key
  function bech32mP2TR(xonly32) {
    const data = [1]; // witness v1
    let acc = 0, bits = 0;
    for (const b of xonly32) {
      acc = (acc << 8) | b;
      bits += 8;
      while (bits >= 5) { bits -= 5; data.push((acc >> bits) & 31); }
    }
    if (bits > 0) data.push((acc << (5 - bits)) & 31);
    const hrpExp = [3, 3, 0, 2, 3]; // "bc"
    const pm = polymod([...hrpExp, ...data, 0, 0, 0, 0, 0, 0]) ^ 0x2bc830a3;
    let s = "bc1";
    for (const v of data) s += CHARSET[v];
    for (let i = 0; i < 6; i++) s += CHARSET[(pm >>> (5 * (5 - i))) & 31];
    return s;
  }

  // ---------------- SHA-256 (for base58check) ----------------
  function sha256(bytes) {
    const K = [0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
    let H = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    const l = bytes.length;
    const withPad = new Uint8Array(((l + 9 + 63) >> 6) << 6);
    withPad.set(bytes);
    withPad[l] = 0x80;
    const bitLen = l * 8;
    const dv = new DataView(withPad.buffer);
    dv.setUint32(withPad.length - 4, bitLen >>> 0);
    dv.setUint32(withPad.length - 8, Math.floor(bitLen / 0x100000000));
    const w = new Uint32Array(64);
    const rotr = (x, n) => (x >>> n) | (x << (32 - n));
    for (let off = 0; off < withPad.length; off += 64) {
      for (let i = 0; i < 16; i++) w[i] = dv.getUint32(off + 4 * i);
      for (let i = 16; i < 64; i++) {
        const s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >>> 3);
        const s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >>> 10);
        w[i] = (w[i-16] + s0 + w[i-7] + s1) >>> 0;
      }
      let [a,b,c,d,e,f,g,h] = H;
      for (let i = 0; i < 64; i++) {
        const S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        const ch = (e & f) ^ (~e & g);
        const t1 = (h + S1 + ch + K[i] + w[i]) >>> 0;
        const S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        const mj = (a & b) ^ (a & c) ^ (b & c);
        const t2 = (S0 + mj) >>> 0;
        h=g; g=f; f=e; e=(d+t1)>>>0; d=c; c=b; b=a; a=(t1+t2)>>>0;
      }
      H = [(H[0]+a)>>>0,(H[1]+b)>>>0,(H[2]+c)>>>0,(H[3]+d)>>>0,(H[4]+e)>>>0,(H[5]+f)>>>0,(H[6]+g)>>>0,(H[7]+h)>>>0];
    }
    const out = new Uint8Array(32);
    for (let i = 0; i < 8; i++) {
      out[4*i] = H[i] >>> 24; out[4*i+1] = (H[i] >>> 16) & 0xff;
      out[4*i+2] = (H[i] >>> 8) & 0xff; out[4*i+3] = H[i] & 0xff;
    }
    return out;
  }

  // ---------------- base58check / WIF ----------------
  const B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  function base58check(payload) {
    const chk = sha256(sha256(payload)).slice(0, 4);
    const data = new Uint8Array(payload.length + 4);
    data.set(payload); data.set(chk, payload.length);
    let zeros = 0;
    while (zeros < data.length && data[zeros] === 0) zeros++;
    let num = bytesToBig(data);
    let out = "";
    while (num > 0n) { out = B58[Number(num % 58n)] + out; num /= 58n; }
    return "1".repeat(zeros) + out;
  }
  // compressed mainnet WIF: 0x80 || key32 || 0x01
  function wif(keyBig) {
    const payload = new Uint8Array(34);
    payload[0] = 0x80;
    payload.set(bigToBytes32(keyBig), 1);
    payload[33] = 0x01;
    return base58check(payload);
  }

  return { P, N, GX, GY, mod, inv, mulG, pubkey, bigToBytes32, bytesToBig,
           bech32mP2TR, sha256, base58check, wif, CHARSET };
})();
