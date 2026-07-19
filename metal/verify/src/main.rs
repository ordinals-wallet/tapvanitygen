// tapverify — CPU reference derivation for tapvanity-metal.
//
// Usage:
//   tapverify <internal-privkey-hex>          standard mode (BIP-341 key-path,
//                                             no script tree; applies TapTweak)
//   tapverify <output-privkey-hex> rawtr      FAST/rawtr mode: the key IS the
//                                             output key; no tweak applied
//
// Uses the same `bitcoin` crate code path as the tapvanitygen reference miner.

use bitcoin::key::{Keypair, Secp256k1, TapTweak, TweakedPublicKey};
use bitcoin::secp256k1::SecretKey;
use bitcoin::{Address, Network, XOnlyPublicKey};

fn main() {
    let hex = std::env::args().nth(1).expect("usage: tapverify <privkey-hex> [rawtr]");
    let rawtr = std::env::args().nth(2).as_deref() == Some("rawtr");
    let bytes: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("bad hex"))
        .collect();
    assert_eq!(bytes.len(), 32, "expected 32-byte key");

    let secp = Secp256k1::new();
    let secret = SecretKey::from_slice(&bytes).expect("invalid secret key");
    let keypair = Keypair::from_secret_key(&secp, &secret);
    let (xonly, _) = XOnlyPublicKey::from_keypair(&keypair);

    if rawtr {
        // The provided key is the OUTPUT key itself (rawtr descriptor
        // semantics): the witness program is its x-only pubkey, untweaked.
        let tweaked = TweakedPublicKey::dangerous_assume_tweaked(xonly);
        let address = Address::p2tr_tweaked(tweaked, Network::Bitcoin);
        println!("address {}", address);
        println!("mode rawtr");
        println!("output_secret {}", hex);
    } else {
        let address = Address::p2tr(&secp, xonly, None, Network::Bitcoin);
        let tweaked = keypair.tap_tweak(&secp, None);
        let out_secret = tweaked.to_inner().secret_key();
        println!("address {}", address);
        println!("mode standard");
        println!("internal_priv {}", hex);
        println!("output_secret {}", out_secret.display_secret());
    }
}
