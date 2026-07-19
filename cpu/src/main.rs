use bitcoin::{
    key::{
        secp256k1::{rand, SecretKey},
        Keypair,
    },
    Address, Network, XOnlyPublicKey,
};
use clap::Parser;
use rayon::prelude::*;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    pattern: String,
    #[arg(short, long, default_value_t = false)]
    suffix: bool,
}

fn main() {
    let args = Args::parse();
    let secp = bitcoin::key::Secp256k1::new();

    loop {
        let payload = vec![0; 1_000_000];
        let start_time = Instant::now();

        let results = payload
            .par_iter()
            .map(|_| {
                let secret = SecretKey::new(&mut rand::thread_rng());
                let keypair = Keypair::from_secret_key(&secp, &secret);
                let (x_only_public_key, _) = XOnlyPublicKey::from_keypair(&keypair);
                let address = Address::p2tr(&secp, x_only_public_key, None, Network::Bitcoin);

                let address_str = format!("{}", address);

                if args.suffix {
                    if address_str.ends_with(&args.pattern) {
                        return Some((address, keypair));
                    }
                } else {
                    if address_str.starts_with(&format!("bc1p{}", args.pattern)) {
                        return Some((address, keypair));
                    }
                }

                return None;
            })
            .filter(|e| e.is_some())
            .collect::<Vec<_>>();

        let duration = start_time.elapsed().as_millis();

        println!("{:.2} H/s", payload.len() as f64 / duration as f64 * 1000.0);

        if results.len() > 0 {
            let (address, keypair) = results[0].clone().unwrap();
            println!("address: {}", address);
            println!("private key: {}", keypair.display_secret());

            break;
        }
    }
}
