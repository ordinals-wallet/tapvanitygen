# Tapvanitygen

Create taproot vanity addresses.

## Building

```
cargo build --release
```

## Usage


To mine a taproot address that starts with `bc1pepe`

```
./target/release/tapvanitygen --pattern epe
```

To mine a taproot address that ends with `spam`

```
./target/release/tapvanitygen --pattern spam --suffix
```
