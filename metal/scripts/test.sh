#!/usr/bin/env bash
# End-to-end tests:
#   1. GPU self-test (known vector k=1)
#   2. Standard mode: mine a trivial prefix, cross-check address + tweaked
#      secret against the Rust reference
#   3. FAST/rawtr mode: mine a trivial prefix, cross-check via rawtr path
#   4. Combined --prefix + --suffix in both modes
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build/tapvanity_metal
VERIFY=verify/target/release/tapverify
FAIL=0

echo "== GPU self-test (known vector k=1) =="
$BIN --self-test

run_case() {  # run_case <label> <verify-mode: standard|rawtr> <expect-glob> <miner args...>
  local label=$1 vmode=$2 glob=$3; shift 3
  echo
  echo "== $label: $BIN $* =="
  local OUT ADDR PRIV TWEAKED REF REF_ADDR REF_OUT
  OUT=$($BIN "$@")
  ADDR=$(echo "$OUT" | awk '/^ADDR /{print $2}')
  PRIV=$(echo "$OUT" | awk '/^PRIV /{print $2}')
  TWEAKED=$(echo "$OUT" | awk '/^TWEAKED /{print $2}')
  echo "   addr $ADDR"
  echo "   priv $PRIV"
  if [[ -z "$ADDR" || -z "$PRIV" ]]; then
    echo "FAIL($label): miner did not report ADDR/PRIV"; FAIL=1; return
  fi
  if ! echo "$OUT" | grep -q '^VERIFY OK'; then
    echo "FAIL($label): miner did not self-verify"; FAIL=1
  fi
  if [[ "$vmode" == rawtr ]]; then
    REF=$($VERIFY "$PRIV" rawtr)
  else
    REF=$($VERIFY "$PRIV")
  fi
  REF_ADDR=$(echo "$REF" | awk '/^address /{print $2}')
  REF_OUT=$(echo "$REF" | awk '/^output_secret /{print $2}')
  if [[ "$REF_ADDR" != "$ADDR" ]]; then
    echo "FAIL($label): address mismatch: miner=$ADDR rust=$REF_ADDR"; FAIL=1
  fi
  if [[ "$vmode" == standard && "$REF_OUT" != "$TWEAKED" ]]; then
    echo "FAIL($label): tweaked secret mismatch: miner=$TWEAKED rust=$REF_OUT"; FAIL=1
  fi
  # shellcheck disable=SC2254
  case "$ADDR" in
    $glob) ;;
    *) echo "FAIL($label): address does not match pattern: $ADDR"; FAIL=1 ;;
  esac
  echo "   OK ($vmode verified)"
}

run_case "standard prefix"   standard 'bc1pqq*'   --prefix qq
run_case "standard combined" standard 'bc1pt*q'   --prefix t --suffix q
run_case "fast prefix"       rawtr    'bc1pqq*'   --fast --prefix qq
run_case "fast combined"     rawtr    'bc1pt*q'   --fast --prefix t --suffix q
run_case "fast suffix"       rawtr    'bc1p*xx'   --fast --suffix xx

if [[ $FAIL -eq 0 ]]; then
  echo
  echo "PASS: all modes verified against Rust reference"
else
  echo
  echo "FAILURES PRESENT"
  exit 1
fi
