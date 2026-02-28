#!/usr/bin/env bash
set -euo pipefail

# Strip leaked ghost cfg flags that break stable cargo flows
unset RUSTFLAGS
unset CARGO_ENCODED_RUSTFLAGS
if [[ -z "${TMPDIR:-}" || ! -d "${TMPDIR:-/nonexistent}" || ! -w "${TMPDIR:-/nonexistent}" ]]; then
  export TMPDIR="/tmp"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERUS_ROOT="${VERUS_ROOT:-$ROOT_DIR/../verus}"
VERUS_SOURCE="$VERUS_ROOT/source"
TOOLCHAIN="${VERUS_TOOLCHAIN:-1.93.0-x86_64-unknown-linux-gnu}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
usage: ./scripts/check-single-file.sh <file> [function_pattern]

Verify a single source file (and its dependencies) without re-checking the
whole crate.

<file>              path relative to the repo root (e.g. src/colocation.rs)
[function_pattern]  optionally restrict verification to matching functions

examples:
  ./scripts/check-single-file.sh src/colocation.rs
  ./scripts/check-single-file.sh src/frequency.rs
EOF
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "error: expected 1-2 args: <file> [function_pattern]"
  echo "run with -h for usage"
  exit 1
fi

FILE="$1"
FUNCTION_PATTERN="${2:-}"

# Normalize to repo-relative path
if [[ "$FILE" == /* ]]; then
  FILE="${FILE#"$ROOT_DIR"/}"
fi
if [[ "$FILE" != src/* && -f "$ROOT_DIR/src/$FILE" ]]; then
  FILE="src/$FILE"
fi

if [[ ! -f "$ROOT_DIR/$FILE" ]]; then
  echo "error: file not found: $FILE"
  exit 1
fi

# Convert file path to Verus module name
#   src/colocation.rs  ->  colocation
#   src/traits/mod.rs  ->  traits
#   src/lib.rs         ->  (whole crate, no module filter)
file_to_module() {
  local f="$1"
  f="${f#src/}"
  f="${f%.rs}"
  if [[ "$f" == */mod ]]; then
    f="${f%/mod}"
  fi
  if [[ "$f" == "lib" ]]; then
    echo ""
    return
  fi
  echo "${f//\//::}"
}

MODULE="$(file_to_module "$FILE")"

if [[ -z "$MODULE" ]]; then
  echo "[check-single-file] Verifying entire crate (lib.rs)"
else
  echo "[check-single-file] Verifying module: $MODULE"
fi
if [[ -n "$FUNCTION_PATTERN" ]]; then
  echo "[check-single-file] Function filter: $FUNCTION_PATTERN"
fi

if [[ ! -x "$VERUS_SOURCE/target-verus/release/cargo-verus" ]]; then
  echo "error: cargo-verus not found at $VERUS_SOURCE/target-verus/release/cargo-verus"
  echo "hint: build Verus tools first (for example via ../VerusCAD/scripts/setup-verus.sh)"
  exit 1
fi

if [[ ! -x "$VERUS_SOURCE/z3" ]]; then
  echo "error: expected patched z3 at $VERUS_SOURCE/z3"
  echo "hint: build Verus tools first (for example via ../VerusCAD/scripts/setup-verus.sh)"
  exit 1
fi

# Build verify scope flags
VERIFY_SCOPE=()
if [[ -n "$MODULE" ]]; then
  VERIFY_SCOPE+=(--verify-module "$MODULE")
fi
if [[ -n "$FUNCTION_PATTERN" ]]; then
  VERIFY_SCOPE+=(--verify-function "$FUNCTION_PATTERN")
fi

run_cargo_verus() {
  if command -v rustup >/dev/null 2>&1; then
    export PATH="$VERUS_SOURCE/target-verus/release:$PATH"
    export VERUS_Z3_PATH="$VERUS_SOURCE/z3"
    export RUSTUP_TOOLCHAIN="$TOOLCHAIN"
    cd "$ROOT_DIR"
    cargo verus verify --manifest-path Cargo.toml -p verus-colocation \
      -- ${VERIFY_SCOPE[@]+"${VERIFY_SCOPE[@]}"} --triggers-mode silent
  elif command -v nix-shell >/dev/null 2>&1; then
    SCOPE_STR=""
    if [[ -n "$MODULE" ]]; then
      SCOPE_STR="--verify-module '$MODULE'"
    fi
    if [[ -n "$FUNCTION_PATTERN" ]]; then
      SCOPE_STR="$SCOPE_STR --verify-function '$FUNCTION_PATTERN'"
    fi
    nix-shell -p rustup --run "
      set -euo pipefail
      export RUSTUP_TOOLCHAIN='$TOOLCHAIN'
      export PATH='$VERUS_SOURCE/target-verus/release':\$PATH
      export VERUS_Z3_PATH='$VERUS_SOURCE/z3'
      cd '$ROOT_DIR'
      cargo verus verify --manifest-path Cargo.toml -p verus-colocation \
        -- $SCOPE_STR --triggers-mode silent
    "
  else
    echo "error: neither rustup nor nix-shell found in PATH"
    echo "hint: install rustup with toolchain $TOOLCHAIN"
    exit 1
  fi
}

run_cargo_verus
