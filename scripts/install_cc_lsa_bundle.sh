#!/usr/bin/env bash
# Install only the explicit CC-LSA file list, preserving replaced files.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'Usage: bash install_cc_lsa_bundle.sh EXTRACTED_STAGE EXISTING_PROJECT' >&2
  exit 2
fi
stage="$(cd -- "$1" && pwd -P)"
target="$(cd -- "$2" && pwd -P)"
manifest="$stage/scripts/cc_lsa_sync_files.txt"
[[ -f "$target/run.py" && -d "$target/src" && -f "$manifest" ]]
files=()
while IFS= read -r relative || [[ -n "$relative" ]]; do
  relative="${relative%$'\r'}"
  [[ -z "$relative" ]] && continue
  case "$relative" in
    /*|..|../*|*/../*|*/..) echo "Invalid relative path: $relative" >&2; exit 2 ;;
  esac
  [[ -f "$stage/$relative" && ! -L "$stage/$relative" ]]
  files+=("$relative")
done < "$manifest"
[[ ${#files[@]} -gt 0 ]]
mkdir -p "$target/.cache"
backup="$(mktemp -d "$target/.cache/cc_lsa_sync_backup.XXXXXX")"
for relative in "${files[@]}"; do
  if [[ -e "$target/$relative" ]]; then
    [[ -f "$target/$relative" && ! -L "$target/$relative" ]]
    mkdir -p "$backup/$(dirname -- "$relative")"
    cp -p -- "$target/$relative" "$backup/$relative"
  fi
done
for relative in "${files[@]}"; do
  mkdir -p "$target/$(dirname -- "$relative")"
  cp -p -- "$stage/$relative" "$target/$relative"
done
# Shell code may have acquired CRLF from a Windows checkout.
sed -i 's/\r$//' "$target/scripts/run_cc_lsa_gate_a.sh" "$target/scripts/install_cc_lsa_bundle.sh"
printf 'Synced %d files. Previous files retained at: %s\n' "${#files[@]}" "$backup"
