#!/usr/bin/env bash
# =============================================================================
# ZipCount extra public datasets.
#
# Usage:
#   bash scripts/download_more.sh ami notsofar ddi
#   bash scripts/download_more.sh all
#
# Env overrides:
#   DATA_ROOT          default: <project>/data_ext
#   PY                 default: ~/.miniconda3/envs/.zipformer/bin/python fallback
#   AMI_MIC            default: sdm
#   NOTSOFAR_SUBSET    default: train_set
#   NOTSOFAR_VERSION   default: 240825.1_train
#   NOTSOFAR_KIND      default: meeting   (meeting|simulated)
#   NOTSOFAR_VOLUME    default: 200hrs    (for simulated only)
#   NOTSOFAR_SAS       optional Azure SAS token/query string for private blobs
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data_ext}"
PY="${PY:-$HOME/miniconda3/envs/.zipformer/bin/python}"
LOG_DIR="$PROJECT_ROOT/logs/download_more"
mkdir -p "$DATA_ROOT" "$LOG_DIR"

if [ ! -x "$PY" ]; then
  echo "ERROR: Python not found: $PY"
  echo "Set PY=/path/to/python or install/use conda env .zipformer."
  exit 1
fi

want() {
  local name="$1"
  if [ "$#" -eq 1 ]; then
    return 1
  fi
  shift
  for arg in "$@"; do
    case "${arg,,}" in
      all|"$name") return 0 ;;
      ddi) [ "$name" = "dipco" ] && return 0 ;;
      nsf) [ "$name" = "notsofar" ] && return 0 ;;
    esac
  done
  return 1
}

download_ami() {
  local mic="${AMI_MIC:-sdm}"
  local out="$DATA_ROOT/ami_${mic}"
  echo "== AMI (${mic}) -> $out =="
  "$PY" - <<PY
from lhotse.recipes.ami import download_ami
download_ami(target_dir="$out", mic="$mic")
print("AMI download done:", "$out")
PY
}

download_dipco() {
  local out="$DATA_ROOT/dipco"
  local tar_path="$out/DipCo.tgz"
  local url="${DIPCO_URL:-https://zenodo.org/record/8122551/files/DipCo.tgz?download=1}"
  echo "== DiPCo/DDI -> $out =="
  mkdir -p "$out"
  wget -c -O "$tar_path" "$url"
  tar xzf "$tar_path" -C "$out"
  echo "DiPCo download done: $out"
}

ensure_azcopy() {
  if command -v azcopy >/dev/null 2>&1; then
    command -v azcopy
    return
  fi

  local tools="$PROJECT_ROOT/third_party/tools"
  local az="$tools/azcopy"
  if [ -x "$az" ]; then
    echo "$az"
    return
  fi

  echo "== Install local azcopy -> $tools ==" >&2
  mkdir -p "$tools"
  local tmp
  tmp="$(mktemp -d)"
  wget -q -O "$tmp/azcopy.tar.gz" "https://aka.ms/downloadazcopy-v10-linux"
  tar xzf "$tmp/azcopy.tar.gz" -C "$tmp"
  local bin
  bin="$(find "$tmp" -type f -name azcopy | head -n 1)"
  cp "$bin" "$az"
  chmod +x "$az"
  rm -rf "$tmp"
  echo "$az"
}

download_notsofar() {
  local kind="${NOTSOFAR_KIND:-meeting}"
  local azcopy_bin
  azcopy_bin="$(ensure_azcopy)"
  local sas="${NOTSOFAR_SAS:-}"
  if [ -n "$sas" ] && [[ "$sas" != \?* ]]; then
    sas="?$sas"
  fi

  if [ "$kind" = "simulated" ]; then
    local version="${NOTSOFAR_VERSION:-v1.5}"
    local volume="${NOTSOFAR_VOLUME:-200hrs}"
    local subset="${NOTSOFAR_SUBSET:-train}"
    local out="$DATA_ROOT/nsf/css-datasets/$version/$volume/$subset"
    local url="https://notsofarsa.blob.core.windows.net/css-datasets/$version/$volume/$subset$sas"
    echo "== NOTSOFAR simulated $version/$volume/$subset -> $out =="
    mkdir -p "$out"
    "$azcopy_bin" copy "$url" "$out" --recursive
  else
    local subset="${NOTSOFAR_SUBSET:-train_set}"
    local version="${NOTSOFAR_VERSION:-240825.1_train}"
    local out="$DATA_ROOT/nsf/sdm/benchmark-datasets/$subset/$version"
    local url="https://notsofarsa.blob.core.windows.net/benchmark-datasets/$subset/$version/MTG$sas"
    echo "== NOTSOFAR meeting $subset/$version -> $out =="
    mkdir -p "$out"
    "$azcopy_bin" copy "$url" "$out" --recursive
  fi
}

if [ "$#" -eq 0 ]; then
  set -- ami notsofar dipco
fi

echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "DATA_ROOT    = $DATA_ROOT"
echo "PY           = $PY"
echo "REQUESTED    = $*"

if want ami "$@"; then
  download_ami 2>&1 | tee -a "$LOG_DIR/ami.log"
fi

if want notsofar "$@"; then
  download_notsofar 2>&1 | tee -a "$LOG_DIR/notsofar.log"
fi

if want dipco "$@"; then
  download_dipco 2>&1 | tee -a "$LOG_DIR/dipco.log"
fi

echo "DONE"
