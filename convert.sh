#!/usr/bin/env bash

export HF_ENDPOINT=https://hf-mirror.com

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_CKPT="${ROOT_DIR}/ckpts/sc22mc/DocFusion"
HF_REPO_ID="${HF_REPO_ID:-sc22mc/DocFusion}"

set -euo pipefail

cd "${ROOT_DIR}"

if [[ ! -d "${HF_CKPT}" || ( ! -f "${HF_CKPT}/model.safetensors" && ! -f "${HF_CKPT}/pytorch_model.bin" ) ]]; then
    echo "[convert] ckpt missing or incomplete: ${HF_CKPT}"
    echo "[convert] downloading from Hugging Face: ${HF_REPO_ID}"
    mkdir -p "$(dirname "${HF_CKPT}")"
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "${HF_REPO_ID}" --local-dir "${HF_CKPT}"
    else
        python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${HF_REPO_ID}", local_dir="${HF_CKPT}", local_dir_use_symlinks=False)
PY
    fi
fi

CONVERT_OUTTYPE="${CONVERT_OUTTYPE:-bf16}"
CONVERT_SCRIPT="${ROOT_DIR}/convert_hf_to_gguf_ende.py"

GGUF_MODEL="${HF_CKPT}/Docfusion-194M-$(echo "${CONVERT_OUTTYPE}" | tr '[:lower:]' '[:upper:]').gguf"
MMPROJ_MODEL="${ROOT_DIR}/ckpts/sc22mc/mmproj-DocFusion"

echo "[convert] script : ${CONVERT_SCRIPT}"
echo "[convert] ckpt   : ${HF_CKPT}"
echo "[convert] outtype: ${CONVERT_OUTTYPE}"

python "${CONVERT_SCRIPT}" "${HF_CKPT}" --outtype "${CONVERT_OUTTYPE}"
python "${CONVERT_SCRIPT}" "${HF_CKPT}" --mmproj --outtype "${CONVERT_OUTTYPE}"

echo "[convert] done"
echo "[convert] gguf  : ${GGUF_MODEL}"
echo "[convert] mmproj: ${MMPROJ_MODEL}"
