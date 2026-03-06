#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_CKPT="${ROOT_DIR}/ckpts/sc22mc/DocFusion"

set -euo pipefail

cd "${ROOT_DIR}"

CONVERT_OUTTYPE="${CONVERT_OUTTYPE:-bf16}"
DOCFUSION_TASK="${DOCFUSION_TASK:-OCR}"
DOCFUSION_IMAGE="${DOCFUSION_IMAGE:-${HF_CKPT}/img_text.png}"
DOCFUSION_MAX_NEW_TOKENS="${DOCFUSION_MAX_NEW_TOKENS:-128}"

DOCFUSION_CLI_BIN="${ROOT_DIR}/build/bin/llama-mtmd-cli"
GGUF_MODEL="${HF_CKPT}/Docfusion-194M-$(echo "${CONVERT_OUTTYPE}" | tr '[:lower:]' '[:upper:]').gguf"
MMPROJ_MODEL="${ROOT_DIR}/ckpts/sc22mc/mmproj-DocFusion"

build_prompt() {
    case "${DOCFUSION_TASK^^}" in
        "OD"|"<OD>")
            printf '%s' $'<OD>\n<image>'
            ;;
        "OCR"|"<OCR>")
            printf '%s' $'<OCR>\n<image>'
            ;;
        "MER"|"<MER>")
            printf '%s' $'<MER>\n<image>'
            ;;
        "TR"|"<TR>")
            printf '%s' $'<TR>\n<image>'
            ;;
        *)
            printf '%s' "${DOCFUSION_TASK}"
            ;;
    esac
}

PROMPT="$(build_prompt)"

cmake -B build -DLLAMA_CURL=OFF
cmake --build build --target llama-mtmd-cli -j "$(nproc)"

echo "[inference] cli    : ${DOCFUSION_CLI_BIN}"
echo "[inference] model  : ${GGUF_MODEL}"
echo "[inference] mmproj : ${MMPROJ_MODEL}"
echo "[inference] image  : ${DOCFUSION_IMAGE}"
printf '[inference] prompt : %q\n' "${PROMPT}"
echo "[inference] n_predict: ${DOCFUSION_MAX_NEW_TOKENS}"

"${DOCFUSION_CLI_BIN}" \
    -m "${GGUF_MODEL}" \
    --mmproj "${MMPROJ_MODEL}" \
    --chat-template chatml \
    --image "${DOCFUSION_IMAGE}" \
    -p "${PROMPT}" \
    -n "${DOCFUSION_MAX_NEW_TOKENS}" \
    --temp 0 \
    --top-k 1 \
    --top-p 1 \
    --seed 42 \
    -ub 2048
