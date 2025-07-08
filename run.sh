#!/bin/bash

set -e

# Convert Hugging Face model to GGUF format
# python convert_hf_to_gguf.py /home/binguo/data/llama.cpp/ckpts/smollm1-135m-d_kv_16  --outtype bf16
# python convert_hf_to_gguf.py /home/binguo/data/llama.cpp/ckpts/smollm1-360m-d_kv_16  --outtype bf16
# python convert_hf_to_gguf.py /home/binguo/data/llama.cpp/ckpts/smollm1-1B7-d_kv_16  --outtype bf16
# python convert_hf_to_gguf.py /home/binguo/data/llama.cpp/ckpts/llama2-7B-d_kv_32  --outtype bf16
# python convert_hf_to_gguf.py /home/binguo/data/models/HuggingFaceTB/SmolLM-360M  --outtype bf16
# python convert_hf_to_gguf.py /home/binguo/data/models/HuggingFaceTB/SmolLM-1.7B  --outtype bf16
# python convert_hf_to_gguf.py /home/binguo/data/models/meta-llama/Llama-2-7b-hf  --outtype bf16

# python convert_hf_to_gguf.py /home/binguo/data/llama.cpp/ckpts/qwen3-0_6B-rope16-dkv16  --outtype bf16

# Check hyperparameters in the GGUF file
# python -c "
# import gguf  
# reader = gguf.GGUFReader('/home/binguo/data/llama.cpp/ckpts/smollm1-135m-d_kv_16/SmolLM-135M-d_kv_16-BF16.gguf')  
# print(f'{reader.fields.keys()}')
# for key, value in reader.fields.items():  
#     if 'mha2mla' in key:  
#         print(f'{key}: {value}')
# "

# Check Tensor in the GGUF file
# python gguf-py/gguf/scripts/gguf_dump.py /home/binguo/data/llama.cpp/ckpts/qwen3-0_6B-rope16-dkv16/qwen3-0.6B-rope16-dkv16-BF16.gguf

# Build llama.cpp
# rm -rf build
# cmake -B build -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Debug
# # cmake -B build -DLLAMA_CURL=OFF
# # export CUDA_HOME=/usr/local/cuda-12.1 && export PATH=$CUDA_HOME/bin:$PATH && export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# # cmake -B build -DLLAMA_CURL=OFF -DGGML_CUDA=on -DGGML_CUDA_F16=on -DGGML_CUDA_FA_ALL_QUANTS=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc
# cmake --build build --config Release -j 32


# Run llama-cli

# ./build/bin/llama-cli -m /home/binguo/data/llama.cpp/ckpts/smollm1-135m-d_kv_8/SmolLM-135M-d_kv_8-BF16.gguf -p "Hi, how are you?" -n 128 -no-cnv

# ./build/bin/llama-cli -m /home/binguo/data/llama.cpp/ckpts/smollm1-135m-d_kv_8/SmolLM-135M-d_kv_8-BF16.gguf -p -cnv

# ./build/bin/llama-cli -m /home/binguo/data/models/HuggingFaceTB/SmolLM-135M/SmolLM-135M-BF16.gguf -cnv

# ./build/bin/llama-cli -m /home/binguo/data/llama.cpp/ckpts/smollm1-360m-d_kv_16/SmolLM-360M-d_kv_16-BF16.gguf -cnv

# ./build/bin/llama-cli -m /home/binguo/data/models/HuggingFaceTB/SmolLM-1.7B/SmolLM-1.7B-BF16.gguf -cnv

# ./build/bin/llama-cli -m /home/binguo/data/llama.cpp/ckpts/smollm1-1B7-d_kv_16/SmolLM-1.7B-d_kv_16-BF16.gguf -cnv

# Run llama-cli -no-cnv

# model_path="/home/binguo/data/models/HuggingFaceTB/SmolLM-135M/SmolLM-135M-BF16.gguf"

# model_path="/home/binguo/data/llama.cpp/ckpts/smollm1-135m-d_kv_16/SmolLM-135M-d_kv_16-BF16.gguf"

# model_path="/home/binguo/data/models/HuggingFaceTB/SmolLM-360M/SmolLM-360M-BF16.gguf"

# model_path="/home/binguo/data/llama.cpp/ckpts/smollm1-360m-d_kv_16/SmolLM-360M-d_kv_16-BF16.gguf"

# model_path="/home/binguo/data/models/HuggingFaceTB/SmolLM-1.7B/SmolLM-1.7B-BF16.gguf"

# model_path="/home/binguo/data/llama.cpp/ckpts/smollm1-1B7-d_kv_16/SmolLM-1.7B-d_kv_16-BF16.gguf"

# model_path="/home/binguo/data/models/meta-llama/Llama-2-7b-hf/Llama-2-7B-hf-BF16.gguf"

# model_path="/home/binguo/data/llama.cpp/ckpts/llama2-7B-d_kv_32/llama2-7B-d_kv_32-BF16.gguf"

# model_path="/home/binguo/data/llama.cpp/ckpts/qwen3-0_6B-rope16-dkv16/qwen3-0.6B-rope16-dkv16-BF16.gguf"

# ./build/bin/llama-cli -m ${model_path} -p "The transformer is a deep learning architecture based on the multi-head attention mechanism, in which text is converted to numerical representations called tokens," -n 128 -no-cnv --temp 0 --seed 42 --top-k 1

# ./build/bin/llama-eval-callback \
#   --model ${model_path} \
#   --prompt "Hi, how are you?" \
#   --seed 42 \
#   -ngl 128 > dbg.txt
