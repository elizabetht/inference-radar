#!/bin/bash

set -euo pipefail

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" 1>&2
}

log "Starting benchmark..."

cleanup() {
    log "Running cleanup..."
    pkill -f "benchmark_input_text"
}
trap cleanup EXIT

# Functions to run each experiment
run_vllm_mo_e() {
    local result=()
    local name="vLLM with MoE support"
    local framework="vLLM"
    local version="v0.3.2"
    local model="facebook/llama-3-70b"
    local node="spark-01"
    local status="pass"
    local notes=()
    local raw_output=""
    
    log "Running $name on $node..."
    
    # Check if venv exists
    if [ ! -d ~/bench/$framework-$version ]; then
        log "Creating new venv for $framework-$version..."
        python3 -m venv ~/bench/$framework-$version
        source ~/bench/$framework-$version/bin/activate
        pip install vllm
        deactivate
    else
        log "Using existing venv for $framework-$version..."
        source ~/bench/$framework-$version/bin/activate
    fi
    
    # Run benchmark
    log "Running benchmark..."
    ssh nvidia@${node} "set -euo pipefail; "
    ssh nvidia@${node} "(
        cd /data/hf_cache/mo_e; 
        python3 -c '
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = \"\"\"$model\"\"\"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda().half()

def benchmark(input_text, num_samples=40, max_new_tokens=100):
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    with torch.no_grad():
        for _ in range(num_samples):
            inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    end_time.record()
    torch.cuda.synchronize()
    total_time = start_time.elapsed_time(end_time)
    total_time /= 1000
    tokens_gen = max_new_tokens * num_samples
    print(f"Total time: {total_time:.2f} ms, Tokens per second: {tokens_gen / total_time:.2f} tok/s, GPU memory usage: {model.meta_data['meta_tensor'][0]['tensor_info']['shape'][1] / 1024 / 1024 / 1024:.2f} GB")
' 'A long research paper discussing' &
   "

    # Wait 60 seconds for server to start
    sleep 60
    raw_output=$(ssh nvidia@${node} "pkill -f 'benchmark_input_text'; wait")
    
    if [[ $raw_output =~ Total\ time: ]]; then
        status="pass"
        metrics=$(echo $raw_output | grep -oP '(?<=Total time: )[^ ]+(?= ms)' -P | tr '\n' ',' | sed 's/,$//')
        metrics="${metrics},tokens_gen_${metrics// /_},gpu_memory_gb_${metrics// /_}"
        notes=()
    else
        status="fail"
        metrics=""
        notes=("Failed to generate benchmark output")
    fi

    echo "{\"name\": \"$name\", \"framework\": \"$framework\", \"version\": \"$version\", \"model\": \"$model\", \"node\": \"$node\", \"status\": \"$status\", \"metrics\": {\"ttft_p50_ms\": $metrics, \"tpot_p50_ms\": $metrics, \"gpu_memory_gb\": ${metrics#*,}, \"mfu_pct\": ""}, \"notes\": [\"$\"\"$ "\", \"Raw output: $raw_output\"], \"raw_output\": \"$raw_output\"}"
}

# Run all experiments
experiments=(run_vllm_mo_e)
results=()
for exp in "${experiments[@]}"; do
    echo "Running $exp..."
    result=$($exp)
    results+=("$result")
done

# Log results as JSON array to stdout
echo "[${results[*]}]"