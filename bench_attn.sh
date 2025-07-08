BATCH_SIZE=$1
INPUT_LEN=$2
OUTPUT_LEN=$3
TRACE_DIR=results/B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}

mkdir -p $TRACE_DIR
export VLLM_TORCH_PROFILER_DIR=${TRACE_DIR}/traces

python3 vllm/benchmarks/benchmark_latency.py \
  --model /workspace/my_llama1layer \
  --tokenizer /workspace/my_llama1layer \
  --input-len $INPUT_LEN \
  --output-len $OUTPUT_LEN \
  --batch-size $BATCH_SIZE \
  --num_iters_warmup 30\
  --n 10 \
  --skip-tokenizer-init\
  # --profile > ${TRACE_DIR}/output.log
