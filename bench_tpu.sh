BATCH_SIZE=$1
INPUT_LEN=$2
OUTPUT_LEN=$3
TP=$4
TRACE_DIR=results/B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}
export VLLM_TPU_PROFILE_DURATION_MS=2000
export VLLM_TPU_PROFILE_DELAY_MS=1000
export VLLM_XLA_USE_SPMD=1

mkdir -p $TRACE_DIR

python3 /workspace/vllm/examples/offline_inference/profiling_tpu/profiling.py \
  --model /workspace/my_llama1layer \
  --tokenizer /workspace/my_llama1layer \
    --skip_tokenizer_init \
    --input-len ${INPUT_LEN} \
    --output-len ${OUTPUT_LEN} \
    --batch-size ${BATCH_SIZE} \
    --enforce-eager \
    --profile-result-dir ${TRACE_DIR} \
    --num-iters-warmup 20 \
    --num-iters 10 \
    --tensor-parallel-size 1 \
    --json JSON > ${TRACE_DIR}/output.json
    # --json JSON > ${TRACE_DIR}/B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}.json