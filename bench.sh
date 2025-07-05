BATCH_SIZE=$1
INPUT_LEN=$2
OUTPUT_LEN=$3
TP=$4
TRACE_DIR=results
export VLLM_TPU_PROFILE_DURATION_MS=2000
export VLLM_TPU_PROFILE_DELAY_MS=1000


mkdir -p $TRACE_DIR

python3 /home/baejh724/vllm/examples/offline_inference/profiling_tpu/profiling.py \
  --model /home/baejh724/my_llama1layer \
  --tokenizer /home/baejh724/my_llama1layer \
    --input-len ${INPUT_LEN} \
    --output-len ${OUTPUT_LEN} \
    --batch-size ${BATCH_SIZE} \
    --enforce-eager \
    --profile-result-dir ${TRACE_DIR} \
    --tensor-parallel-size ${TP} \
    # --json JSON > ${TRACE_DIR}/B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}.json