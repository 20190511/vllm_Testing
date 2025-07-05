import os
import csv
import json


start_kernel = "cudaGraphLaunch"
cmd_dict = {
    "void vllm::rms_norm_kernel": ["ffn_ln"],
    "ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn": ["QKV_gen", ""]
}

'''
QKV Gen
Attention
Attn_proj
Attn_ln
FFN
FFN_ln
'''
def query_dict_clear(query: dict):
    for q in query.keys():
        query[q] = 0
    

skip_dicts = [
    "ac2g", "cudaMemcpyAsync", "cudaEventQuery", "cudaEventRecord", "Memcpy HtoD (Pinned -> Device)",
    "cudaStreamIsCapturing", "cudaLaunchKernel", "Memset (Device)", "Memcpy DtoD (Device -> Device)",
    "Memcpy DtoH (Device -> Pageable)", "cudaGraphLaunch", "cudaDriverGetVersion"
]
def read_json_read(trace_file):
    with open(trace_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    traceEvents = data["traceEvents"]
    iter_query = dict()
    query = dict()
    cuda_graph_id = []
    for trace in traceEvents:
        if start_kernel in trace["name"]:
            cuda_graph_id.append(trace['args']["correlation"])

    cur_corr_idx = 0
    query = {
                "QKV_gen": 0.0,
                "Attention": 0.0,
                "Attn_proj": 0.0,
                "Attn_ln": 0.0,
                "FFN": 0.0,
                "FFN_ln": 0.0,
    }

    atten_gem = False
    print(len(cuda_graph_id))
    for trace in traceEvents:
        if trace['name'] == start_kernel:
            iter_query[cur_corr_idx] = query
            query = {
                "QKV_gen": 0.0,
                "Attention": 0.0,
                "Attn_proj": 0.0,
                "Attn_ln": 0.0,
                "FFN": 0.0,
                "FFN_ln": 0.0,
            }
            cur_corr_idx += 1

            if cur_corr_idx == len(cuda_graph_id):
                break
            atten_gem = False

        if not 'args' in trace.keys():
            continue
        elif not 'correlation' in trace['args'].keys():
            continue
        elif trace['args']['correlation'] != cuda_graph_id[cur_corr_idx]:
            continue

        if ("wmma" in trace['name'] or "gemm" in trace['name']) and query['QKV_gen'] == 0.0:
            query['QKV_gen'] += trace['dur']
            
        # elif "vllm::rotary_embedding_kernel" in trace["name"]:
        #     query['Attention'] += trace['dur']  # Position encoding 적용도 Attention 단계로 취급
        
        elif "vllm::reshape_and_cache_flash_kernel" in trace["name"]:
            query['Attention'] += trace['dur']  
            
        elif "flash::flash_fwd_splitkv_kernel" in trace["name"]:
            query['Attention'] += trace['dur']

        elif "flash_fwd_splitkv_combine_kernel" in trace["name"]:
            query['Attention'] += trace['dur']

        elif "at::native::(anonymous namespace)::cunn_SoftMaxForward" in trace["name"]:
            query['Attention'] += trace['dur']

        elif "at::native::indexSelectLargeIndex" in trace["name"]:
            query['Attention'] += trace['dur']  # Token indexing, head별 gather 시 사용됨

        elif "gemm" in trace['name'] and atten_gem == False:
            query['Attn_proj'] += trace['dur']
            atten_gem = True

        elif "vllm::fused_add_rms_norm_kernel" in trace["name"] and query['Attn_ln'] == 0.0:
            query['Attn_ln'] += trace['dur']

        elif "gemm" in trace['name'] and atten_gem == True:
            query['FFN'] += trace['dur']  # Linear projection for output of Attention

        elif "vllm::act_and_mul_kernel" in trace["name"]:
            query['FFN'] += trace['dur']  # silu 등 비선형 활성화 함수
            
        elif "cutlass_80_tensorop_f16_s16816gemm_relu" in trace["name"]:
            query['FFN'] += trace['dur']  # relu(fused) 포함된 FFN matmul
            
        elif "vllm::fused_add_rms_norm_kernel" in trace["name"]:
            query['FFN_ln'] += trace['dur']
 
        # if cur_corr_idx == 8000:
        #     print(trace['name'])
        #     print()

    return iter_query


def dict_to_file(prefix, dicts):

    QKV_gen = {}
    attn = {}
    attn_proj = {}
    ffn = {}
    
    start = 15
    for ctx, dicts in dicts.items():
        if start == 15:
            start += 1
            continue
        QKV_gen[start] = dicts["QKV_gen"]
        attn[start] = dicts["Attention"]
        attn_proj[start] = dicts["Attn_proj"] + dicts["Attn_ln"]
        ffn[start] = dicts["FFN_ln"] + dicts["FFN"]
        start += 1

    result_dir = os.path.join("result_dict")

    with open(f"{os.path.join(result_dir, f"{prefix}_QKV.csv")}", "w", newline="") as f:
        writer = csv.writer(f)
        for key, value in QKV_gen.items():
            writer.writerow([key, value])

    with open(f"{os.path.join(result_dir, f"{prefix}_Attention.csv")}", "w", newline="") as f:
        writer = csv.writer(f)
        for key, value in attn.items():
            writer.writerow([key, value])  


    with open(f"{os.path.join(result_dir, f"{prefix}_Proj.csv")}", "w", newline="") as f:
        writer = csv.writer(f)
        for key, value in attn_proj.items():
            writer.writerow([key, value])  

    with open(f"{os.path.join(result_dir, f"{prefix}_FFN.csv")}", "w", newline="") as f:
        writer = csv.writer(f)
        for key, value in ffn.items():
            writer.writerow([key, value])  

if __name__ == "__main__":

    trace_file = "/workspace/vllm_benchmark_result/B1_gen8192_n1/4c640c172249_23018.1751292299957302740.pt.trace.json"
    # trace_file = "/workspace/vllm_benchmark_result/B32_gen8192_n1/4c640c172249_17843.1751298706077887381.pt.trace.json"
    dicts = read_json_read(trace_file)
    dict_to_file ("B1_gen8192_n1", dicts)