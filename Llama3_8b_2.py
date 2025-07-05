import torch
from safetensors.torch import save_file

def generate_dummy_llama3_weights(hidden_size=4096, vocab_size=128256, num_layers=1):
    state_dict = {}

    # 임베딩
    state_dict["model.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size)

    # Layer 0
    for l in range(num_layers):
        prefix = f"model.layers.{l}"
        state_dict[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)

        intermediate_size = 14336
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(intermediate_size, hidden_size)
        state_dict[f"{prefix}.mlp.up_proj.weight"] = torch.randn(intermediate_size, hidden_size)
        state_dict[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden_size, intermediate_size)

        state_dict[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden_size)
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(hidden_size)

    # 출력 projection
    state_dict["model.norm.weight"] = torch.ones(hidden_size)
    state_dict["lm_head.weight"] = torch.randn(vocab_size, hidden_size)

    # 저장
    save_file(state_dict, "./my_llama1layer/model.safetensors")

generate_dummy_llama3_weights()
