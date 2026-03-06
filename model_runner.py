"""
model_runner.py — Model inference and data extraction for the LLM visualizer.

Handles loading the SmolLM2-360M model and extracting internals
(attention, Q/K/V projections, hidden states, MLP weights, etc.)
during a forward pass for visualization.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    """Load model and tokenizer from local path (should be cached by Streamlit)."""
    model_path = "./smollm_local"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        output_attentions=True,
        output_hidden_states=True,
    )
    model.eval()
    model.to(DEVICE)
    return tokenizer, model


def get_model_config(model):
    """Extract static model config dict."""
    return {
        "num_layers": model.config.num_hidden_layers,
        "num_heads": model.config.num_attention_heads,
        "num_kv_heads": model.config.num_key_value_heads,
        "hidden_size": model.config.hidden_size,
        "head_dim": model.config.hidden_size // model.config.num_attention_heads,
        "intermediate_size": model.config.intermediate_size,
        "vocab_size": model.config.vocab_size,
        "num_params": sum(p.numel() for p in model.parameters()),
    }


def run_model(prompt: str, tokenizer, model):
    """
    Run the model on the given prompt and extract all visualization data.

    Returns a dict with tokens, attention data, Q/K/V projections,
    hidden states, MLP weights, predictions, and generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu()[0])

    # Register hooks to capture attention outputs (before residual add)
    attn_outputs = {}
    qkv_raw = {}  # {layer_idx: {'q': tensor, 'k': tensor, 'v': tensor}}
    hooks = []
    for li, layer_module in enumerate(model.model.layers):
        def make_hook(idx):
            def hook_fn(module, inp, out):
                try:
                    attn_out = out[0] if isinstance(out, tuple) else out
                    attn_outputs[idx] = attn_out.detach().cpu().float()
                except Exception:
                    attn_outputs[idx] = None
            return hook_fn
        hooks.append(layer_module.self_attn.register_forward_hook(make_hook(li)))

        # Q/K/V projection hooks
        def make_qkv_hook(idx, proj_name):
            def hook_fn(module, inp, out):
                try:
                    if idx not in qkv_raw:
                        qkv_raw[idx] = {}
                    qkv_raw[idx][proj_name] = out.detach().cpu().float()
                except Exception:
                    pass
            return hook_fn
        hooks.append(layer_module.self_attn.q_proj.register_forward_hook(make_qkv_hook(li, 'q')))
        hooks.append(layer_module.self_attn.k_proj.register_forward_hook(make_qkv_hook(li, 'k')))
        hooks.append(layer_module.self_attn.v_proj.register_forward_hook(make_qkv_hook(li, 'v')))

    with torch.no_grad():
        outputs = model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    hidden_states = outputs.hidden_states
    layer_norms = []
    for hs in hidden_states:
        layer_norms.append(float(hs.norm(dim=-1).mean()))

    # Per-token energy: L2 norm of each token's hidden state at the final layer
    final_hidden = hidden_states[-1]  # shape: [1, seq_len, hidden_size]
    token_energies = final_hidden[0].norm(dim=-1).cpu().float().tolist()
    token_energies = [round(e, 2) for e in token_energies]

    attention_data = {}
    for i, attn in enumerate(outputs.attentions):
        # Per-head: [num_heads, seq, seq]
        per_head = attn[0].cpu().numpy().tolist()
        attention_data[i] = per_head

    logits = outputs.logits[:, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_k = torch.topk(probs, 10)
    predictions = []
    softmax_data = []
    for idx, prob in zip(top_k.indices[0], top_k.values[0]):
        word = tokenizer.decode([idx.item()])
        raw_logit = float(logits[0, idx.item()])
        predictions.append({"word": word.strip(), "prob": round(float(prob), 4)})
        softmax_data.append({"word": word.strip(), "logit": round(raw_logit, 2), "prob": round(float(prob), 4)})

    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    gen_input = tokenizer(chat_prompt, return_tensors="pt")
    gen_input = {k: v.to(DEVICE) for k, v in gen_input.items()}
    gen_ids = model.generate(
        **gen_input,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(gen_ids[0][gen_input["input_ids"].shape[1]:], skip_special_tokens=True)

    embed_weight = model.model.embed_tokens.weight[:20, :20].detach().cpu().float().numpy().tolist()

    layer_activations = []
    for hs in hidden_states[1:]:
        vals = hs[0, 0, :64].cpu().numpy().tolist()
        layer_activations.append(vals)

    # MLP weight norms per layer
    mlp_norms = []
    for layer_module in model.model.layers:
        gate_norm = float(layer_module.mlp.gate_proj.weight.norm())
        mlp_norms.append(round(gate_norm, 2))

    # Per-layer detailed data for 3D visualization
    NORM_SAMPLE = 32   # number of RMSNorm weight values to send
    HEATMAP_SZ = 16    # NxN slice of MLP weight matrices
    layer_details = []
    for layer_module in model.model.layers:
        # RMSNorm weights
        ln1_w = layer_module.input_layernorm.weight[:NORM_SAMPLE].detach().cpu().float().tolist()
        ln2_w = layer_module.post_attention_layernorm.weight[:NORM_SAMPLE].detach().cpu().float().tolist()
        # MLP weight heatmaps (small slice)
        gate_w = layer_module.mlp.gate_proj.weight[:HEATMAP_SZ, :HEATMAP_SZ].detach().cpu().float().tolist()
        up_w = layer_module.mlp.up_proj.weight[:HEATMAP_SZ, :HEATMAP_SZ].detach().cpu().float().tolist()
        down_w = layer_module.mlp.down_proj.weight[:HEATMAP_SZ, :HEATMAP_SZ].detach().cpu().float().tolist()
        layer_details.append({
            "ln1_weights": [round(v, 4) for v in ln1_w],
            "ln2_weights": [round(v, 4) for v in ln2_w],
            "gate_heatmap": [[round(v, 4) for v in row] for row in gate_w],
            "up_heatmap": [[round(v, 4) for v in row] for row in up_w],
            "down_heatmap": [[round(v, 4) for v in row] for row in down_w],
        })

    # Build per-layer attention contribution data
    attn_contributions = []
    for li in range(len(model.model.layers)):
        # x = hidden state input to this layer (from outputs.hidden_states)
        seq_len = hidden_states[li].shape[1]
        vis_seq_len = min(seq_len, 12)
        
        x_matrix = []
        attn_matrix = []
        sum_matrix = []
        
        for t in range(vis_seq_len):
            x_full = hidden_states[li][0, t, :].detach().cpu().float().tolist()
            x_subset = x_full[:3] + x_full[-2:]
            x_subset = [round(v, 4) for v in x_subset]
            
            a_full = attn_outputs.get(li)
            if a_full is not None:
                a_vec = a_full[0, t, :].detach().cpu().float().tolist()
                a_subset = a_vec[:3] + a_vec[-2:]
            else:
                a_subset = [0]*5
            a_subset = [round(v, 4) for v in a_subset]
            
            s_subset = [round(x_subset[i] + a_subset[i], 4) for i in range(len(x_subset))]
            
            x_matrix.append(x_subset)
            attn_matrix.append(a_subset)
            sum_matrix.append(s_subset)
            
        attn_contributions.append({"x": x_matrix, "attn": attn_matrix, "sum": sum_matrix})

    # Extract word embeddings for each token (subset of values for visualization)
    token_embeddings = []
    with torch.no_grad():
        full_embeddings = model.model.embed_tokens(input_ids)[0]  # shape: (seq_len, hidden_size)
        for i in range(full_embeddings.shape[0]):
            emb_vec = full_embeddings[i].detach().cpu().float().tolist()
            # We want first 3 and last 2 values
            subset = emb_vec[:3] + emb_vec[-2:]
            subset = [round(v, 4) for v in subset]
            token_embeddings.append(subset)

    # Build per-head Q/K/V heatmap data for visualization
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads  # 64
    QKV_SAMPLE_DIMS = 32  # columns to sample for heatmap (from head_dim)
    qkv_data = []
    
    for li in range(len(model.model.layers)):
        layer_qkv = qkv_raw.get(li, {})
        entry = {}
        for proj_name in ('q', 'k', 'v'):
            tensor = layer_qkv.get(proj_name)
            if tensor is not None:
                # tensor shape: [1, seq_len, hidden_size]
                seq_len = tensor.shape[1]
                entry[proj_name] = []
                for h in range(num_heads):
                    start_idx = h * head_dim
                    hd = min(head_dim, tensor.shape[2] - start_idx)
                    n_dim = min(hd, QKV_SAMPLE_DIMS)
                    head_tensor = tensor[0, :, start_idx:start_idx+hd]  # [seq_len, head_dim]
                    sample = head_tensor[:, :n_dim].tolist()
                    entry[proj_name].append([[round(v, 4) for v in row] for row in sample])
                entry[f'{proj_name}_shape'] = [int(seq_len), int(head_dim)]
            else:
                entry[proj_name] = []
                entry[f'{proj_name}_shape'] = [0, 0]

        # Compute attention output O = A @ V for all heads
        attn_layer = attention_data.get(li, [])
        v_tensor = layer_qkv.get('v')
        entry['output'] = []
        if len(attn_layer) == num_heads and v_tensor is not None:
            for h in range(num_heads):
                start_idx = h * head_dim
                hd = min(head_dim, v_tensor.shape[2] - start_idx)
                n_dim = min(hd, QKV_SAMPLE_DIMS)
                A = torch.tensor(attn_layer[h], dtype=torch.float32)  # [seq, seq]
                V_h = v_tensor[0, :, start_idx:start_idx+hd].float()   # [seq, head_dim]
                O = A @ V_h                  # [seq, head_dim]
                sample = O[:, :n_dim].tolist()
                entry['output'].append([[round(float(v), 4) for v in row] for row in sample])
        
        qkv_data.append(entry)

    return {
        "tokens": [t.replace("\u2581", " ").replace("\u0120", " ") for t in tokens],
        "token_ids": input_ids[0].tolist(),
        "token_embeddings": token_embeddings,
        "layer_norms": layer_norms,
        "attention": attention_data,
        "predictions": predictions,
        "softmax_data": softmax_data,
        "generated_text": generated_text,
        "embed_sample": embed_weight,
        "layer_activations": layer_activations,
        "mlp_norms": mlp_norms,
        "layer_details": layer_details,
        "attn_contributions": attn_contributions,
        "qkv_data": qkv_data,
        "token_energies": token_energies,
    }
