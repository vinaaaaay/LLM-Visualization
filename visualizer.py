import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import time

st.set_page_config(layout="wide", page_title="LLM Visualization — SmolLM2-360M", page_icon="⚡")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=3000, key="datarefresh")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

@st.cache_resource
def load_model():
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

tokenizer, model = load_model()

# Static model config (always available, even before any prompt)
MODEL_CONFIG = {
    "num_layers": model.config.num_hidden_layers,
    "num_heads": model.config.num_attention_heads,
    "hidden_size": model.config.hidden_size,
    "vocab_size": model.config.vocab_size,
    "num_params": sum(p.numel() for p in model.parameters()),
}

# ---------------------------------------------------------------------------
# Read shared state
# ---------------------------------------------------------------------------
def read_shared_state():
    for path in ("shared_state.json", "shared_prompt.txt"):
        if os.path.exists(path):
            with open(path, "r") as f:
                content = f.read().strip()
            if not content:
                continue
            if path.endswith(".json"):
                try:
                    data = json.loads(content)
                    return data.get("prompt", ""), data.get("timestamp", 0)
                except json.JSONDecodeError:
                    continue
            else:
                return content, 0
    return "", 0

current_prompt, prompt_ts = read_shared_state()

# ---------------------------------------------------------------------------
# Run model & extract internals
# ---------------------------------------------------------------------------
@st.cache_data
def run_model(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu()[0])

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states
    layer_norms = []
    for hs in hidden_states:
        layer_norms.append(float(hs.norm(dim=-1).mean()))

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

    return {
        "tokens": [t.replace("\u2581", " ").replace("\u0120", " ") for t in tokens],
        "token_ids": input_ids[0].tolist(),
        "layer_norms": layer_norms,
        "attention": attention_data,
        "predictions": predictions,
        "softmax_data": softmax_data,
        "generated_text": generated_text,
        "embed_sample": embed_weight,
        "layer_activations": layer_activations,
        "mlp_norms": mlp_norms,
    }


# ---------------------------------------------------------------------------
# Build the HTML visualization
# ---------------------------------------------------------------------------
def build_html(config: dict, data: dict | None, state: str = "idle") -> str:
    """
    state: "idle" | "processing" | "complete"
    config: always has model config (num_layers, etc.)
    data: None for idle/processing, dict for complete
    """
    config_json = json.dumps(config)
    data_json = json.dumps(data) if data else "null"

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  background: #0d1117;
  color: #c9d1d9;
  font-family: 'Inter', sans-serif;
  overflow-x: hidden;
}}
.mono {{ font-family: 'JetBrains Mono', monospace; }}

/* ===== HEADER ===== */
.header {{
  text-align: center;
  padding: 16px 24px 12px;
  background: linear-gradient(180deg, #111820 0%, #0d1117 100%);
  border-bottom: 1px solid #1e2a3a;
}}
.header h1 {{ font-size: 22px; font-weight: 700; color: #e6edf3; margin-bottom: 4px; }}
.header .stats {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: #6e7681; letter-spacing: 1px;
}}
.header .stats span {{ color: #58a6ff; font-weight: 500; }}

/* ===== INPUT BAR ===== */
.input-section {{
  padding: 12px 24px;
  border-bottom: 1px solid #1e2a3a;
  display: flex;
  align-items: center;
  gap: 12px;
}}
.input-label {{
  font-size: 11px; color: #8b949e; text-transform: uppercase;
  letter-spacing: 2px; font-weight: 600; white-space: nowrap;
}}
.input-text {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 16px; color: #e6edf3; font-weight: 500;
}}
.input-text.idle {{ color: #484f58; font-style: italic; font-weight: 400; font-size: 14px; }}

/* ===== ARCHITECTURE STRIP ===== */
.arch-section {{
  padding: 12px 0;
  border-bottom: 1px solid #1e2a3a;
  background: #0a0e14;
}}
.arch-title {{
  padding: 0 24px 8px;
  font-size: 11px; color: #8b949e; text-transform: uppercase;
  letter-spacing: 2px; font-weight: 600;
  display: flex; align-items: center; gap: 8px;
}}
.arch-title .hint {{
  font-size: 10px; color: #484f58; letter-spacing: 1px;
  text-transform: none; font-weight: 400;
}}
.arch-scroll {{
  overflow-x: auto; overflow-y: hidden;
  padding: 10px 24px 14px;
  display: flex; align-items: center; gap: 0;
  scrollbar-width: thin; scrollbar-color: #1e2a3a transparent;
}}
.arch-scroll::-webkit-scrollbar {{ height: 6px; }}
.arch-scroll::-webkit-scrollbar-track {{ background: transparent; }}
.arch-scroll::-webkit-scrollbar-thumb {{ background: #1e2a3a; border-radius: 3px; }}

.a-block {{
  flex-shrink: 0;
  border: 1px solid; border-radius: 5px;
  padding: 6px 10px; text-align: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; font-weight: 500;
  letter-spacing: 1px; text-transform: uppercase;
  cursor: pointer; transition: all 0.25s;
}}
.a-block:hover {{ transform: scale(1.08); z-index: 10; }}
.a-block .blabel {{ font-size: 9px; white-space: nowrap; }}
.a-block .bdetail {{ font-size: 7px; opacity: 0.5; text-transform: none; letter-spacing: 0; margin-top: 1px; white-space: nowrap; }}

.a-block.embed {{ background: rgba(88,166,255,0.10); border-color: rgba(88,166,255,0.4); color: #58a6ff; }}
.a-block.ln {{ background: rgba(63,185,80,0.06); border-color: rgba(63,185,80,0.25); color: #3fb950; padding: 4px 6px; }}
.a-block.attn {{ background: rgba(188,140,255,0.10); border-color: rgba(188,140,255,0.4); color: #bc8cff; }}
.a-block.mlp {{ background: rgba(240,136,62,0.10); border-color: rgba(240,136,62,0.4); color: #f0883e; }}
.a-block.out {{ background: rgba(57,210,192,0.10); border-color: rgba(57,210,192,0.4); color: #39d2c0; }}
.a-block.active {{ transform: scale(1.1); z-index: 10; box-shadow: 0 0 18px rgba(188,140,255,0.3); border-color: #bc8cff !important; }}

/* Processing glow animation */
.a-block.processing {{
  animation: procGlow 2s ease-in-out infinite;
}}
@keyframes procGlow {{
  0%, 100% {{ filter: brightness(1); }}
  50% {{ filter: brightness(1.8); }}
}}

.layer-grp {{
  flex-shrink: 0;
  display: flex; align-items: center; gap: 0;
  padding: 2px 4px;
  border: 1px solid transparent; border-radius: 6px;
  position: relative; cursor: pointer; transition: all 0.25s;
}}
.layer-grp:hover {{ border-color: #2d3a4a; background: rgba(255,255,255,0.01); }}
.layer-grp.active {{ border-color: rgba(188,140,255,0.3); background: rgba(188,140,255,0.04); }}
.layer-grp .ltag {{
  position: absolute; top: -9px; left: 50%; transform: translateX(-50%);
  background: #0a0e14; padding: 0 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 7px; color: #484f58; letter-spacing: 1px; white-space: nowrap;
}}

/* Processing wave on layer groups */
.layer-grp.wave {{
  animation: layerWave 0.6s ease-out forwards;
}}
@keyframes layerWave {{
  0% {{ border-color: transparent; background: transparent; }}
  30% {{ border-color: #bc8cff88; background: rgba(188,140,255,0.15); }}
  100% {{ border-color: transparent; background: transparent; }}
}}

.arrow {{
  flex-shrink: 0; width: 14px; height: 2px;
  background: linear-gradient(90deg, #3fb95055, #3fb95020);
  position: relative;
}}
.arrow::after {{
  content: ''; position: absolute; right: 0; top: -3px;
  border-left: 5px solid #3fb95033;
  border-top: 4px solid transparent;
  border-bottom: 4px solid transparent;
}}
/* Processing: animated flow dots on arrows */
.arrow.flow::before {{
  content: '';
  position: absolute; top: -2px; left: 0;
  width: 6px; height: 6px;
  background: #3fb950; border-radius: 50%;
  box-shadow: 0 0 8px #3fb950;
  animation: flowDot 0.8s ease-in-out infinite;
}}
@keyframes flowDot {{
  0% {{ left: 0; opacity: 1; }}
  100% {{ left: 14px; opacity: 0; }}
}}

.arrow-inner {{
  flex-shrink: 0; width: 5px; height: 1px; background: #ffffff0d;
}}

/* ===== PANELS ===== */
.panels-grid {{
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 14px; padding: 16px 24px 24px;
}}
.panel {{
  background: #111927; border: 1px solid #1e2a3a;
  border-radius: 8px; padding: 16px 20px; transition: border-color 0.3s;
}}
.panel:hover {{ border-color: #2d3a4a; }}
.panel-step {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; font-weight: 700; letter-spacing: 2px;
  text-transform: uppercase;
  display: flex; align-items: center; gap: 8px;
}}
.panel-step .dot {{ width: 7px; height: 7px; border-radius: 50%; }}
.panel-title {{ font-size: 15px; font-weight: 600; color: #e6edf3; margin: 2px 0 4px; }}
.panel-desc {{ font-size: 12px; color: #8b949e; line-height: 1.5; margin-bottom: 12px; }}
.panel-desc em {{ color: #58a6ff; font-style: normal; font-weight: 500; }}

/* Idle panel placeholder */
.panel-idle {{
  text-align: center; padding: 30px 20px; color: #2d3a4a;
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  letter-spacing: 1px;
}}
.panel-idle .idle-icon {{ font-size: 28px; margin-bottom: 8px; opacity: 0.4; }}

/* Processing panel */
.panel-proc {{
  text-align: center; padding: 30px 20px;
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  letter-spacing: 1px; color: #8b949e;
}}
.panel-proc .mini-spin {{
  width: 24px; height: 24px; margin: 0 auto 10px;
  border: 2px solid #1e2a3a; border-top-color: #bc8cff;
  border-radius: 50%; animation: spin 1s linear infinite;
}}

/* Tokens */
.token-flow {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.token-chip {{
  background: #161b26; border: 1px solid #252d3a;
  padding: 6px 10px 4px; border-radius: 5px; text-align: center;
  animation: chipIn 0.4s ease-out backwards;
}}
.token-chip .tword {{ font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #e6edf3; font-weight: 500; }}
.token-chip .tid {{ font-family: 'JetBrains Mono', monospace; font-size: 8px; color: #484f58; margin-top: 2px; }}
@keyframes chipIn {{ from {{ transform: translateY(8px); opacity: 0; }} to {{ transform: translateY(0); opacity: 1; }} }}

/* Attention */
.attn-controls {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
.attn-controls button {{
  background: #1e2a3a; border: 1px solid #2d3a4a; color: #8b949e;
  padding: 4px 12px; border-radius: 4px; cursor: pointer;
  font-family: 'JetBrains Mono', monospace; font-size: 12px; transition: all 0.2s;
}}
.attn-controls button:hover {{ background: #2d3a4a; color: #e6edf3; }}
.attn-controls .lnum {{ font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #bc8cff; font-weight: 700; min-width: 100px; text-align: center; }}
.heatmap-wrap {{ overflow-x: auto; padding-bottom: 8px; }}
.heatmap-grid {{ display: grid; gap: 2px; width: fit-content; }}
.hm-label {{ font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #484f58; text-align: center; overflow: hidden; white-space: nowrap; }}
.hm-labels {{ display: flex; gap: 2px; margin-bottom: 3px; }}
.hm-cell {{ width: 34px; height: 34px; border-radius: 3px; cursor: crosshair; transition: transform 0.15s; }}
.hm-cell:hover {{ transform: scale(1.3); z-index: 10; }}
.tooltip {{
  display: none; position: fixed;
  background: #1c2333; border: 1px solid #3d4a5c; border-radius: 5px;
  padding: 6px 10px; font-family: 'JetBrains Mono', monospace; font-size: 11px;
  color: #e6edf3; z-index: 9999; pointer-events: none; white-space: nowrap;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}}

/* Predictions */
.pred-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; animation: barIn 0.5s ease-out backwards; }}
.pred-rank {{ font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #484f58; width: 18px; text-align: right; }}
.pred-word {{ font-family: 'JetBrains Mono', monospace; width: 90px; text-align: right; font-size: 12px; color: #e6edf3; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.pred-bar-bg {{ flex: 1; height: 20px; background: #161b26; border-radius: 4px; overflow: hidden; }}
.pred-bar {{ height: 100%; border-radius: 4px; transition: width 0.8s cubic-bezier(0.22,1,0.36,1); }}
.pred-pct {{ font-family: 'JetBrains Mono', monospace; width: 55px; font-size: 11px; color: #6e7681; }}
@keyframes barIn {{ from {{ transform: translateX(-10px); opacity: 0; }} to {{ transform: translateX(0); opacity: 1; }} }}

/* Energy */
.energy-chart {{ width: 100%; height: 55px; }}
.energy-label {{ font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #484f58; margin-top: 4px; text-align: center; }}

/* Response */
.response-box {{
  background: #0a0e14; border: 1px solid #1e2a3a; border-radius: 6px;
  padding: 16px 18px; font-size: 14px; color: #e6edf3; line-height: 1.7;
  max-height: 220px; overflow-y: auto; font-family: 'Inter', sans-serif;
}}
.cursor-blink {{
  display: inline-block; width: 9px; height: 16px; background: #58a6ff;
  animation: blink 1s step-end infinite; vertical-align: text-bottom;
  margin-left: 2px; border-radius: 1px;
}}
@keyframes blink {{ 50% {{ opacity: 0; }} }}
/* Softmax detail panel */
.softmax-overlay {{
  display: none;
  background: #111927; border: 1px solid #39d2c066;
  border-radius: 8px; padding: 20px 24px;
  margin: 0 24px 4px;
  animation: slideDown 0.3s ease-out;
}}
.softmax-overlay.show {{ display: block; }}
@keyframes slideDown {{ from {{ opacity:0; transform:translateY(-10px); }} to {{ opacity:1; transform:translateY(0); }} }}
.softmax-title {{ font-size: 14px; font-weight: 600; color: #39d2c0; margin-bottom: 4px; }}
.softmax-desc {{ font-size: 12px; color: #8b949e; margin-bottom: 14px; line-height: 1.5; }}
.softmax-desc em {{ color: #39d2c0; font-style: normal; font-weight: 500; }}
.sm-grid {{ display: grid; grid-template-columns: 90px 1fr 40px 20px 1fr 55px; gap: 4px 8px; align-items: center; }}
.sm-word {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #e6edf3; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.sm-bar-bg {{ height: 18px; background: #161b26; border-radius: 3px; overflow: hidden; }}
.sm-bar {{ height: 100%; border-radius: 3px; transition: width 0.6s cubic-bezier(0.22,1,0.36,1); }}
.sm-val {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #6e7681; }}
.sm-arrow {{ font-size: 12px; color: #39d2c0; text-align: center; }}
.sm-header {{ font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #484f58; letter-spacing: 1px; text-transform: uppercase; padding-bottom: 4px; border-bottom: 1px solid #1e2a3a; margin-bottom: 2px; }}
.sm-close {{ float: right; background: none; border: 1px solid #2d3a4a; color: #8b949e; padding: 2px 10px; border-radius: 4px; cursor: pointer; font-size: 11px; font-family: 'JetBrains Mono', monospace; }}
.sm-close:hover {{ background: #2d3a4a; color: #e6edf3; }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}

::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: #1e2a3a; border-radius: 3px; }}
</style>
</head>
<body>

<div class="tooltip" id="tooltip"></div>

<div class="header">
  <h1>🧠 Inside an AI's Brain</h1>
  <div class="stats" id="stats"></div>
</div>

<div id="app"></div>

<script>
const CFG = {config_json};
const DATA = {data_json};
const STATE = "{state}";
let currentLayer = 0;

function init() {{
  const fmt = CFG.num_params > 1e9 ? (CFG.num_params/1e9).toFixed(1)+'B' : (CFG.num_params/1e6).toFixed(0)+'M';
  let statusText = STATE === 'idle' ? 'Ready — type a prompt in the terminal'
    : STATE === 'processing' ? '<span style="color:#bc8cff;">⚡ Processing...</span>'
    : 'Visualization active';
  document.getElementById('stats').innerHTML =
    `SmolLM2 — <span>${{fmt}} params</span> · <span>${{CFG.num_layers}} layers</span> · <span>${{CFG.num_heads}} heads</span> · <span>hidden ${{CFG.hidden_size}}</span> · ${{statusText}}`;

  if (DATA) currentLayer = Math.min(15, CFG.num_layers - 1);
  render();
  if (STATE === 'processing') startProcessingAnim();
}}

function render() {{
  let h = '';
  // Input bar
  if (DATA) {{
    h += `<div class="input-section">
      <div class="input-label">Your Input</div>
      <div class="input-text">${{esc(DATA.tokens.join(''))}}</div>
    </div>`;
  }} else if (STATE === 'processing') {{
    h += `<div class="input-section">
      <div class="input-label">Your Input</div>
      <div class="input-text" style="color:#bc8cff;">Processing...</div>
    </div>`;
  }} else {{
    h += `<div class="input-section">
      <div class="input-label">Your Input</div>
      <div class="input-text idle">Waiting for input from terminal...</div>
    </div>`;
  }}

  // Architecture strip (ALWAYS rendered)
  h += renderArch();
  h += renderSoftmaxPanel();

  // Panels
  h += '<div class="panels-grid">';
  if (DATA) {{
    h += renderStep1();
    h += renderStep2();
    h += renderStep3();
    h += renderStep4();
  }} else if (STATE === 'processing') {{
    h += procPanel('Step 1', '58a6ff', 'Reading the Words', 'Tokenizing input...');
    h += procPanel('Step 2', 'bc8cff', 'Connecting the Dots', 'Computing attention...');
    h += procPanel('Step 3', '39d2c0', 'What Comes Next?', 'Ranking predictions...');
    h += procPanel('Step 4', '3fb950', 'The AI Speaks', 'Generating response...');
  }} else {{
    h += idlePanel('Step 1', '58a6ff', 'Reading the Words', 'The AI breaks your sentence into tokens', '📝');
    h += idlePanel('Step 2', 'bc8cff', 'Connecting the Dots', 'See which words the AI connects', '🔗');
    h += idlePanel('Step 3', '39d2c0', 'What Comes Next?', 'Watch the AI rank its word predictions', '📊');
    h += idlePanel('Step 4', '3fb950', 'The AI Speaks', 'The AI builds a response token by token', '💬');
  }}
  h += '</div>';

  document.getElementById('app').innerHTML = h;
  if (DATA) attachEvents();
}}

function idlePanel(step, color, title, desc, icon) {{
  return `<div class="panel">
    <div class="panel-step"><span class="dot" style="background:#${{color}};"></span>${{step}}</div>
    <div class="panel-title">${{title}}</div>
    <div class="panel-idle"><div class="idle-icon">${{icon}}</div>${{desc}}</div>
  </div>`;
}}

function procPanel(step, color, title, msg) {{
  return `<div class="panel">
    <div class="panel-step"><span class="dot" style="background:#${{color}};"></span>${{step}}</div>
    <div class="panel-title">${{title}}</div>
    <div class="panel-proc"><div class="mini-spin"></div>${{msg}}</div>
  </div>`;
}}

/* ─── PROCESSING ANIMATION ─── */
function startProcessingAnim() {{
  const groups = document.querySelectorAll('.layer-grp');
  const arrows = document.querySelectorAll('.arrow');
  let idx = 0;

  // Add flow to all arrows
  arrows.forEach(a => a.classList.add('flow'));

  // Wave through layers sequentially
  function wave() {{
    if (idx < groups.length) {{
      groups[idx].classList.add('wave');
      // Also pulse the blocks inside
      groups[idx].querySelectorAll('.a-block').forEach(b => b.classList.add('processing'));
      setTimeout(() => {{
        groups[idx].querySelectorAll('.a-block').forEach(b => b.classList.remove('processing'));
      }}, 500);
      idx++;
      setTimeout(wave, 80);
    }} else {{
      // Loop
      setTimeout(() => {{
        groups.forEach(g => g.classList.remove('wave'));
        idx = 0;
        wave();
      }}, 800);
    }}
  }}
  wave();
}}

/* ─── ARCHITECTURE STRIP ─── */
function renderArch() {{
  let h = '<div class="arch-section">';
  h += '<div class="arch-title">Full Neural Network Architecture <span class="hint">← scroll to explore all ' + CFG.num_layers + ' layers' + (DATA ? ' · click to inspect' : '') + ' →</span></div>';
  h += '<div class="arch-scroll" id="arch-scroll">';

  h += ab('embed','Embed',CFG.hidden_size+'d',false);
  h += aw();

  for (let i = 0; i < CFG.num_layers; i++) {{
    const act = DATA ? (i === currentLayer) : false;
    h += `<div class="layer-grp ${{act?'active':''}}" ${{DATA ? 'onclick="selectLayer('+i+')"' : ''}}>`;
    h += `<div class="ltag">L${{i}}</div>`;
    h += ab('ln','LN','',act);
    h += ai();
    h += ab('attn','Attn',CFG.num_heads+'h',act);
    h += ai();
    h += ab('ln','LN','',act);
    h += ai();
    h += ab('mlp','MLP','',act);
    h += '</div>';
    if (i < CFG.num_layers - 1) h += aw();
  }}

  h += aw();
  h += ab('ln','LN','final',false);
  h += aw();
  h += ab('out','Softmax',CFG.vocab_size.toLocaleString(),false, true);
  h += '</div></div>';
  return h;
}}
function ab(c,l,d,a,isSoftmax) {{
  const click = isSoftmax && DATA ? ' onclick="toggleSoftmax()"' : '';
  return `<div class="a-block ${{c}} ${{a?'active':''}}"${{click}}><div class="blabel">${{l}}</div>${{d?'<div class="bdetail">'+d+'</div>':''}}</div>`;
}}
function aw() {{ return '<div class="arrow"></div>'; }}
function ai() {{ return '<div class="arrow-inner"></div>'; }}

function selectLayer(i) {{
  currentLayer = i;
  render();
  setTimeout(() => {{
    const el = document.querySelector('.layer-grp.active');
    if (el) el.scrollIntoView({{behavior:'smooth',inline:'center',block:'nearest'}});
  }}, 50);
}}

/* ─── STEP 1 ─── */
function renderStep1() {{
  let h = '<div class="panel">';
  h += '<div class="panel-step"><span class="dot" style="background:#58a6ff;"></span>Step 1</div>';
  h += '<div class="panel-title">Reading the Words</div>';
  h += '<div class="panel-desc">The AI breaks your sentence into pieces called <em>tokens</em> and converts each to a number. It knows <em>' + CFG.vocab_size.toLocaleString() + ' words</em>.</div>';
  h += '<div class="token-flow">';
  DATA.tokens.forEach((t,i) => {{
    h += `<div class="token-chip" style="animation-delay:${{i*0.08}}s"><div class="tword">${{esc(t||'·')}}</div><div class="tid">${{DATA.token_ids[i]}}</div></div>`;
  }});
  h += '</div></div>';
  return h;
}}

/* ─── STEP 2 ─── */
function renderStep2() {{
  let h = '<div class="panel" style="grid-column: 1 / -1;">';
  h += '<div class="panel-step"><span class="dot" style="background:#bc8cff;"></span>Step 2</div>';
  h += '<div class="panel-title">Full Transformer Architecture — 3D View</div>';
  h += '<div class="panel-desc">The entire <em>' + CFG.num_layers + '-layer</em> transformer with <em>' + CFG.num_heads + ' attention heads</em> per layer. Each layer: <em>LN → Attention (heads fan out) → LN → MLP</em>. Drag to rotate, scroll to zoom, right-drag to pan.</div>';
  h += '<div class="attn-controls">';
  h += '<button onclick="resetCam()">⟲ Reset Camera</button>';
  h += '<button onclick="toggleFullscreen()" style="margin-left: 10px;">⛶ Fullscreen</button>';
  h += '</div>';
  h += '<div id="three-container" style="position:relative;border:1px solid #1e2a3a;border-radius:6px;overflow:hidden;background:#080b10;height:600px;"></div>';
  h += '</div>';
  return h;
}}



/* ─── Three.js 3D Attention Head Viewer ─── */
let threeScene, threeCamera, threeRenderer, threeControls, threeAnimId;
let defaultCamPos = null;

function resetCam() {{
  if (threeCamera && defaultCamPos) {{
    threeCamera.position.copy(defaultCamPos);
    threeCamera.lookAt(0, 0, 0);
    if (threeControls) threeControls.reset();
  }}
}}

function toggleFullscreen() {{
  const container = document.getElementById('three-container');
  if (!container) return;
  if (!document.fullscreenElement) {{
    if (container.requestFullscreen) container.requestFullscreen();
    else if (container.webkitRequestFullscreen) container.webkitRequestFullscreen();
    else if (container.msRequestFullscreen) container.msRequestFullscreen();
  }} else {{
    if (document.exitFullscreen) document.exitFullscreen();
    else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
    else if (document.msExitFullscreen) document.msExitFullscreen();
  }}
}}

function initHeadCanvas() {{
  if (!DATA || typeof THREE === 'undefined') return;
  const container = document.getElementById('three-container');
  if (!container) return;

  // Cleanup previous
  if (threeAnimId) cancelAnimationFrame(threeAnimId);
  if (threeRenderer) {{
    threeRenderer.dispose();
    container.innerHTML = '';
  }}

  const w = container.clientWidth;
  const h = container.clientHeight;

  // Scene
  threeScene = new THREE.Scene();
  threeScene.background = new THREE.Color(0x080b10);

  // Camera
  threeCamera = new THREE.PerspectiveCamera(50, w / h, 0.1, 2000);

  // Renderer
  threeRenderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
  threeRenderer.setPixelRatio(window.devicePixelRatio);
  threeRenderer.setSize(w, h);
  container.appendChild(threeRenderer.domElement);

  // Controls
  threeControls = new THREE.OrbitControls(threeCamera, threeRenderer.domElement);
  threeControls.enableDamping = true;
  threeControls.dampingFactor = 0.08;
  threeControls.rotateSpeed = 0.6;
  threeControls.zoomSpeed = 1.2;
  threeControls.panSpeed = 3.0;
  threeControls.screenSpacePanning = true;
  threeControls.minDistance = 1;
  threeControls.maxDistance = 2000;

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  threeScene.add(ambientLight);
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.4);
  dirLight.position.set(5, 10, 7);
  threeScene.add(dirLight);

  // Build the full architecture scene
  buildFullArchScene();

  // Window resize handler
  function onWindowResize() {{
    if (!threeCamera || !threeRenderer) return;
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    threeCamera.aspect = cw / ch;
    threeCamera.updateProjectionMatrix();
    threeRenderer.setSize(cw, ch);
  }}
  window.addEventListener('resize', onWindowResize);

  // Animate loop
  function animate() {{
    threeAnimId = requestAnimationFrame(animate);
    threeControls.update();
    threeRenderer.render(threeScene, threeCamera);
  }}
  animate();
}}

function buildFullArchScene() {{
  if (!DATA || !threeScene) return;

  // Clear old objects
  const toRemove = [];
  threeScene.traverse(child => {{ if (child.userData.arch) toRemove.push(child); }});
  toRemove.forEach(c => threeScene.remove(c));

  const numLayers = CFG.num_layers;
  const numHeads = CFG.num_heads;
  const seqLen = DATA.tokens.length;

  // Layout constants
  const layerSpacing = 8;       // distance between layers along X
  const blockH = 1.0;           // height of LN/MLP blocks
  const blockW = 1.5;           // width
  const blockD = 1.0;           // depth
  const headPlaneSize = 1.2;    // size of each attention head plane
  const headDepthSpacing = headPlaneSize + 0.15; // Z spacing — tile side by side
  const subSpacing = 2.5;       // spacing within a layer (between LN, Attn, MLP)

  const totalX = (numLayers + 2) * layerSpacing;
  let cursor = -totalX / 2;

  // Helper: create a labeled box
  function addBox(x, y, z, w, h, d, color, label) {{
    const geo = new THREE.BoxGeometry(w, h, d);
    const mat = new THREE.MeshPhongMaterial({{
      color: color,
      transparent: true,
      opacity: 0.85,
      shininess: 30,
    }});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, y, z);
    mesh.userData.arch = true;
    threeScene.add(mesh);

    // Edges
    const edges = new THREE.EdgesGeometry(geo);
    const lineMat = new THREE.LineBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.15 }});
    const wire = new THREE.LineSegments(edges, lineMat);
    wire.position.copy(mesh.position);
    wire.userData.arch = true;
    threeScene.add(wire);

    // Label sprite
    if (label) {{
      const lc = document.createElement('canvas');
      lc.width = 256;
      lc.height = 48;
      const lx = lc.getContext('2d');
      lx.fillStyle = '#e6edf3';
      lx.font = 'bold 18px JetBrains Mono, monospace';
      lx.textAlign = 'center';
      lx.fillText(label, 128, 30);
      const lt = new THREE.CanvasTexture(lc);
      const sm = new THREE.SpriteMaterial({{ map: lt, transparent: true }});
      const sp = new THREE.Sprite(sm);
      sp.position.set(x, y + h / 2 + 0.5, z);
      sp.scale.set(2.5, 0.5, 1);
      sp.userData.arch = true;
      threeScene.add(sp);
    }}

    return mesh;
  }}

  // Helper: connect two points
  function addLine(p1, p2, color) {{
    const geo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
    const mat = new THREE.LineBasicMaterial({{ color: color || 0x3fb950, transparent: true, opacity: 0.35 }});
    const line = new THREE.Line(geo, mat);
    line.userData.arch = true;
    threeScene.add(line);
  }}

  // Helper: create attention head texture plane
  function addHeadPlane(x, y, z, headIdx, layerIdx) {{
    const heads = DATA.attention[layerIdx];
    if (!heads || !heads[headIdx]) return;
    const headData = heads[headIdx];
    const sl = headData.length;
    const texSz = Math.max(sl * 4, 32);
    const hc = document.createElement('canvas');
    hc.width = texSz;
    hc.height = texSz;
    const cx = hc.getContext('2d');
    const cp = texSz / sl;
    for (let r = 0; r < sl; r++) {{
      for (let c = 0; c < sl; c++) {{
        const v = Math.min(1, Math.max(0, headData[r][c]));
        cx.fillStyle = `rgb(${{Math.round(30+v*150)}},${{Math.round(20+v*70)}},${{Math.round(60+v*195)}})`;
        cx.fillRect(c * cp, r * cp, cp, cp);
      }}
    }}
    const tex = new THREE.CanvasTexture(hc);
    tex.magFilter = THREE.NearestFilter;
    tex.minFilter = THREE.NearestFilter;
    const geo = new THREE.PlaneGeometry(headPlaneSize, headPlaneSize);
    const mat = new THREE.MeshBasicMaterial({{ map: tex, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, y, z);
    mesh.rotation.x = -Math.PI / 2;  // lay flat, face up
    mesh.userData.arch = true;
    threeScene.add(mesh);

    // Edge
    const edg = new THREE.EdgesGeometry(geo);
    const lm = new THREE.LineBasicMaterial({{ color: 0xbc8cff, transparent: true, opacity: 0.3 }});
    const wf = new THREE.LineSegments(edg, lm);
    wf.position.copy(mesh.position);
    wf.rotation.copy(mesh.rotation);
    wf.userData.arch = true;
    threeScene.add(wf);
  }}

  // ═══ EMBEDDING BLOCK — DETAILED ═══
  const embedStartX = cursor;
  let embedOutEdgeX = embedStartX + blockW * 0.6;

  function addTextSprite(x, y, z, text, color, fontSize, scaleW, scaleH) {{
    const c = document.createElement('canvas');
    c.width = 256;
    c.height = 64;
    const ctx = c.getContext('2d');
    ctx.fillStyle = color;
    ctx.font = 'bold ' + fontSize + 'px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(text, 128, 40);
    const tex = new THREE.CanvasTexture(c);
    const mat = new THREE.SpriteMaterial({{ map: tex, transparent: true }});
    const sprite = new THREE.Sprite(mat);
    sprite.position.set(x, y, z);
    sprite.scale.set(scaleW || 3.0, scaleH || 0.75, 1);
    sprite.userData.arch = true;
    threeScene.add(sprite);
    return sprite;
  }}

  if (DATA.tokens && DATA.token_ids) {{
    const numTokens = Math.min(DATA.tokens.length, 12);
    const zStep = 1.2;
    const totalZ = (numTokens - 1) * zStep;
    
    const colTokX = embedStartX;
    const colIdxX = embedStartX + 3;
    const colEmbX = embedStartX + 7;
    const colPosX = embedStartX + 10;
    const embedFnX = colPosX + 4;
    embedOutEdgeX = embedFnX + blockW * 0.75;

    addTextSprite(colTokX, 2, 0, 'Token', '#e6edf3', 22);
    addTextSprite(colIdxX, 2, 0, 'ID', '#d2a8ff', 22);
    addTextSprite(colEmbX, 2, 0, 'Word Vector', '#58a6ff', 22);
    addTextSprite(colPosX, 2, 0, '+ Pos Enc', '#3fb950', 22);

    for (let i = 0; i < numTokens; i++) {{
      const zPos = -totalZ / 2 + i * zStep;
      
      const tokStr = ('"' + DATA.tokens[i] + '"').slice(0, 12);
      addTextSprite(colTokX, 0, zPos, tokStr, '#e6edf3', 18);
      
      const idStr = DATA.token_ids[i].toString();
      addTextSprite(colIdxX, 0, zPos, idStr, '#d2a8ff', 16);
      
      addLine(new THREE.Vector3(colTokX+1.2, 0, zPos), new THREE.Vector3(colIdxX-1.2, 0, zPos), 0x484f58);

      // Embedding vector block
      addBox(colEmbX, 0, zPos, 2.5, 0.3, 0.8, 0x58a6ff);
      addLine(new THREE.Vector3(colIdxX+1.2, 0, zPos), new THREE.Vector3(colEmbX-1.5, 0, zPos), 0x484f58);

      // Positional encoding vector block
      addBox(colPosX, 0, zPos, 2.5, 0.3, 0.8, 0x3fb950);
      addTextSprite(colPosX - 1.8, 0, zPos, '+', '#ffffff', 24, 1.0, 0.25);
      addLine(new THREE.Vector3(colEmbX+1.5, 0, zPos), new THREE.Vector3(colPosX-1.5, 0, zPos), 0x484f58);

      // Connect to final unified block
      addLine(new THREE.Vector3(colPosX+1.5, 0, zPos), new THREE.Vector3(embedFnX-blockW*0.6-0.1, 0, 0), 0x58a6ff);
    }}
    
    addBox(embedFnX, 0, 0, blockW * 1.5, blockH * 1.5, blockD * 1.5, 0x1f6feb, 'Embed Output');
    cursor = embedFnX + layerSpacing;
  }} else {{
    // Fallback if no detailed data
    addBox(embedStartX, 0, 0, blockW * 1.2, blockH * 1.5, blockD * 1.2, 0x58a6ff, 'Embed');
    cursor += layerSpacing;
  }}

  // ═══ TRANSFORMER LAYERS ═══
  for (let li = 0; li < numLayers; li++) {{
    const layerX = cursor;
    const normVal = DATA.layer_norms[li + 1] || 1;
    const mlpVal = (DATA.mlp_norms && DATA.mlp_norms[li]) || 1;

    // LN1 block - RMSNorm
    const ln1X = layerX;
    const ln1Color = new THREE.Color().setHSL(0.08, 0.7, 0.3 + Math.min(normVal / 30, 0.4));
    addBox(ln1X, 0, 0, blockW * 0.6, blockH * 0.8, blockD * 0.6, ln1Color, 'RMSNorm');

    // Connector from LN1 to Attention heads
    const attnCenterX = layerX + subSpacing;
    addLine(
      new THREE.Vector3(ln1X + blockW * 0.3 + 0.1, 0, 0),
      new THREE.Vector3(attnCenterX - headPlaneSize / 2 - 0.3, 0, 0),
      0xbc8cff
    );

    // ATTENTION HEADS - flat tiles fanned out along Z
    const headTotalZ = (numHeads - 1) * headDepthSpacing;
    for (let hi = 0; hi < numHeads; hi++) {{
      const hz = -headTotalZ / 2 + hi * headDepthSpacing;
      addHeadPlane(attnCenterX, 0, hz, hi, li);

      // Connect to LN1 (fan-in line)
      if (hi === 0 || hi === numHeads - 1 || hi === Math.floor(numHeads / 2)) {{
        addLine(
          new THREE.Vector3(ln1X + blockW * 0.3, 0, 0),
          new THREE.Vector3(attnCenterX - headPlaneSize / 2, 0, hz),
          0xbc8cff
        );
      }}
    }}

    // LN2 after attention (Add & Norm)
    const ln2X = attnCenterX + subSpacing;
    addBox(ln2X, 0, 0, blockW * 0.6, blockH * 0.8, blockD * 0.6, ln1Color, 'Add+Norm');

    // Connector from heads to LN2 (fan-out → merge)
    for (let hi = 0; hi < numHeads; hi++) {{
      if (hi === 0 || hi === numHeads - 1 || hi === Math.floor(numHeads / 2)) {{
        const hz = -headTotalZ / 2 + hi * headDepthSpacing;
        addLine(
          new THREE.Vector3(attnCenterX + headPlaneSize / 2, 0, hz),
          new THREE.Vector3(ln2X - blockW * 0.3, 0, 0),
          0xbc8cff
        );
      }}
    }}

    // MLP / Feed-Forward block
    const mlpX = ln2X + subSpacing;
    const mlpColor = new THREE.Color().setHSL(0.1, 0.8, 0.25 + Math.min(mlpVal / 200, 0.35));
    addBox(mlpX, 0, 0, blockW, blockH * 1.2, blockD, mlpColor, 'FFN');

    // LN2 → MLP connector
    addLine(
      new THREE.Vector3(ln2X + blockW * 0.3, 0, 0),
      new THREE.Vector3(mlpX - blockW / 2 - 0.1, 0, 0),
      0x3fb950
    );

    // Layer label on top
    const layerLblC = document.createElement('canvas');
    layerLblC.width = 128;
    layerLblC.height = 32;
    const llx = layerLblC.getContext('2d');
    llx.fillStyle = '#6e7681';
    llx.font = '12px JetBrains Mono, monospace';
    llx.textAlign = 'center';
    llx.fillText('L' + li, 64, 22);
    const llt = new THREE.CanvasTexture(layerLblC);
    const llsm = new THREE.SpriteMaterial({{ map: llt, transparent: true }});
    const lls = new THREE.Sprite(llsm);
    lls.position.set(attnCenterX, blockH + 1.2, 0);
    lls.scale.set(1, 0.3, 1);
    lls.userData.arch = true;
    threeScene.add(lls);

    // Connect to embedding or previous layer's MLP
    if (li === 0) {{
      addLine(
        new THREE.Vector3(embedOutEdgeX + 0.1, 0, 0),
        new THREE.Vector3(ln1X - blockW * 0.3 - 0.1, 0, 0),
        0x58a6ff
      );
    }} else {{
      const prevMlpX = layerX - layerSpacing;
      addLine(
        new THREE.Vector3(prevMlpX + blockW / 2 + 0.1, 0, 0),
        new THREE.Vector3(ln1X - blockW * 0.3 - 0.1, 0, 0),
        0x3fb950
      );
    }}

    // Advance cursor for next layer
    cursor = mlpX + layerSpacing;
  }}

  // Connect last layer MLP to final blocks
  const prevMlpX = cursor - layerSpacing;

  // ═══ FINAL LN ═══
  const finalLnX = cursor;
  addBox(finalLnX, 0, 0, blockW * 0.8, blockH, blockD * 0.8, 0xd29922, 'LN final');
  addLine(
    new THREE.Vector3(prevMlpX + blockW / 2 + 0.1, 0, 0),
    new THREE.Vector3(finalLnX - blockW * 0.4 - 0.1, 0, 0),
    0x3fb950
  );
  cursor += layerSpacing;

  // ═══ SOFTMAX — DETAILED VISUALIZATION ═══
  const softmaxX = cursor;
  // Keep a smaller label box for "Softmax"
  addBox(softmaxX, 0, 0, blockW * 0.8, blockH, blockD * 0.8, 0x39d2c0, 'Softmax');
  addLine(
    new THREE.Vector3(finalLnX + blockW * 0.4 + 0.1, 0, 0),
    new THREE.Vector3(softmaxX - blockW * 0.4 - 0.1, 0, 0),
    0x39d2c0
  );

  // If we have softmax_data, draw logit bars and probability bars
  if (DATA.softmax_data && DATA.softmax_data.length > 0) {{
    const sd = DATA.softmax_data;
    const numTokens = Math.min(sd.length, 10);
    const barW = 0.4;
    const barD = 0.4;
    const maxLogit = Math.max(...sd.map(d => Math.abs(d.logit)));
    const maxProb = sd[0]?.prob || 1;
    const logitScale = 4;    // max bar height for logits
    const probScale = 5;     // max bar height for probs
    const barSpacingZ = barD + 0.3;
    const totalBarZ = (numTokens - 1) * barSpacingZ;

    // Logit bars (left of softmax box)
    const logitGroupX = softmaxX + 3;
    // Prob bars (right)
    const probGroupX = softmaxX + 7;

    // Title sprites
    function addTitle(x, text, color) {{
      const tc = document.createElement('canvas');
      tc.width = 256;
      tc.height = 48;
      const tx = tc.getContext('2d');
      tx.fillStyle = color;
      tx.font = 'bold 20px JetBrains Mono, monospace';
      tx.textAlign = 'center';
      tx.fillText(text, 128, 32);
      const tt = new THREE.CanvasTexture(tc);
      const tm = new THREE.SpriteMaterial({{ map: tt, transparent: true }});
      const ts = new THREE.Sprite(tm);
      ts.position.set(x, logitScale + 1.5, 0);
      ts.scale.set(3, 0.6, 1);
      ts.userData.arch = true;
      threeScene.add(ts);
    }}
    addTitle(logitGroupX, 'Raw Logits', '#f47067');
    addTitle(probGroupX, 'Probabilities', '#39d2c0');

    // Connector from softmax box to logit group
    addLine(
      new THREE.Vector3(softmaxX + blockW * 0.4 + 0.1, 0, 0),
      new THREE.Vector3(logitGroupX - barW / 2 - 0.3, 0, 0),
      0x39d2c0
    );

    // Arrow between logit and prob groups
    addLine(
      new THREE.Vector3(logitGroupX + barW / 2 + 0.6, logitScale * 0.4, 0),
      new THREE.Vector3(probGroupX - barW / 2 - 0.6, probScale * 0.4, 0),
      0xffffff
    );
    // Arrow label
    const arrowC = document.createElement('canvas');
    arrowC.width = 256;
    arrowC.height = 48;
    const arx = arrowC.getContext('2d');
    arx.fillStyle = '#8b949e';
    arx.font = '16px JetBrains Mono, monospace';
    arx.textAlign = 'center';
    arx.fillText('softmax(\u00b7)', 128, 32);
    const art = new THREE.CanvasTexture(arrowC);
    const arsm = new THREE.SpriteMaterial({{ map: art, transparent: true }});
    const arsp = new THREE.Sprite(arsm);
    arsp.position.set((logitGroupX + probGroupX) / 2, logitScale * 0.4 + 0.7, 0);
    arsp.scale.set(2.5, 0.5, 1);
    arsp.userData.arch = true;
    threeScene.add(arsp);

    for (let i = 0; i < numTokens; i++) {{
      const d = sd[i];
      const zPos = -totalBarZ / 2 + i * barSpacingZ;

      // --- LOGIT BAR ---
      const logitH = Math.max(0.15, (Math.abs(d.logit) / maxLogit) * logitScale);
      const logitGeo = new THREE.BoxGeometry(barW, logitH, barD);
      const logitColor = d.logit >= 0 ? 0xf47067 : 0x6e7681;
      const logitMat = new THREE.MeshPhongMaterial({{
        color: logitColor,
        transparent: true,
        opacity: 0.85,
        shininess: 40,
      }});
      const logitMesh = new THREE.Mesh(logitGeo, logitMat);
      logitMesh.position.set(logitGroupX, logitH / 2, zPos);
      logitMesh.userData.arch = true;
      threeScene.add(logitMesh);

      // Logit edge
      const logitEdge = new THREE.EdgesGeometry(logitGeo);
      const logitWire = new THREE.LineSegments(logitEdge,
        new THREE.LineBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.12 }}));
      logitWire.position.copy(logitMesh.position);
      logitWire.userData.arch = true;
      threeScene.add(logitWire);

      // Logit value label
      const lvC = document.createElement('canvas');
      lvC.width = 128;
      lvC.height = 32;
      const lvx = lvC.getContext('2d');
      lvx.fillStyle = '#e6edf3';
      lvx.font = '12px JetBrains Mono, monospace';
      lvx.textAlign = 'center';
      lvx.fillText(d.logit.toFixed(1), 64, 22);
      const lvt = new THREE.CanvasTexture(lvC);
      const lvsm = new THREE.SpriteMaterial({{ map: lvt, transparent: true }});
      const lvs = new THREE.Sprite(lvsm);
      lvs.position.set(logitGroupX, logitH + 0.35, zPos);
      lvs.scale.set(1.2, 0.3, 1);
      lvs.userData.arch = true;
      threeScene.add(lvs);

      // --- PROBABILITY BAR ---
      const probH = Math.max(0.15, (d.prob / maxProb) * probScale);
      const probGeo = new THREE.BoxGeometry(barW, probH, barD);
      const probMat = new THREE.MeshPhongMaterial({{
        color: 0x39d2c0,
        transparent: true,
        opacity: 0.85,
        shininess: 40,
      }});
      const probMesh = new THREE.Mesh(probGeo, probMat);
      probMesh.position.set(probGroupX, probH / 2, zPos);
      probMesh.userData.arch = true;
      threeScene.add(probMesh);

      // Prob edge
      const probEdge = new THREE.EdgesGeometry(probGeo);
      const probWire = new THREE.LineSegments(probEdge,
        new THREE.LineBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.12 }}));
      probWire.position.copy(probMesh.position);
      probWire.userData.arch = true;
      threeScene.add(probWire);

      // Prob % label
      const pvC = document.createElement('canvas');
      pvC.width = 128;
      pvC.height = 32;
      const pvx = pvC.getContext('2d');
      pvx.fillStyle = '#e6edf3';
      pvx.font = '12px JetBrains Mono, monospace';
      pvx.textAlign = 'center';
      pvx.fillText((d.prob * 100).toFixed(1) + '%', 64, 22);
      const pvt = new THREE.CanvasTexture(pvC);
      const pvsm = new THREE.SpriteMaterial({{ map: pvt, transparent: true }});
      const pvs = new THREE.Sprite(pvsm);
      pvs.position.set(probGroupX, probH + 0.35, zPos);
      pvs.scale.set(1.2, 0.3, 1);
      pvs.userData.arch = true;
      threeScene.add(pvs);

      // Token label (shared, to the right of prob bar)
      const tkC = document.createElement('canvas');
      tkC.width = 256;
      tkC.height = 40;
      const tkx = tkC.getContext('2d');
      tkx.fillStyle = i === 0 ? '#f0e68c' : '#8b949e';
      tkx.font = (i === 0 ? 'bold ' : '') + '14px JetBrains Mono, monospace';
      tkx.textAlign = 'left';
      const tokenStr = (d.word || '').trim() || '(empty)';
      tkx.fillText(tokenStr.slice(0, 12), 10, 26);
      const tkt = new THREE.CanvasTexture(tkC);
      const tksm = new THREE.SpriteMaterial({{ map: tkt, transparent: true }});
      const tks = new THREE.Sprite(tksm);
      tks.position.set(probGroupX + 1.8, probH / 2, zPos);
      tks.scale.set(2.5, 0.5, 1);
      tks.userData.arch = true;
      threeScene.add(tks);

      // Connector: logit bar top → prob bar top
      addLine(
        new THREE.Vector3(logitGroupX + barW / 2 + 0.05, logitH * 0.7, zPos),
        new THREE.Vector3(probGroupX - barW / 2 - 0.05, probH * 0.7, zPos),
        0x484f58
      );
    }}

    // Update cursor for camera calc
    cursor = probGroupX + 4;
  }}

  // ═══ CAMERA SETUP ═══
  const centerX = (embedStartX + softmaxX) / 2;
  const extent = softmaxX - embedStartX;
  threeCamera.position.set(centerX, extent * 0.3, extent * 0.4);
  threeCamera.lookAt(centerX, 0, 0);
  defaultCamPos = threeCamera.position.clone();
  threeControls.target.set(centerX, 0, 0);
  threeControls.saveState();
}}

/* ─── STEP 3 ─── */
function renderStep3() {{
  let h = '<div class="panel">';
  h += '<div class="panel-step"><span class="dot" style="background:#39d2c0;"></span>Step 3</div>';
  h += '<div class="panel-title">What Comes Next?</div>';
  h += '<div class="panel-desc">After thinking through <em>' + CFG.num_layers + ' layers</em>, the AI ranks every word it knows (<em>' + CFG.vocab_size.toLocaleString() + ' words!</em>) by probability.</div>';

  const mx = DATA.predictions[0]?.prob || 1;
  DATA.predictions.forEach((p,i) => {{
    const w = Math.max(3, (p.prob/mx)*100);
    const hue = 160 + i*14;
    h += `<div class="pred-row" style="animation-delay:${{i*0.06}}s">
      <span class="pred-rank">#${{i+1}}</span>
      <span class="pred-word">${{esc(p.word||'(empty)')}}</span>
      <div class="pred-bar-bg"><div class="pred-bar" style="width:${{w}}%;background:hsl(${{hue}},55%,50%);"></div></div>
      <span class="pred-pct">${{(p.prob*100).toFixed(1)}}%</span>
    </div>`;
  }});

  h += '<div style="margin-top:14px;border-top:1px solid #1e2a3a;padding-top:10px;">';
  h += '<div class="energy-label" style="margin-bottom:4px;text-align:left;">Hidden State Energy per Layer</div>';
  h += renderEnergy(DATA.layer_norms);
  h += '<div class="energy-label">Layer 0 → ' + CFG.num_layers + '</div>';
  h += '</div></div>';
  return h;
}}

function renderEnergy(norms) {{
  const w=400,ht=55,pad=4;
  const mn=Math.min(...norms),mx=Math.max(...norms),rng=mx-mn||1;
  const pts = norms.map((v,i) => {{
    const x = pad+(i/(norms.length-1))*(w-2*pad);
    const y = ht-pad-((v-mn)/rng)*(ht-2*pad);
    return x.toFixed(1)+','+y.toFixed(1);
  }}).join(' ');
  const fill = pad+','+(ht-pad)+' '+pts+' '+(w-pad).toFixed(1)+','+(ht-pad);
  const cx = pad+(currentLayer/(norms.length-1))*(w-2*pad);
  const cy = ht-pad-((norms[currentLayer]-mn)/rng)*(ht-2*pad);
  return `<svg class="energy-chart" viewBox="0 0 ${{w}} ${{ht}}" preserveAspectRatio="none">
    <defs><linearGradient id="eg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#f0883e" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="#f0883e" stop-opacity="0.02"/>
    </linearGradient></defs>
    <polygon points="${{fill}}" fill="url(#eg)"/>
    <polyline points="${{pts}}" fill="none" stroke="#f0883e" stroke-width="2" stroke-opacity="0.75" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="${{cx}}" cy="${{cy}}" r="4" fill="#f0883e" stroke="#0d1117" stroke-width="2"/>
  </svg>`;
}}

/* ─── STEP 4 ─── */
function renderStep4() {{
  let h = '<div class="panel">';
  h += '<div class="panel-step"><span class="dot" style="background:#3fb950;"></span>Step 4</div>';
  h += '<div class="panel-title">The AI Speaks</div>';
  h += '<div class="panel-desc">By picking the most likely next word <em>again and again</em>, the AI builds a complete response — one token at a time.</div>';
  h += '<div class="response-box" id="response-box"></div>';
  h += '</div>';
  return h;
}}

/* ─── TYPEWRITER ─── */
function doTypewriter() {{
  if (!DATA) return;
  const box = document.getElementById('response-box');
  if (!box) return;
  const text = DATA.generated_text || '';
  let i = 0;
  box.innerHTML = '<span class="cursor-blink"></span>';
  function tick() {{
    if (i < text.length) {{
      box.innerHTML = esc(text.slice(0,i+1)) + '<span class="cursor-blink"></span>';
      i++;
      setTimeout(tick, 16 + Math.random()*14);
    }} else {{
      box.innerHTML = esc(text) + '<span class="cursor-blink"></span>';
    }}
  }}
  setTimeout(tick, 500);
}}

/* ─── EVENTS ─── */
function attachEvents() {{
  const tip = document.getElementById('tooltip');
  document.querySelectorAll('[data-tip]').forEach(el => {{
    el.addEventListener('mouseenter', e => {{
      tip.textContent = el.dataset.tip;
      tip.style.display = 'block';
      tip.style.left = (e.clientX+14)+'px';
      tip.style.top = (e.clientY-10)+'px';
    }});
    el.addEventListener('mousemove', e => {{
      tip.style.left = (e.clientX+14)+'px';
      tip.style.top = (e.clientY-10)+'px';
    }});
    el.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});
  }});
  doTypewriter();
  initHeadCanvas();
}}

/* ─── SOFTMAX PANEL ─── */
let softmaxOpen = false;
function toggleSoftmax() {{
  softmaxOpen = !softmaxOpen;
  const el = document.getElementById('softmax-panel');
  if (el) el.classList.toggle('show', softmaxOpen);
}}

function renderSoftmaxPanel() {{
  if (!DATA || !DATA.softmax_data) return '';
  const sd = DATA.softmax_data;
  const maxLogit = Math.max(...sd.map(d => Math.abs(d.logit)));
  const maxProb = sd[0]?.prob || 1;

  let h = '<div class="softmax-overlay" id="softmax-panel">';
  h += '<button class="sm-close" onclick="toggleSoftmax()">✕ Close</button>';
  h += '<div class="softmax-title">🔬 Softmax Transformation</div>';
  h += '<div class="softmax-desc">The final layer outputs raw scores called <em>logits</em>. The <em>Softmax</em> function squishes them into probabilities that add up to 100%. Bigger logits get exponentially more probability.</div>';

  h += '<div class="sm-grid">';
  // Headers
  h += '<div class="sm-header">Token</div><div class="sm-header">Raw Logit</div><div class="sm-header">Value</div><div></div><div class="sm-header">Probability</div><div class="sm-header">%</div>';

  sd.forEach((d, i) => {{
    const logitW = Math.max(3, (Math.abs(d.logit) / maxLogit) * 100);
    const probW = Math.max(3, (d.prob / maxProb) * 100);
    const logitColor = d.logit >= 0 ? '#f47067' : '#6e7681';

    h += `<div class="sm-word">${{esc(d.word || '(empty)')}}</div>`;
    h += `<div class="sm-bar-bg"><div class="sm-bar" style="width:${{logitW}}%;background:${{logitColor}};"></div></div>`;
    h += `<div class="sm-val">${{d.logit}}</div>`;
    h += '<div class="sm-arrow">→</div>';
    h += `<div class="sm-bar-bg"><div class="sm-bar" style="width:${{probW}}%;background:#39d2c0;"></div></div>`;
    h += `<div class="sm-val">${{(d.prob * 100).toFixed(1)}}%</div>`;
  }});
  h += '</div></div>';
  return h;
}}

/* ─── HELPERS ─── */
function ac(v) {{
  const t = Math.min(1,Math.max(0,v));
  return `rgb(${{Math.round(13+t*175)}},${{Math.round(15+t*100)}},${{Math.round(30+t*225)}})`;
}}
function esc(s) {{
  const d = document.createElement('div');
  d.appendChild(document.createTextNode(s));
  return d.innerHTML;
}}

init();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .stApp { background-color: #0d1117 !important; }
    header, footer, .stDeployButton, #MainMenu { display: none !important; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    iframe { border: none !important; }
</style>
""", unsafe_allow_html=True)

if current_prompt:
    # Show processing state (architecture with animation) while model runs
    placeholder = st.empty()
    with placeholder.container():
        st.components.v1.html(
            build_html(MODEL_CONFIG, None, "processing"),
            height=1800, scrolling=True
        )

    data = run_model(current_prompt)

    # Replace with complete visualization
    placeholder.empty()
    st.components.v1.html(
        build_html(MODEL_CONFIG, data, "complete"),
        height=1800, scrolling=True
    )
else:
    # Show idle state: full architecture visible with empty panels
    st.components.v1.html(
        build_html(MODEL_CONFIG, None, "idle"),
        height=1800, scrolling=True
    )