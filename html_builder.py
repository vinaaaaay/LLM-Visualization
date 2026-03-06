"""
html_builder.py — HTML/CSS/JS template builder for the LLM visualizer.

Generates the self-contained HTML page with embedded Three.js 3D visualization,
architecture strip, step panels, and all interactive components.
"""

import json


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

/* Dots block for hidden layers */
.dots-block {{
  flex-shrink: 0;
  display: flex; align-items: center; gap: 6px;
  padding: 8px 14px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px; font-weight: 600;
  color: #6e7681; letter-spacing: 3px;
  border: 1px dashed #2d3a4a; border-radius: 6px;
  background: rgba(30, 42, 58, 0.3);
  animation: dotsPulse 2.5s ease-in-out infinite;
  cursor: default;
  white-space: nowrap;
}}
.dots-block .dots-count {{
  font-size: 10px; color: #484f58;
  letter-spacing: 1px; font-weight: 400;
}}
@keyframes dotsPulse {{
  0%, 100% {{ opacity: 0.6; }}
  50% {{ opacity: 1; }}
}}

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

  if (DATA) currentLayer = CFG.num_layers - 1;  // default to last (visible) layer
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
      const currentIndex = idx; // Capture idx in the closure for the timeout callback
      groups[currentIndex].classList.add('wave');
      // Also pulse the blocks inside
      groups[currentIndex].querySelectorAll('.a-block').forEach(b => b.classList.add('processing'));
      setTimeout(() => {{
        if (groups[currentIndex]) {{
          groups[currentIndex].querySelectorAll('.a-block').forEach(b => b.classList.remove('processing'));
        }}
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

  const SHOW_THRESHOLD = 5;
  const SHOW_START = 3;  // first N layers to show
  const SHOW_END = 2;    // last N layers to show
  const shouldCollapse = CFG.num_layers > SHOW_THRESHOLD;
  const hiddenCount = shouldCollapse ? CFG.num_layers - SHOW_START - SHOW_END : 0;

  function renderLayerBlock(i) {{
    const act = DATA ? (i === currentLayer) : false;
    let s = `<div class="layer-grp ${{act?'active':''}}" ${{DATA ? 'onclick="selectLayer('+i+')"' : ''}}>`;
    s += `<div class="ltag">L${{i}}</div>`;
    s += ab('ln','LN','',act);
    s += ai();
    s += ab('attn','Attn',CFG.num_heads+'h',act);
    s += ai();
    s += ab('ln','LN','',act);
    s += ai();
    s += ab('mlp','MLP','',act);
    s += '</div>';
    return s;
  }}

  if (shouldCollapse) {{
    // First SHOW_START layers
    for (let i = 0; i < SHOW_START; i++) {{
      h += renderLayerBlock(i);
      h += aw();
    }}
    // Dots
    h += `<div class="dots-block">⋯ <span class="dots-count">${{hiddenCount}} hidden layers</span> ⋯</div>`;
    h += aw();
    // Last SHOW_END layers
    for (let i = CFG.num_layers - SHOW_END; i < CFG.num_layers; i++) {{
      h += renderLayerBlock(i);
      if (i < CFG.num_layers - 1) h += aw();
    }}
  }} else {{
    for (let i = 0; i < CFG.num_layers; i++) {{
      h += renderLayerBlock(i);
      if (i < CFG.num_layers - 1) h += aw();
    }}
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
  const layerSpacing = 16;      // distance between layers along X
  const blockH = 1.0;           // height of LN/MLP blocks
  const blockW = 1.5;           // width
  const blockD = 1.0;           // depth
  const headPlaneSize = 1.2;    // size of each attention head plane
  const headDepthSpacing = headPlaneSize + 0.15; // Z spacing — tile side by side
  const subSpacing = 5.5;       // spacing within a layer (between LN, Attn, MLP)

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

  // Helper: connect two points (thick, solid lines)
  function addLine(p1, p2, color) {{
    const geo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
    const mat = new THREE.LineBasicMaterial({{ color: color || 0x3fb950, transparent: true, opacity: 0.75, linewidth: 2 }});
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
    
    const colTokX = embedStartX - 2;
    const colIdxX = colTokX + 3.5;
    const colMatrixX = colIdxX + 4;
    const colEmbX = colMatrixX + 4;
    const embedFnX = colEmbX + 7; 
    embedOutEdgeX = embedFnX + blockW * 0.75;

    addTextSprite(colTokX, 2, 0, 'Token', '#e6edf3', 22);
    addTextSprite(colIdxX, 2, 0, 'ID', '#d2a8ff', 22);
    addTextSprite(colEmbX, 3.5, 0, 'Token Embedding', '#58a6ff', 22);
    addTextSprite(colMatrixX, 3.5, 0, 'Embedding Matrix', '#8b949e', 22);
    addTextSprite(colMatrixX, 2.5, 0, 'W_e', '#8b949e', 18, 1.5, 0.4);

    // Explanatory labels at the bottom
    // We create a custom high-res canvas sprite to prevent blurriness on long text
    const baseLineY = -2.5; // Shared baseline under all matrix/token items
    
    function addDescLabel(x, text, yDrop) {{
      const dCanvas = document.createElement('canvas');
      const dCtx = dCanvas.getContext('2d');
      const dFs = 48; // Huge font size for sharp resolution
      dCtx.font = 'bold ' + dFs + 'px JetBrains Mono, monospace';
      
      const metrics = dCtx.measureText(text);
      dCanvas.width = metrics.width + 20;
      dCanvas.height = dFs + 20;
      
      // Re-fetch context after resizing
      const ctx = dCanvas.getContext('2d');
      ctx.font = 'bold ' + dFs + 'px JetBrains Mono, monospace';
      ctx.fillStyle = '#e6edf3'; // Bright, highly readable off-white
      ctx.textAlign = 'center';
      ctx.fillText(text, dCanvas.width / 2, dFs);
      
      const dTex = new THREE.CanvasTexture(dCanvas);
      const dMat = new THREE.SpriteMaterial({{ map: dTex, transparent: true }});
      const dSpr = new THREE.Sprite(dMat);
      
      // Scale down the giant sprite so it fits in the 3D world but remains sharp
      const sprH = 0.5; 
      dSpr.scale.set(sprH * dCanvas.width / dCanvas.height, sprH, 1);
      
      const finalY = baseLineY - yDrop;
      dSpr.position.set(x, finalY, 0);
      dSpr.userData.arch = true;
      threeScene.add(dSpr);

      // Draw pointer line linking the staggered text UP to the column baseline
      addLine(new THREE.Vector3(x, finalY + sprH/2, 0), new THREE.Vector3(x, baseLineY + 0.5, 0), 0xffffff);
    }}

    // Stagger dropping them down to prevent overlap
    addDescLabel(colTokX, '1. Split text into tokens', 0.0);
    addDescLabel(colIdxX, '2. Map to vocabulary IDs', 1.0);
    addDescLabel(colMatrixX, '3. Lookup vector by ID', 2.0);
    addDescLabel(colEmbX, '4. Extract Word Vectors', 3.0);

    // Draw the Embedding Matrix as a cool glowing tech-grid
    const matZSize = totalZ + 2.0;

    const mCanvas = document.createElement('canvas');
    const mRows = DATA.embed_sample ? DATA.embed_sample.length : 20;
    const mCols = DATA.embed_sample ? DATA.embed_sample[0].length : 20;
    const cellSize = 16;
    mCanvas.width = mCols * cellSize;
    mCanvas.height = mRows * cellSize;
    const mCtx = mCanvas.getContext('2d');
    
    // Dark background for matrix
    mCtx.fillStyle = '#0d1117';
    mCtx.fillRect(0, 0, mCanvas.width, mCanvas.height);
    
    for (let r = 0; r < mRows; r++) {{
      for (let c = 0; c < mCols; c++) {{
        let val = DATA.embed_sample ? DATA.embed_sample[r][c] : (Math.random() - 0.5);
        // Cool neon colors based on weight values
        const alpha = Math.min(1.0, Math.abs(val) * 5.0 + 0.1);
        if (val > 0) {{
          mCtx.fillStyle = `rgba(57, 210, 192, ${{alpha}})`; // neon teal
        }} else {{
          mCtx.fillStyle = `rgba(188, 140, 255, ${{alpha}})`; // neon purple
        }}
        mCtx.fillRect(c * cellSize + 1, r * cellSize + 1, cellSize - 2, cellSize - 2);
      }}
    }}
    
    const matTex = new THREE.CanvasTexture(mCanvas);
    matTex.magFilter = THREE.NearestFilter; // keep it sharp and blocky

    // Give it slight thickness instead of flat plane
    const matGeometry = new THREE.BoxGeometry(0.6, 4.0, matZSize);
    const matMaterial = new THREE.MeshBasicMaterial({{
      map: matTex,
      transparent: true,
      opacity: 0.9,
    }});
    const matBox = new THREE.Mesh(matGeometry, matMaterial);
    matBox.position.set(colMatrixX, 0, 0);
    matBox.userData.arch = true;
    threeScene.add(matBox);
    
    // Add glowing wireframe edges
    const edges = new THREE.EdgesGeometry(matGeometry);
    const lineMat = new THREE.LineBasicMaterial({{ color: 0x58a6ff, opacity: 0.8, transparent: true }});
    const matEdges = new THREE.LineSegments(edges, lineMat);
    matBox.add(matEdges);

    for (let i = 0; i < numTokens; i++) {{
      const zPos = -totalZ / 2 + i * zStep;
      
      const tokStr = ('"' + DATA.tokens[i] + '"').slice(0, 12);
      addTextSprite(colTokX, 0, zPos, tokStr, '#e6edf3', 18);
      
      const idStr = DATA.token_ids[i].toString();
      addTextSprite(colIdxX, 0, zPos, idStr, '#d2a8ff', 16);
      
      // Line from Token -> ID
      addLine(new THREE.Vector3(colTokX+1.5, 0, zPos), new THREE.Vector3(colIdxX-1.0, 0, zPos), 0x484f58);

      // Line from ID -> Matrix
      addLine(new THREE.Vector3(colIdxX+1.0, 0, zPos), new THREE.Vector3(colMatrixX-0.1, 0, zPos), 0x484f58);

      // Line from Matrix -> Word Vector
      addLine(new THREE.Vector3(colMatrixX+0.1, 0, zPos), new THREE.Vector3(colEmbX-2.0, 0, zPos), 0x58a6ff);

      // Embedding vector sprite (actual values)
      if (DATA.token_embeddings && DATA.token_embeddings[i]) {{
        const w = DATA.token_embeddings[i];
        const lines = [
          (w[0] >= 0 ? '+' : '') + w[0].toFixed(3),
          (w[1] >= 0 ? '+' : '') + w[1].toFixed(3),
          (w[2] >= 0 ? '+' : '') + w[2].toFixed(3),
          ' \u22ef',
          (w[3] >= 0 ? '+' : '') + w[3].toFixed(3),
          (w[4] >= 0 ? '+' : '') + w[4].toFixed(3)
        ];
        const vc = document.createElement('canvas');
        const fs = 14;
        const lh = fs + 4;
        const cW = 120, cH = lines.length * lh + 8;
        vc.width = cW; vc.height = cH;
        const vx = vc.getContext('2d');
        vx.font = 'bold ' + fs + 'px JetBrains Mono, monospace';
        
        // Brackets
        vx.strokeStyle = '#58a6ff'; vx.lineWidth = 1.5;
        vx.beginPath(); vx.moveTo(12, 4); vx.lineTo(6, 4); vx.lineTo(6, cH-4); vx.lineTo(12, cH-4); vx.stroke();
        vx.beginPath(); vx.moveTo(cW-12, 4); vx.lineTo(cW-6, 4); vx.lineTo(cW-6, cH-4); vx.lineTo(cW-12, cH-4); vx.stroke();
        
        for (let li = 0; li < lines.length; li++) {{
          const val = parseFloat(lines[li]);
          vx.fillStyle = isNaN(val) ? '#6e7681' : (val >= 0 ? '#58a6ff' : '#f85149');
          vx.textAlign = 'center';
          vx.fillText(lines[li], cW/2, (li+1) * lh);
        }}
        const vTex = new THREE.CanvasTexture(vc);
        const vMat = new THREE.SpriteMaterial({{ map: vTex, transparent: true }});
        const vSpr = new THREE.Sprite(vMat);
        const sprH = 1.0;
        vSpr.scale.set(sprH * cW / cH, sprH, 1);
        vSpr.position.set(colEmbX, 0, zPos);
        vSpr.userData.arch = true;
        threeScene.add(vSpr);

      }} else {{
         // Fallback box if data is missing
         addBox(colEmbX, 0, zPos, 2.5, 0.3, 0.8, 0x58a6ff);
      }}

      // We no longer draw individual lines from tokens to the Embedding Output matrix
      // because the new Matrix visualization is a single unified block representing the whole sequence.
    }}

    // Draw one single centered connecting line from the middle of the Token Embeddings stack to the Embed Output Matrix
    addLine(new THREE.Vector3(colEmbX + 1.2, 0, 0), new THREE.Vector3(embedFnX - 2.5, 0, 0), 0x58a6ff);
    
    // ═══ Sequence Embedding Output Matrix ═══
    // Instead of a single box, show a stacked block representing (SeqLen, HiddenSize)
    const eoRows = numTokens;
    // We'll show first 3 and last 2 columns
    const numColsToShow = DATA.token_embeddings && DATA.token_embeddings[0] ? DATA.token_embeddings[0].length : 5;
    
    const oCanvas = document.createElement('canvas');
    const fs = 14;
    const lh = fs + 6;
    const colW = 65;
    
    const cW = numColsToShow * colW + 40; 
    const cH = eoRows * lh + 16;
    oCanvas.width = cW; oCanvas.height = cH;
    const oCtx = oCanvas.getContext('2d');
    
    oCtx.font = 'bold ' + fs + 'px JetBrains Mono, monospace';
    
    // Brackets
    oCtx.strokeStyle = '#58a6ff'; oCtx.lineWidth = 2;
    oCtx.beginPath(); oCtx.moveTo(16, 4); oCtx.lineTo(8, 4); oCtx.lineTo(8, cH-4); oCtx.lineTo(16, cH-4); oCtx.stroke();
    oCtx.beginPath(); oCtx.moveTo(cW-16, 4); oCtx.lineTo(cW-8, 4); oCtx.lineTo(cW-8, cH-4); oCtx.lineTo(cW-16, cH-4); oCtx.stroke();
    
    for (let r = 0; r < eoRows; r++) {{
      const y = (r + 1) * lh;
      for (let c = 0; c < numColsToShow; c++) {{
        let x = 20 + c * colW + colW/2;
        if (c === 3) {{
          oCtx.fillStyle = '#6e7681';
          oCtx.textAlign = 'center';
          oCtx.fillText('\u22ef', x, y);
          continue;
        }}
        
        let val = (DATA.token_embeddings && DATA.token_embeddings[r]) ? DATA.token_embeddings[r][c] : 0;
        oCtx.fillStyle = val >= 0 ? '#39d2c0' : '#f85149';
        oCtx.textAlign = 'center';
        const strVal = (val >= 0 ? '+' : '') + val.toFixed(2);
        oCtx.fillText(strVal, x, y);
      }}
    }}
    
    const oTex = new THREE.CanvasTexture(oCanvas);
    const oMat = new THREE.SpriteMaterial({{ map: oTex, transparent: true }});
    const oSpr = new THREE.Sprite(oMat);
    const sprH = 2.0;
    oSpr.scale.set(sprH * cW / cH, sprH, 1);
    oSpr.position.set(embedFnX, 0, 0);
    oSpr.userData.arch = true;
    threeScene.add(oSpr);
    
    addTextSprite(embedFnX, 3.5, 0, 'Embed Output Matrix', '#58a6ff', 20);
    // Made the dimension explicit and clearly visible right above the matrix
    addTextSprite(embedFnX, 1.8, 0, `[${{numTokens}} \u00D7 ${{CFG.hidden_size}}]`, '#8b949e', 18, 3.0, 0.6);
    addDescLabel(embedFnX, '5. Stack vectors into sequence matrix', 4.0);

    cursor = embedFnX + layerSpacing;
  }} else {{
    // Fallback if no detailed data
    addBox(embedStartX, 0, 0, blockW * 1.2, blockH * 1.5, blockD * 1.2, 0x58a6ff, 'Embed');
    cursor += layerSpacing;
  }}

  // ═══ TRANSFORMER LAYERS (collapsed: show first 3 + last 2) ═══
  const SHOW_THRESHOLD_3D = 5;
  const SHOW_START_3D = 3;
  const SHOW_END_3D = 2;
  const shouldCollapse3D = numLayers > SHOW_THRESHOLD_3D;
  const hiddenCount3D = shouldCollapse3D ? numLayers - SHOW_START_3D - SHOW_END_3D : 0;

  // Determine which layers to render
  const layersToRender = [];
  if (shouldCollapse3D) {{
    for (let i = 0; i < SHOW_START_3D; i++) layersToRender.push(i);
    layersToRender.push(-1);  // sentinel for dots
    for (let i = numLayers - SHOW_END_3D; i < numLayers; i++) layersToRender.push(i);
  }} else {{
    for (let i = 0; i < numLayers; i++) layersToRender.push(i);
  }}

  let prevMlpXForConnect = null;
  let isFirstRendered = true;

  for (let ri = 0; ri < layersToRender.length; ri++) {{
    const li = layersToRender[ri];

    // ─── DOTS placeholder ───
    if (li === -1) {{
      const dotsX = cursor + layerSpacing * 0.5;
      // Three dots with label
      for (let di = 0; di < 3; di++) {{
        const dotGeo = new THREE.SphereGeometry(0.25, 16, 16);
        const dotMat = new THREE.MeshPhongMaterial({{ color: 0x6e7681, transparent: true, opacity: 0.7 }});
        const dotMesh = new THREE.Mesh(dotGeo, dotMat);
        dotMesh.position.set(dotsX + di * 1.2, 0, 0);
        dotMesh.userData.arch = true;
        threeScene.add(dotMesh);
      }}
      // Label
      addTextSprite(dotsX + 1.2, 1.8, 0, hiddenCount3D + ' hidden layers', '#6e7681', 16, 4.0, 0.6);

      // Connect previous MLP to dots
      if (prevMlpXForConnect !== null) {{
        addLine(
          new THREE.Vector3(prevMlpXForConnect + blockW / 2 + 0.1, 0, 0),
          new THREE.Vector3(dotsX - 0.5, 0, 0),
          0x3fb950
        );
      }}
      // Set cursor past the dots gap
      cursor = dotsX + 3 * 1.2 + layerSpacing * 0.5;
      // Mark that next layer should connect from dots
      prevMlpXForConnect = dotsX + 3 * 1.2 - 0.5;
      isFirstRendered = false;
      continue;
    }}

    const layerX = cursor;
    const details = DATA.layer_details ? DATA.layer_details[li] : null;
    const qkvLayerData = DATA.qkv_data ? DATA.qkv_data[li] : null;
    const headDim = CFG.head_dim || 64;

    // ═══ HEATMAP HELPER ═══
    function drawHeatmap(data, cx, cy, cz, planeW, planeH, borderColor) {{
      if (!data || data.length === 0) return;
      const rows = data.length;
      const cols = data[0].length;
      const cellW = 4, cellH = 4;
      const cW = cols * cellW, cH = rows * cellH;
      const hc = document.createElement('canvas');
      hc.width = cW; hc.height = cH;
      const hx = hc.getContext('2d');
      // Find min/max for normalization
      let mn = Infinity, mx = -Infinity;
      for (let r = 0; r < rows; r++) {{
        for (let c = 0; c < cols; c++) {{
          const v = data[r][c];
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }}
      }}
      const rng = mx - mn || 1;
      // Viridis-inspired colormap
      function viridis(t) {{
        // Simplified viridis: dark purple → teal → yellow
        const r = Math.max(0, Math.min(255, Math.floor(68 + t * (253 - 68))));
        const g = Math.max(0, Math.min(255, Math.floor(1 + t * (231 - 1))));
        const b = Math.max(0, Math.min(255, Math.floor(84 + (t < 0.5 ? t * 2 * (170-84) : (1-t)*2*170))));
        return `rgb(${{r}},${{g}},${{b}})`;
      }}
      for (let r = 0; r < rows; r++) {{
        for (let c = 0; c < cols; c++) {{
          const t = (data[r][c] - mn) / rng;
          hx.fillStyle = viridis(t);
          hx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }}
      }}
      const mexTex = new THREE.CanvasTexture(hc);
      mexTex.magFilter = THREE.NearestFilter;
      const geo = new THREE.PlaneGeometry(planeW, planeH);
      const mat = new THREE.MeshBasicMaterial({{ map: mexTex, side: THREE.DoubleSide }});
      const mesh = new THREE.Mesh(geo, mat);
      mesh.rotation.x = -Math.PI / 2; // [NEW] Lay flat on the floor
      mesh.position.set(cx, cy, cz);
      mesh.userData.arch = true;
      threeScene.add(mesh);
      // Border wireframe
      const edgeGeo = new THREE.EdgesGeometry(geo);
      const edgeMat = new THREE.LineBasicMaterial({{ color: borderColor || 0x484f58, transparent: true, opacity: 0.6 }});
      const edges = new THREE.LineSegments(edgeGeo, edgeMat);
      edges.rotation.x = -Math.PI / 2; // [NEW] Lay flat on the floor
      edges.position.copy(mesh.position);
      edges.userData.arch = true;
      threeScene.add(edges);
      return mesh;
    }}

    // ═══ ATTENTION HEAD DIAGRAM LAYOUT (FLAT 2D REPLICA) ═══
    // We construct this flat on the X-Z floor (y=0).
    // The camera looks from +Z, so -Z is "UP" on the screen and +Z is "DOWN".
    const rowZ = 5.0;          
    const qRowZ = -rowZ;       // Q row Z (Top of diagram)
    const kRowZ = 0;           // K row Z (Middle)
    const vRowZ = rowZ;        // V row Z (Bottom)
    const heatW = 4.5;         
    const heatH = 2.2;         
    const wCircleR = 0.6;      
    const stepX = 4.5;         

    // ═══ STAGE 1: Weight Matrix Circles (W_Q, W_K, W_V) ═══
    const wX = layerX;
    function addWeightCircle(x, z, label, dimLabel, color) {{
      const cGeo = new THREE.CylinderGeometry(wCircleR, wCircleR, 0.15, 24);
      const cMat = new THREE.MeshPhongMaterial({{ color: color, transparent: true, opacity: 0.85, emissive: color, emissiveIntensity: 0.15 }});
      const cMesh = new THREE.Mesh(cGeo, cMat);
      // rotation.x = 0 means the cylinder's flat face points UP (Y). Perfect for a floor coin.
      cMesh.position.set(x, 0, z);
      cMesh.userData.arch = true;
      threeScene.add(cMesh);
      addTextSprite(x, 0.5, z - 0.1, label, '#e6edf3', 24, 2.5, 0.8);
      addTextSprite(x, 0.0, z + wCircleR + 0.6, dimLabel, '#8b949e', 18, 3.5, 0.6); // Below circle in Z
    }}
    addWeightCircle(wX, qRowZ, 'W_Q', CFG.hidden_size + '\\u00D7' + headDim, 0xf47067);
    addWeightCircle(wX, kRowZ, 'W_K', CFG.hidden_size + '\\u00D7' + headDim, 0x39d2c0);
    addWeightCircle(wX, vRowZ, 'W_V', CFG.hidden_size + '\\u00D7' + headDim, 0xbc8cff);

    // X Split to Q/K/V Circles
    addLine(new THREE.Vector3(wX - 2.0, 0, kRowZ), new THREE.Vector3(wX - 2.0, 0, qRowZ), 0x8b949e);
    addLine(new THREE.Vector3(wX - 2.0, 0, kRowZ), new THREE.Vector3(wX - 2.0, 0, vRowZ), 0x8b949e);
    addLine(new THREE.Vector3(wX - 2.0, 0, qRowZ), new THREE.Vector3(wX - wCircleR - 0.1, 0, qRowZ), 0x8b949e);
    addLine(new THREE.Vector3(wX - 2.0, 0, kRowZ), new THREE.Vector3(wX - wCircleR - 0.1, 0, kRowZ), 0x8b949e);
    addLine(new THREE.Vector3(wX - 2.0, 0, vRowZ), new THREE.Vector3(wX - wCircleR - 0.1, 0, vRowZ), 0x8b949e);

    const qkvHeatX = wX + stepX + heatW / 2;
    const ktX = qkvHeatX + heatW/2 + stepX - 1.0;
    const qktX = ktX + heatH/2 + stepX;
    const qktCenterZ = (qRowZ + kRowZ) / 2;
    const qktHeatSize = heatH * 1.5; 
    const maskX = qktX;
    const maskZ = qktCenterZ + qktHeatSize/2 + 1.2;
    const attnPatX = qktX + qktHeatSize/2 + stepX + 1.5;
    const attnPatSize = 4.0;
    const attnPatZ = qktCenterZ;
    const avOpX = attnPatX + attnPatSize/2 + stepX - 1.0;
    const avOpZ = vRowZ;
    const outX = avOpX + stepX + heatW/2;

    const numHeads = qkvLayerData && qkvLayerData.q ? qkvLayerData.q.length : 1;
    const seqLen = qkvLayerData && qkvLayerData.q_shape ? qkvLayerData.q_shape[0] : (DATA.tokens ? DATA.tokens.length : 0);
    const yLayerGap = 3.5; // Gap between each head on the Y axis

    // Fan connectors from W circles to the different heads in Y direction
    for (let hi = 0; hi < numHeads; hi++) {{
      const yOff = hi * yLayerGap;
      addLine(new THREE.Vector3(wX + wCircleR + 0.1, 0, qRowZ), new THREE.Vector3(qkvHeatX - heatW/2 - 0.1, yOff, qRowZ), 0xf47067);
      addLine(new THREE.Vector3(wX + wCircleR + 0.1, 0, kRowZ), new THREE.Vector3(qkvHeatX - heatW/2 - 0.1, yOff, kRowZ), 0x39d2c0);
      addLine(new THREE.Vector3(wX + wCircleR + 0.1, 0, vRowZ), new THREE.Vector3(qkvHeatX - heatW/2 - 0.1, yOff, vRowZ), 0xbc8cff);
    }}

    // ═══ STAGE 2 - 8: Multi-Head Loop ═══
    // We draw from back to front (hi=numHeads-1 down to 0) so text rendering depth is proper if not using z-buffer.
    for (let hi = numHeads - 1; hi >= 0; hi--) {{
      const yOff = hi * yLayerGap;

      // ═══ STAGE 2: Q, K, V Heatmaps ═══
      if (qkvLayerData && qkvLayerData.q && qkvLayerData.q.length > hi) {{
        drawHeatmap(qkvLayerData.q[hi], qkvHeatX, yOff, qRowZ, heatW, heatH, 0xf47067);
        drawHeatmap(qkvLayerData.k[hi], qkvHeatX, yOff, kRowZ, heatW, heatH, 0x39d2c0);
        drawHeatmap(qkvLayerData.v[hi], qkvHeatX, yOff, vRowZ, heatW, heatH, 0xbc8cff);
      }} else {{
        addBox(qkvHeatX, yOff, qRowZ, heatW, 0.1, heatH, 0xf47067, 'Q');
        addBox(qkvHeatX, yOff, kRowZ, heatW, 0.1, heatH, 0x39d2c0, 'K');
        addBox(qkvHeatX, yOff, vRowZ, heatW, 0.1, heatH, 0xbc8cff, 'V');
      }}

      // ═══ STAGE 3: K^T (Transposed K) ═══
      if (qkvLayerData && qkvLayerData.k && qkvLayerData.k.length > hi) {{
        const kData = qkvLayerData.k[hi];
        const kRows = kData.length;
        const kCols = kData[0].length;
        const kT = [];
        for (let c = 0; c < kCols; c++) {{
          const row = [];
          for (let r = 0; r < kRows; r++) {{ row.push(kData[r][c]); }}
          kT.push(row);
        }}
        drawHeatmap(kT, ktX, yOff, kRowZ, heatH, heatW * 0.5, 0x39d2c0);
      }} else {{
        addBox(ktX, yOff, kRowZ, heatH, 0.1, heatW * 0.5, 0x39d2c0, 'K^T');
      }}

      // Connector: K heatmap → K^T
      addLine(new THREE.Vector3(qkvHeatX + heatW/2 + 0.1, yOff, kRowZ), new THREE.Vector3(ktX - heatH/2 - 0.1, yOff, kRowZ), 0x39d2c0);

      // ═══ STAGE 4: QK^T matrix (seq×seq) ═══
      const attnData = DATA.attention[li] && DATA.attention[li].length > hi ? DATA.attention[li][hi] : null;
      if (attnData && attnData.length > 0) {{
        drawHeatmap(attnData, qktX, yOff, qktCenterZ, qktHeatSize, qktHeatSize, 0xd2a8ff);
      }} else {{
        addBox(qktX, yOff, qktCenterZ, qktHeatSize, 0.1, qktHeatSize, 0xd2a8ff, 'QK^T');
      }}

      // Connector: Q heatmap → QK^T (Down/Right)
      addLine(new THREE.Vector3(qkvHeatX + heatW/2 + 0.1, yOff, qRowZ), new THREE.Vector3(qktX, yOff, qRowZ), 0xf47067);
      addLine(new THREE.Vector3(qktX, yOff, qRowZ), new THREE.Vector3(qktX, yOff, qktCenterZ - qktHeatSize/2 - 0.1), 0xf47067);
      
      // Connector: K^T → QK^T (Up/Right)
      addLine(new THREE.Vector3(ktX + heatH/2 + 0.1, yOff, kRowZ), new THREE.Vector3(ktX + heatH/2 + 0.8, yOff, kRowZ), 0x39d2c0);
      addLine(new THREE.Vector3(ktX + heatH/2 + 0.8, yOff, kRowZ), new THREE.Vector3(ktX + heatH/2 + 0.8, yOff, qktCenterZ), 0x39d2c0);
      addLine(new THREE.Vector3(ktX + heatH/2 + 0.8, yOff, qktCenterZ), new THREE.Vector3(qktX - qktHeatSize/2 - 0.1, yOff, qktCenterZ), 0x39d2c0);

      // ═══ STAGE 5: Mask & Softmax box ═══
      addBox(maskX, yOff, maskZ, 2.0, 0.1, 1.2, 0xf7c948, hi === 0 ? 'Mask & Softmax' : '');
      // Connector: QK^T → Mask&Softmax
      addLine(new THREE.Vector3(qktX, yOff, qktCenterZ + qktHeatSize/2 + 0.1), new THREE.Vector3(maskX, yOff, maskZ - 0.6), 0xd2a8ff);

      // ═══ STAGE 6: Attention Pattern (post-softmax) ═══
      if (attnData && attnData.length > 0) {{
        drawHeatmap(attnData, attnPatX, yOff, attnPatZ, attnPatSize, attnPatSize, 0xd2a8ff);
      }}

      // Connector: Mask&Softmax → Attention Pattern
      addLine(new THREE.Vector3(maskX + 1.0 + 0.1, yOff, maskZ), new THREE.Vector3(attnPatX, yOff, maskZ), 0xf7c948);
      addLine(new THREE.Vector3(attnPatX, yOff, maskZ), new THREE.Vector3(attnPatX, yOff, attnPatZ + attnPatSize/2 + 1.2), 0xf7c948);

      // ═══ STAGE 7: ⊗ AV operation ═══
      addBox(avOpX, yOff, avOpZ, 1.0, 0.1, 1.0, 0xe6edf3, '\\u2297');
      // Connector: Attention Pattern → ⊗ AV
      addLine(new THREE.Vector3(attnPatX + attnPatSize/2, yOff, attnPatZ + attnPatSize/2), new THREE.Vector3(avOpX, yOff, attnPatZ + attnPatSize/2), 0xd2a8ff);
      addLine(new THREE.Vector3(avOpX, yOff, attnPatZ + attnPatSize/2), new THREE.Vector3(avOpX, yOff, avOpZ - 0.5), 0xd2a8ff);
      // Connector: V heatmap → ⊗ AV
      addLine(new THREE.Vector3(qkvHeatX + heatW/2 + 0.1, yOff, vRowZ), new THREE.Vector3(avOpX - 0.5, yOff, vRowZ), 0xbc8cff);

      // ═══ STAGE 8: Output O = AV heatmap ═══
      if (qkvLayerData && qkvLayerData.output && qkvLayerData.output.length > hi) {{
        drawHeatmap(qkvLayerData.output[hi], outX, yOff, vRowZ, heatW, heatH, 0x58a6ff);
      }} else {{
        addBox(outX, yOff, vRowZ, heatW, 0.1, heatH, 0x58a6ff, 'O = AV');
      }}
      // Connector: ⊗AV → O heatmap
      addLine(new THREE.Vector3(avOpX + 0.5, yOff, avOpZ), new THREE.Vector3(outX - heatW/2 - 0.1, yOff, vRowZ), 0xe6edf3);
      
      // We only draw the text labels and axis labels for the front-most head (hi === 0)
      if (hi === 0) {{
        addTextSprite(qkvHeatX - heatW/4, 0, qRowZ + heatH/2 + 0.6, 'QUERIES   Q = XW_Q', '#f47067', 20, 7.0, 0.6);
        addTextSprite(qkvHeatX + heatW/2 - 0.5, 0, qRowZ + heatH/2 + 0.6, seqLen + '\\u00D7' + headDim, '#8b949e', 16, 2.5, 0.5);
        addTextSprite(qkvHeatX - heatW/4, 0, kRowZ + heatH/2 + 0.6, 'KEYS   K = XW_K', '#39d2c0', 20, 7.0, 0.6);
        addTextSprite(qkvHeatX + heatW/2 - 0.5, 0, kRowZ + heatH/2 + 0.6, seqLen + '\\u00D7' + headDim, '#8b949e', 16, 2.5, 0.5);
        addTextSprite(qkvHeatX - heatW/4, 0, vRowZ + heatH/2 + 0.6, 'VALUES   V = XW_V', '#bc8cff', 20, 7.0, 0.6);
        addTextSprite(qkvHeatX + heatW/2 - 0.5, 0, vRowZ + heatH/2 + 0.6, seqLen + '\\u00D7' + headDim, '#8b949e', 16, 2.5, 0.5);

        addTextSprite(ktX, 0.0, kRowZ - heatW*0.25 - 0.6, 'K\\u1d40', '#39d2c0', 24, 2.0, 0.7);
        addTextSprite(ktX, 0.0, kRowZ + heatW*0.25 + 0.6, 'Transpose', '#8b949e', 16, 3.5, 0.5);

        addTextSprite(qktX, 0, qktCenterZ - qktHeatSize/2 - 0.6, 'QK\\u1d40', '#d2a8ff', 24, 2.5, 0.7);
        addTextSprite(qktX + qktHeatSize/2 + 0.2, 0, qktCenterZ + qktHeatSize/2 + 0.6, seqLen + '\\u00D7' + seqLen, '#8b949e', 16, 2.5, 0.5);

        addTextSprite(attnPatX + attnPatSize/2 - 0.8, 0.1, attnPatZ - attnPatSize/2 + 0.6, 'ATTENTION', '#d2a8ff', 16, 4.0, 0.4);
        addTextSprite(attnPatX + attnPatSize/2 - 0.8, 0.1, attnPatZ - attnPatSize/2 + 1.1, 'PATTERN', '#d2a8ff', 16, 4.0, 0.4);
        addTextSprite(attnPatX + attnPatSize/2 - 0.5, 0.1, attnPatZ - attnPatSize/2 + 1.6, seqLen + '\\u00D7' + seqLen, '#8b949e', 14, 2.5, 0.4);

        if (DATA.tokens) {{
          const numTkn = Math.min(DATA.tokens.length, seqLen || DATA.tokens.length);
          for (let ti = 0; ti < numTkn; ti++) {{
            const frac = numTkn > 1 ? ti / (numTkn - 1) : 0.5;
            const tZ = attnPatZ - attnPatSize/2 + frac * attnPatSize;
            addTextSprite(attnPatX - attnPatSize/2 - 0.8, 0.0, tZ, DATA.tokens[ti].trim() || '?', '#e6edf3', 14, 2.0, 0.4);
            const tX = attnPatX - attnPatSize/2 + frac * attnPatSize;
            addTextSprite(tX, 0.0, attnPatZ + attnPatSize/2 + 0.6, DATA.tokens[ti].trim() || '?', '#e6edf3', 14, 2.0, 0.4);
          }}
        }}

        addTextSprite(attnPatX, 0, attnPatZ - attnPatSize/2 - 1.5, 'A = Softmax(QK\\u1d40/\\u221Ad)', '#e6edf3', 28, 10.0, 0.8);
        addTextSprite(avOpX, 0, avOpZ + 1.2, 'AV', '#e6edf3', 24, 2.5, 0.7);

        addTextSprite(outX - heatW/4, 0, vRowZ + heatH/2 + 0.6, 'ATTENTION HEAD OUT', '#58a6ff', 18, 7.5, 0.6); 
        addTextSprite(outX + heatW/4, 0, vRowZ + heatH/2 + 0.6, 'O = AV', '#e6edf3', 24, 3.5, 0.7);            
        addTextSprite(outX + heatW/2 - 0.5, 0, vRowZ + heatH/2 + 1.2, seqLen + '\\u00D7' + headDim, '#8b949e', 16, 2.5, 0.5); 
        
        addTextSprite(wX + 15, 0, qRowZ - 3.0, 'ATTENTION HEAD (Layer ' + li + ')', '#6e7681', 26, 14.0, 0.8);
      }}
    }}

    // ═══ STAGE 9: Stacked Attentions Output and W_O ═══
    // We concatenate all layers along X to form [seqLen x hidden_size]
    const concatX = outX + heatW/2 + stepX;
    
    // Draw the giant concatenated flat box stretching backwards in Y to represent all heads combined
    // Or we just draw one box that takes the output array
    // Since we don't extract the fully concatenated layer output heatmap explicitly, we'll draw a unified flat box
    const totalYSize = numHeads > 1 ? (numHeads - 1) * yLayerGap + 0.5 : 2.0;
    const centerY = totalYSize / 2;
    addBox(concatX, centerY, vRowZ, heatW * 0.5, totalYSize, heatH, 0x58a6ff, '');
    addTextSprite(concatX, centerY, vRowZ + heatH/2 + 1.0, 'Stacked Attention', '#58a6ff', 18, 6.5, 0.6);
    addTextSprite(concatX, centerY - 1.0, vRowZ + heatH/2 + 1.8, 'Head Outputs', '#58a6ff', 18, 5.0, 0.6);
    addTextSprite(concatX + heatW*0.25 + 0.8, centerY, vRowZ + heatH/2 + 1.0, seqLen + '\\u00D7' + CFG.hidden_size, '#8b949e', 16, 2.5, 0.5);

    // Connector from each head's O to the Concat Box
    for (let hi = 0; hi < numHeads; hi++) {{
      const yOff = hi * yLayerGap;
      addLine(new THREE.Vector3(outX + heatW/2 + 0.1, yOff, vRowZ), new THREE.Vector3(concatX - heatW * 0.25 - 0.1, yOff, vRowZ), 0x58a6ff);
    }}

    // Draw W_O Matrix Box
    const woX = concatX + heatW*0.25 + stepX;
    addBox(woX, centerY, vRowZ, 2.0, totalYSize, 2.0, 0x8b949e, 'W_O');
    addTextSprite(woX, centerY, vRowZ + 1.0 + 1.0, 'Output Weight Matrix', '#8b949e', 18, 7.5, 0.6);
    addTextSprite(woX, centerY, vRowZ + 1.0 + 1.8, CFG.hidden_size + '\\u00D7' + CFG.hidden_size, '#8b949e', 14, 3.5, 0.5);

    // Connector: Concat -> W_O
    addLine(new THREE.Vector3(concatX + heatW*0.25 + 0.1, centerY, vRowZ), new THREE.Vector3(woX - 1.0 - 0.1, centerY, vRowZ), 0x8b949e);

    // Draw Final Layer Out Heatmap Box
    const layerOutX = woX + 1.0 + stepX;
    addBox(layerOutX, centerY, vRowZ, heatW, 0.1, heatH, 0x58a6ff, 'Layer Out');
    addTextSprite(layerOutX, centerY, vRowZ + heatH/2 + 1.0, 'Layer Out', '#e6edf3', 24, 4.0, 0.7);
    addTextSprite(layerOutX + heatW/2 + 0.8, centerY, vRowZ, seqLen + '\\u00D7' + CFG.hidden_size, '#8b949e', 16, 2.5, 0.5);

    // Connector: W_O -> Layer Out
    addLine(new THREE.Vector3(woX + 1.0 + 0.1, centerY, vRowZ), new THREE.Vector3(layerOutX - heatW/2 - 0.1, centerY, vRowZ), 0xe6edf3);

    // ═══ ADD+NORM after attention ═══
    const ln2X = layerOutX + heatW/2 + stepX;
    const ln2EdgeRight = ln2X + blockW / 2;

    // Straight residual line that totally bypasses the Attention Head.
    // In diagram, residual skips along the bottom. We'll send it WAY OUT in -Z, over the top.
    const resZ = qRowZ - 2.5; 
    addLine(new THREE.Vector3(layerX - 0.5, 0, 0), new THREE.Vector3(layerX - 0.5, 0, resZ), 0x58a6ff);
    addLine(new THREE.Vector3(layerX - 0.5, 0, resZ), new THREE.Vector3(ln2X, 0, resZ), 0x58a6ff);
    addLine(new THREE.Vector3(ln2X, 0, resZ), new THREE.Vector3(ln2X, 0, 0), 0x58a6ff);
    addTextSprite((layerX + ln2X) / 2, 0, resZ - 0.6, 'Residual', '#58a6ff', 20, 4.5, 0.6);

    // Final connector to O. Wait, LN2 box is placed at Y=0, Z=0. O heatmap is at Z=vRowZ (+5.0).
    addBox(ln2X, 0, 0, blockW, 0.1, blockH * 1.5, 0x58a6ff, 'Add+Norm'); // Flat block!

    // Connector: Layer Out → Add+Norm
    // Go from LayerOut Box (Y=centerY, Z=vRowZ) right, then curve down to (Y=0, Z=0) where Add+Norm is.
    addLine(new THREE.Vector3(layerOutX + heatW/2 + 0.1, centerY, vRowZ), new THREE.Vector3(ln2X - blockW/2 - 0.5, centerY, vRowZ), 0x58a6ff);
    addLine(new THREE.Vector3(ln2X - blockW/2 - 0.5, centerY, vRowZ), new THREE.Vector3(ln2X - blockW/2 - 0.5, 0, vRowZ), 0x58a6ff);
    addLine(new THREE.Vector3(ln2X - blockW/2 - 0.5, 0, vRowZ), new THREE.Vector3(ln2X - blockW/2 - 0.5, 0, 0), 0x58a6ff);
    addLine(new THREE.Vector3(ln2X - blockW/2 - 0.5, 0, 0), new THREE.Vector3(ln2X - blockW/2 - 0.1, 0, 0), 0x58a6ff);

    // ═══ MLP (compact boxes) ═══
    const mlpStartX = ln2X + 3.0;
    const mlpBoxW = 1.3;
    const mlpBoxH = 0.9;
    const mlpBoxD = blockD * 0.8;
    const mlpInnerSpacing = 2.2;
    const gateZ = 1.2;

    addLine(new THREE.Vector3(ln2EdgeRight + 0.1, 0, 0), new THREE.Vector3(mlpStartX - mlpBoxW/2 - 0.3, 0, 0), 0xf0883e);

    addBox(mlpStartX, 0, gateZ, mlpBoxW, mlpBoxH, mlpBoxD, 0xf0883e, 'Gate');
    addBox(mlpStartX, 0, -gateZ, mlpBoxW, mlpBoxH, mlpBoxD, 0xbc8cff, 'Up');
    addLine(new THREE.Vector3(mlpStartX - mlpBoxW/2 - 0.3, 0, 0), new THREE.Vector3(mlpStartX - mlpBoxW/2 - 0.05, 0, gateZ), 0xf0883e);
    addLine(new THREE.Vector3(mlpStartX - mlpBoxW/2 - 0.3, 0, 0), new THREE.Vector3(mlpStartX - mlpBoxW/2 - 0.05, 0, -gateZ), 0xbc8cff);

    const siluMulX = mlpStartX + mlpInnerSpacing;
    addBox(siluMulX, 0, 0, mlpBoxW * 1.3, mlpBoxH, mlpBoxD, 0xf7c948, 'SiLU \\u2299');
    addLine(new THREE.Vector3(mlpStartX + mlpBoxW/2 + 0.1, 0, gateZ), new THREE.Vector3(siluMulX - mlpBoxW * 0.65 - 0.1, 0, 0), 0xf0883e);
    addLine(new THREE.Vector3(mlpStartX + mlpBoxW/2 + 0.1, 0, -gateZ), new THREE.Vector3(siluMulX - mlpBoxW * 0.65 - 0.1, 0, 0), 0xbc8cff);

    const downX = siluMulX + mlpInnerSpacing;
    addBox(downX, 0, 0, mlpBoxW, mlpBoxH, mlpBoxD, 0x39d2c0, 'Down');
    addLine(new THREE.Vector3(siluMulX + mlpBoxW * 0.65 + 0.1, 0, 0), new THREE.Vector3(downX - mlpBoxW/2 - 0.1, 0, 0), 0xf7c948);

    addTextSprite((mlpStartX + downX) / 2, 2.5, 0, 'Feed-Forward (MLP)', '#f0883e', 22, 9.0, 0.7);

    const mlpEndX = downX + mlpBoxW / 2;
    const mlpX = mlpEndX;

    // Connect to embedding or previous layer's MLP
    if (isFirstRendered && li === 0) {{
      addLine(
        new THREE.Vector3(embedOutEdgeX + 0.1, 0, 0),
        new THREE.Vector3(layerX - 0.5 - 0.1, 0, 0),
        0x58a6ff
      );
    }} else if (prevMlpXForConnect !== null) {{
      addLine(
        new THREE.Vector3(prevMlpXForConnect + 0.1, 0, 0),
        new THREE.Vector3(layerX - 0.5 - 0.1, 0, 0),
        0x3fb950
      );
    }}

    prevMlpXForConnect = mlpX;
    isFirstRendered = false;

    // Advance cursor for next layer
    cursor = mlpX + layerSpacing;
  }}

  // Connect last layer MLP to final blocks

  // ═══ FINAL LN ═══
  const finalLnX = cursor;
  addBox(finalLnX, 0, 0, blockW * 0.8, blockH, blockD * 0.8, 0xd29922, 'LN final');
  addLine(
    new THREE.Vector3(prevMlpXForConnect + 0.1, 0, 0),
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
      tc.width = 320;
      tc.height = 64;
      const tx = tc.getContext('2d');
      tx.fillStyle = color;
      tx.font = 'bold 32px JetBrains Mono, monospace'; // [INCREASED FONT SIZE]
      tx.textAlign = 'center';
      tx.fillText(text, 160, 48);
      const tt = new THREE.CanvasTexture(tc);
      const tm = new THREE.SpriteMaterial({{ map: tt, transparent: true }});
      const ts = new THREE.Sprite(tm);
      ts.position.set(x, logitScale + 1.5, 0);
      ts.scale.set(4, 0.8, 1); // [INCREASED SCALE]
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
      lvC.width = 192;
      lvC.height = 48;
      const lvx = lvC.getContext('2d');
      lvx.fillStyle = '#e6edf3';
      lvx.font = '24px JetBrains Mono, monospace'; // [INCREASED FONT SIZE]
      lvx.textAlign = 'center';
      lvx.fillText(d.logit.toFixed(1), 96, 32);
      const lvt = new THREE.CanvasTexture(lvC);
      const lvsm = new THREE.SpriteMaterial({{ map: lvt, transparent: true }});
      const lvs = new THREE.Sprite(lvsm);
      lvs.position.set(logitGroupX, logitH + 0.35, zPos);
      lvs.scale.set(2.0, 0.5, 1); // [INCREASED SCALE]
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
      pvC.width = 192;
      pvC.height = 48;
      const pvx = pvC.getContext('2d');
      pvx.fillStyle = '#e6edf3';
      pvx.font = '24px JetBrains Mono, monospace'; // [INCREASED FONT SIZE]
      pvx.textAlign = 'center';
      pvx.fillText((d.prob * 100).toFixed(1) + '%', 96, 32);
      const pvt = new THREE.CanvasTexture(pvC);
      const pvsm = new THREE.SpriteMaterial({{ map: pvt, transparent: true }});
      const pvs = new THREE.Sprite(pvsm);
      pvs.position.set(probGroupX, probH + 0.35, zPos);
      pvs.scale.set(2.0, 0.5, 1); // [INCREASED SCALE]
      pvs.userData.arch = true;
      threeScene.add(pvs);

      // Token label (shared, to the right of prob bar)
      const tkC = document.createElement('canvas');
      tkC.width = 384;
      tkC.height = 64;
      const tkx = tkC.getContext('2d');
      tkx.fillStyle = i === 0 ? '#f0e68c' : '#8b949e';
      tkx.font = (i === 0 ? 'bold ' : '') + '28px JetBrains Mono, monospace'; // [INCREASED FONT SIZE]
      tkx.textAlign = 'left';
      const tokenStr = (d.word || '').trim() || '(empty)';
      tkx.fillText(tokenStr.slice(0, 12), 20, 42);
      const tkt = new THREE.CanvasTexture(tkC);
      const tksm = new THREE.SpriteMaterial({{ map: tkt, transparent: true }});
      const tks = new THREE.Sprite(tksm);
      tks.position.set(probGroupX + 2.5, probH / 2, zPos);
      tks.scale.set(4.0, 0.8, 1); // [INCREASED SCALE]
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
  
  // Target the volumetric center of the attention head stack to fix pendulum rotations
  const numHeadsTarget = (DATA.attention && DATA.attention[0]) ? DATA.attention[0].length : 1;
  const stackCenterY = numHeadsTarget > 1 ? (numHeadsTarget - 1) * 3.5 / 2 : 0;
  
  threeCamera.position.set(centerX - extent * 0.4, stackCenterY + extent * 0.4, extent * 0.6);
  threeCamera.lookAt(centerX, stackCenterY, 0);
  defaultCamPos = threeCamera.position.clone();
  threeControls.target.set(centerX, stackCenterY, 0);
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
