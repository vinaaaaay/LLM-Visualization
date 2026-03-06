"""
visualizer.py — Main Streamlit entry point for the LLM Visualization app.

This is the orchestrator that ties together:
  - model_runner.py  → model loading, inference, data extraction
  - html_builder.py  → HTML/CSS/JS visualization template
"""

import streamlit as st
import json
import os

st.set_page_config(layout="wide", page_title="LLM Visualization — SmolLM2-360M", page_icon="⚡")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=10000, key="datarefresh")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------
from model_runner import load_model, get_model_config, run_model as _run_model

@st.cache_resource
def _cached_load():
    return load_model()

tokenizer, model = _cached_load()
MODEL_CONFIG = get_model_config(model)

# ---------------------------------------------------------------------------
# Wrap run_model with Streamlit caching
# ---------------------------------------------------------------------------
@st.cache_data
def run_model(prompt: str):
    return _run_model(prompt, tokenizer, model)

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
# Import HTML builder
# ---------------------------------------------------------------------------
from html_builder import build_html

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