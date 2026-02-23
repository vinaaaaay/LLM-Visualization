import os
import sys
import json
import subprocess
import time

SHARED_STATE_FILE = "shared_state.json"

BANNER = r"""
╔═══════════════════════════════════════════════════╗
║                                                   ║
║   ⚡  L L M   V I S U A L I Z E R   ⚡           ║
║   ─────────────────────────────────────           ║
║   SmolLM2-360M · Transformer Internals            ║
║                                                   ║
║   Type a prompt → watch the neural network        ║
║   process it in real time on Screen 2.            ║
║                                                   ║
║   Type 'exit' to quit.                            ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
"""

def write_state(prompt: str):
    """Write prompt + timestamp to the shared state file."""
    state = {"prompt": prompt, "timestamp": time.time()}
    with open(SHARED_STATE_FILE, "w") as f:
        json.dump(state, f)

def cleanup():
    """Remove shared state file on exit."""
    for path in (SHARED_STATE_FILE, "shared_prompt.txt"):
        if os.path.exists(path):
            os.remove(path)

def main():
    # Find the python executable in the current environment
    python_exe = sys.executable

    # Kill any existing Streamlit on port 8501
    subprocess.run("lsof -ti :8501 | xargs kill -9 2>/dev/null", shell=True, capture_output=True)
    time.sleep(1)

    # Launch Streamlit visualizer in background — use the same Python
    # so it runs in the same conda env, and DON'T suppress output
    print("--- Launching visualization dashboard... ---")
    log_file = open("streamlit_log.txt", "w")
    proc = subprocess.Popen(
        [python_exe, "-m", "streamlit", "run", "visualizer.py",
         "--server.port", "8501",
         "--browser.gatherUsageStats", "false"],
        stdout=log_file,
        stderr=log_file,
    )
    time.sleep(5)

    # Check if Streamlit started successfully
    if proc.poll() is not None:
        log_file.close()
        print("\n❌ Streamlit failed to start! Error log:")
        with open("streamlit_log.txt", "r") as f:
            print(f.read())
        return

    print("✅ Dashboard should be open at http://localhost:8501")
    print(BANNER)

    try:
        while True:
            user_input = input("\033[92m[PROMPT]\033[0m ▸ ")

            if user_input.strip().lower() in ("exit", "quit", "q"):
                print("\n👋 Shutting down...")
                cleanup()
                proc.terminate()
                log_file.close()
                break

            if not user_input.strip():
                continue

            write_state(user_input.strip())
            print(f"\033[90m   ↳ Sent to visualizer. Watch the dashboard!\033[0m")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Cleaning up...")
        cleanup()
        proc.terminate()
        log_file.close()

if __name__ == "__main__":
    main()