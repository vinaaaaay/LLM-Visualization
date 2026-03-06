import os
import sys
import json
import subprocess
import time
import shutil

SHARED_STATE_FILE = "shared_state.json"

# ── ANSI color codes ──
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
ITALIC  = "\033[3m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
MAGENTA = "\033[95m"
RED     = "\033[91m"
GRAY    = "\033[90m"
WHITE   = "\033[97m"
BG_DARK = "\033[48;5;234m"

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def term_width():
    return shutil.get_terminal_size((80, 24)).columns

def center(text, width=None):
    w = width or term_width()
    return text.center(w)

def print_banner():
    w = min(term_width(), 64)
    bar = f"{CYAN}{'━' * w}{RESET}"

    print()
    print(bar)
    print()

    title_art = [
        "██╗     ██╗     ███╗   ███╗",
        "██║     ██║     ████╗ ████║",
        "██║     ██║     ██╔████╔██║",
        "██║     ██║     ██║╚██╔╝██║",
        "███████╗███████╗██║ ╚═╝ ██║",
        "╚══════╝╚══════╝╚═╝     ╚═╝",
    ]
    for line in title_art:
        print(center(f"{BOLD}{CYAN}{line}{RESET}", w))

    print()
    print(center(f"{BOLD}{WHITE}V I S U A L I Z E R{RESET}", w))
    print(center(f"{DIM}{GRAY}━━━━━━━━━━━━━━━━━━━{RESET}", w))
    print()
    print(center(f"{YELLOW}SmolLM2-360M{RESET}  {DIM}·{RESET}  {WHITE}Transformer Internals{RESET}", w))
    print()
    print(bar)
    print()
    print(center(f"{WHITE}Type a prompt below and watch the neural network{RESET}", w))
    print(center(f"{WHITE}process it {BOLD}in real time{RESET}{WHITE} on the dashboard.{RESET}", w))
    print()
    print(center(f"{DIM}{GRAY}Type {WHITE}'exit'{GRAY} to quit{RESET}", w))
    print()
    print(bar)
    print()


def spinning_loader(msg, duration=4):
    """Show a spinner while waiting."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        frame = frames[i % len(frames)]
        print(f"\r  {CYAN}{frame}{RESET} {GRAY}{msg}{RESET}", end="", flush=True)
        time.sleep(0.08)
        i += 1
    print(f"\r  {GREEN}✓{RESET} {msg}                    ")


def print_sent(prompt):
    w = min(term_width(), 64)
    print(f"  {GREEN}✓{RESET} {DIM}Sent to visualizer{RESET}")
    print(f"  {DIM}{GRAY}{'─' * (w - 4)}{RESET}")
    print()


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
    python_exe = sys.executable

    clear_screen()

    # Kill any existing Streamlit on port 8501
    subprocess.run("lsof -ti :8501 | xargs kill -9 2>/dev/null", shell=True, capture_output=True)
    time.sleep(1)

    # Launch Streamlit
    print(f"\n  {CYAN}⚡{RESET} {BOLD}Initializing LLM Visualizer...{RESET}\n")

    log_file = open("streamlit_log.txt", "w")
    proc = subprocess.Popen(
        [python_exe, "-m", "streamlit", "run", "visualizer.py",
         "--server.port", "8501",
         "--browser.gatherUsageStats", "false"],
        stdout=log_file,
        stderr=log_file,
    )

    spinning_loader("Loading model weights...", 2)
    spinning_loader("Starting Streamlit dashboard...", 3)

    # Check if Streamlit started
    if proc.poll() is not None:
        log_file.close()
        print(f"\n  {RED}✗ Streamlit failed to start!{RESET}")
        with open("streamlit_log.txt", "r") as f:
            print(f.read())
        return

    print(f"\n  {GREEN}●{RESET} Dashboard live at {BOLD}{CYAN}http://localhost:8501{RESET}")
    print()

    print_banner()

    prompt_count = 0
    try:
        while True:
            prompt_count += 1
            user_input = input(f"  {BOLD}{MAGENTA}[{prompt_count}]{RESET} {WHITE}▸ {RESET}")

            if user_input.strip().lower() in ("exit", "quit", "q"):
                print(f"\n  {YELLOW}⚡{RESET} {DIM}Shutting down...{RESET}\n")
                cleanup()
                proc.terminate()
                log_file.close()
                break

            if not user_input.strip():
                continue

            write_state(user_input.strip())
            print_sent(user_input.strip())

    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}⚡{RESET} {DIM}Interrupted. Cleaning up...{RESET}\n")
        cleanup()
        proc.terminate()
        log_file.close()


if __name__ == "__main__":
    main()