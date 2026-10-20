"""Unified terminal output with colors, progress, and stage indicators."""

import sys
import os
import time
import threading
from enum import Enum
from typing import Optional

# --- color codes (ANSI) ---
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "grey": "\033[90m",
}


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR = _supports_color()


def _c(code: str, text: str) -> str:
    if not _COLOR:
        return text
    return f"{_COLORS.get(code, '')}{text}{_RESET}"


def _cb(text: str) -> str:
    """Bold text."""
    if not _COLOR:
        return text
    return f"{_BOLD}{text}{_RESET}"


# --- stage tags ---
class Stage(str, Enum):
    KG = "KG"
    LOCATE = "LOCATE"
    REPAIR = "REPAIR"
    LLM = "LLM"
    VALIDATE = "VALIDATE"
    DOCKER = "DOCKER"
    PATCH = "PATCH"
    DEBUG = "DEBUG"
    OK = "OK"
    FAIL = "FAIL"
    WARN = "WARN"
    INFO = "INFO"


_STAGE_STYLE = {
    Stage.KG: ("blue", "🏗"),
    Stage.LOCATE: ("magenta", "🔍"),
    Stage.REPAIR: ("cyan", "🔧"),
    Stage.LLM: ("yellow", "🤖"),
    Stage.VALIDATE: ("cyan", "✓"),
    Stage.DOCKER: ("cyan", "🐳"),
    Stage.PATCH: ("magenta", "📝"),
    Stage.DEBUG: ("yellow", "🪲"),
    Stage.OK: ("green", "✅"),
    Stage.FAIL: ("red", "❌"),
    Stage.WARN: ("yellow", "⚠"),
    Stage.INFO: ("grey", "ℹ"),
}


def _stage_prefix(stage: Stage) -> str:
    color, icon = _STAGE_STYLE.get(stage, ("grey", " "))
    label = f"{icon} {stage.value:<7}"
    return _c(color, label)


# --- spinner ---
_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_spinner_active = False
_spinner_thread: Optional[threading.Thread] = None


def _spinner_run(msg: str, stop_event: threading.Event):
    i = 0
    while not stop_event.is_set():
        frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
        sys.stderr.write(f"\r  {_c('cyan', frame)} {_c('dim', msg)}  ")
        sys.stderr.flush()
        i += 1
        time.sleep(0.1)
    sys.stderr.write("\r" + " " * (len(msg) + 20) + "\r")
    sys.stderr.flush()


class Spinner:
    def __init__(self, msg: str):
        self._stop = threading.Event()
        self._msg = msg

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=_spinner_run, args=(self._msg, self._stop), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def update(self, msg: str):
        self._msg = msg


# --- public API ---
def rule(title: str = ""):
    """Print a horizontal rule with optional title."""
    width = 60
    if title:
        pad = max(2, width - len(title) - 4)
        left = pad // 2
        right = pad - left
        print()
        print(f"{_c('grey', '─' * left)} {_cb(title)} {_c('grey', '─' * right)}")
    else:
        print(_c("grey", "─" * width))


def stage(stage: Stage, message: str = ""):
    """Print a stage header line. e.g. [KG] Building knowledge graph..."""
    prefix = _stage_prefix(stage)
    msg = _c("grey", message) if message else ""
    print(f"{prefix} {msg}", flush=True)


def ok(message: str = ""):
    """Success indicator."""
    stage(Stage.OK, message)


def fail(message: str = ""):
    """Failure indicator."""
    stage(Stage.FAIL, message)


def warn(message: str = ""):
    """Warning indicator."""
    stage(Stage.WARN, message)


def info(message: str = ""):
    """Info indicator."""
    stage(Stage.INFO, message)


def detail(key: str, value: str):
    """Print a key-value detail line."""
    print(f"  {_c('dim', key + ':')} {value}", flush=True)


def step(step_num: int, total: int, message: str = ""):
    """Print a numbered step."""
    prefix = f"[{step_num}/{total}]"
    print(f"  {_c('cyan', prefix)} {message}", flush=True)


def progress(current: int, total: int, prefix: str = ""):
    """Inline progress update (overwrites current line)."""
    pct = current / max(total, 1) * 100
    bar_width = 20
    filled = int(bar_width * current / max(total, 1))
    bar = "█" * filled + "░" * (bar_width - filled)
    msg = f"  {prefix} {_c('cyan', bar)} {current}/{total} ({pct:.0f}%)"
    sys.stderr.write(f"\r{msg}")
    sys.stderr.flush()
    if current >= total:
        sys.stderr.write("\n")
        sys.stderr.flush()


def spinner(msg: str) -> Spinner:
    """Context manager for spinner animation. Usage:
    with term.spinner("Running Docker..."):
        result = run_docker()
    """
    return Spinner(msg)


def set_color_mode(enabled: bool):
    global _COLOR
    _COLOR = enabled
