"""Global Unicode and UTF-8 Compatibility System for VideoForge.
Eliminates Windows default 'charmap' / cp1252 encoding errors globally across
all modules, file I/O, subprocesses, standard streams, and logging handlers.
"""
import builtins
import io
import logging
import os
import subprocess
import sys
from typing import Any, Optional, Union

# Set global environment indicators for Python child processes
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"


def configure_streams():
    """Reconfigures standard input/output/error streams to strictly use UTF-8 with safe error replacement."""
    for stream_name in ("stdout", "stderr", "stdin"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


# 1. Patch builtins.open to default to UTF-8 for text mode on Windows
_original_open = builtins.open


def utf8_open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    """Wrapper around builtins.open that guarantees UTF-8 for text files unless binary mode is specified."""
    mode_str = str(mode) if mode is not None else "r"
    if "b" not in mode_str:
        if encoding is None:
            encoding = "utf-8"
        if errors is None:
            errors = "replace"
    return _original_open(
        file,
        mode=mode,
        buffering=buffering,
        encoding=encoding,
        errors=errors,
        newline=newline,
        closefd=closefd,
        opener=opener,
    )


builtins.open = utf8_open


# 2. Patch subprocess.run and subprocess.Popen for text mode UTF-8
_original_subprocess_run = subprocess.run


def utf8_subprocess_run(*args, **kwargs):
    """Guarantees subprocess text mode decodes and encodes as UTF-8 on Windows."""
    if kwargs.get("text") or kwargs.get("universal_newlines"):
        if "encoding" not in kwargs or kwargs["encoding"] is None:
            kwargs["encoding"] = "utf-8"
        if "errors" not in kwargs or kwargs["errors"] is None:
            kwargs["errors"] = "replace"
    # Ensure subprocess inherits UTF-8 environment
    if "env" in kwargs and kwargs["env"] is not None:
        kwargs["env"].setdefault("PYTHONIOENCODING", "utf-8")
        kwargs["env"].setdefault("PYTHONUTF8", "1")
    return _original_subprocess_run(*args, **kwargs)


subprocess.run = utf8_subprocess_run


class Utf8Popen(subprocess.Popen):
    """Subclass of subprocess.Popen guaranteeing UTF-8 text mode without breaking inheritance."""
    def __init__(self, *args, **kwargs):
        if kwargs.get("text") or kwargs.get("universal_newlines"):
            if "encoding" not in kwargs or kwargs["encoding"] is None:
                kwargs["encoding"] = "utf-8"
            if "errors" not in kwargs or kwargs["errors"] is None:
                kwargs["errors"] = "replace"
        if "env" in kwargs and kwargs["env"] is not None:
            kwargs["env"].setdefault("PYTHONIOENCODING", "utf-8")
            kwargs["env"].setdefault("PYTHONUTF8", "1")
        super().__init__(*args, **kwargs)


subprocess.Popen = Utf8Popen


# 3. Configure logging to handle Unicode safely
class UTF8StreamHandler(logging.StreamHandler):
    """A StreamHandler that emits UTF-8 text safely without charmap crashes."""
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                stream.write(msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace") + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


# Apply configuration immediately
configure_streams()


def is_devanagari_or_hindi(text: str) -> bool:
    """Checks whether the given string contains Devanagari script characters."""
    if not text:
        return False
    return any("\u0900" <= ch <= "\u097F" for ch in text)
