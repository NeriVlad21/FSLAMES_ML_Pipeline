from typing import NoReturn


def fail(message: str) -> NoReturn:
    """Stop a command with a concise, user-facing error."""
    raise SystemExit(f"ERROR: {message}")
