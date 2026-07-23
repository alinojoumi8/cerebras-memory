"""Run a Hermes MCP configuration command with deterministic confirmations.

Windows PowerShell 5.1 writes piped native-process input as UTF-16, while the
Hermes Python CLI reads stdin as text.  Feeding confirmations from Python keeps
the installer non-interactive and avoids silently accepted ``Cancelled`` exits.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    command = sys.argv[1:]
    if not command:
        print("A Hermes command is required.", file=sys.stderr)
        return 2

    result = subprocess.run(
        command,
        input="y\ny\ny\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    combined = f"{result.stdout}\n{result.stderr}".casefold()
    if "cancelled" in combined or "canceled" in combined:
        return 2
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
