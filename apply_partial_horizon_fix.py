#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import shutil
import sys


TARGET = Path("src/r2dn_dc_motor/validation/thermal_test_bank.py")
ERROR_TEXT = 'raise ValueError("duration must equal one of the locked cumulative horizons")'
ENV_GUARD = 'if os.environ.get("R2DN_ALLOW_PARTIAL_HORIZON") != "1":'


def ensure_os_import(lines: list[str]) -> list[str]:
    if any(line.strip() == "import os" for line in lines):
        return lines

    # Insert after a future import when present; otherwise before the first import.
    for index, line in enumerate(lines):
        if line.startswith("from __future__ import"):
            insert_at = index + 1
            while insert_at < len(lines) and not lines[insert_at].strip():
                insert_at += 1
            lines.insert(insert_at, "import os\n")
            return lines

    for index, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            lines.insert(index, "import os\n")
            return lines

    lines.insert(0, "import os\n")
    return lines


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: file not found: {TARGET}", file=sys.stderr)
        return 1

    original = TARGET.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    if ENV_GUARD in original:
        print("Patch is already applied.")
        return 0

    match_indexes = [
        index for index, line in enumerate(lines) if ERROR_TEXT in line
    ]
    if len(match_indexes) != 1:
        print(
            f"ERROR: expected exactly one matching guard, found {len(match_indexes)}.",
            file=sys.stderr,
        )
        return 1

    index = match_indexes[0]
    raise_line = lines[index]
    indentation = raise_line[: len(raise_line) - len(raise_line.lstrip())]

    lines.insert(index, f'{indentation}{ENV_GUARD}\n')
    lines[index + 1] = f"    {raise_line}"

    lines = ensure_os_import(lines)
    patched = "".join(lines)

    backup = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(patched, encoding="utf-8")

    print(f"Patched: {TARGET}")
    print(f"Backup:  {backup}")
    print('Diagnostic override: R2DN_ALLOW_PARTIAL_HORIZON=1')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
