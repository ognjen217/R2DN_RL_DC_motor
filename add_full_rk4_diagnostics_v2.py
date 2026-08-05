#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import shutil
import sys

TARGET = Path("src/r2dn_dc_motor/validation/thermal_test_bank.py")
MARKER = "FULL/RK4 diagnostic:"
HELPER_NAME = "_summarize_full_rk4_result"

HELPER_LINES = [
    "def _summarize_full_rk4_result(result: object) -> str:\n",
    "    \"\"\"Return a compact, array-safe summary for failed FULL/RK4 rollouts.\"\"\"\n",
    "    import numpy as np\n",
    "\n",
    "    try:\n",
    "        fields = vars(result)\n",
    "    except TypeError:\n",
    "        return repr(result)\n",
    "\n",
    "    summary: dict[str, object] = {}\n",
    "    for name, value in fields.items():\n",
    "        if isinstance(value, np.ndarray):\n",
    "            array = np.asarray(value)\n",
    "            array_summary: dict[str, object] = {\n",
    "                \"shape\": tuple(int(item) for item in array.shape),\n",
    "                \"dtype\": str(array.dtype),\n",
    "            }\n",
    "            if array.size and np.issubdtype(array.dtype, np.number):\n",
    "                finite = array[np.isfinite(array)]\n",
    "                if finite.size:\n",
    "                    array_summary.update(\n",
    "                        {\n",
    "                            \"min\": float(np.min(finite)),\n",
    "                            \"max\": float(np.max(finite)),\n",
    "                            \"last\": np.asarray(array[-1]).tolist(),\n",
    "                        }\n",
    "                    )\n",
    "                else:\n",
    "                    array_summary[\"finite_values\"] = 0\n",
    "            summary[name] = array_summary\n",
    "        elif isinstance(value, (str, int, float, bool, type(None))):\n",
    "            summary[name] = value\n",
    "        else:\n",
    "            text = repr(value)\n",
    "            summary[name] = text if len(text) <= 300 else text[:297] + \"...\"\n",
    "\n",
    "    return repr(summary)\n",
]


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: file not found: {TARGET}", file=sys.stderr)
        return 1

    original = TARGET.read_text(encoding="utf-8")
    if MARKER in original:
        print("Diagnostic patch is already applied.")
        return 0

    lines = original.splitlines(keepends=True)

    if f"def {HELPER_NAME}(" not in original:
        indexes = [
            i for i, line in enumerate(lines)
            if line.startswith("def run_thermal_test_bank(")
        ]
        if len(indexes) != 1:
            print("ERROR: could not uniquely locate run_thermal_test_bank().", file=sys.stderr)
            return 1
        insert_at = indexes[0]
        lines[insert_at:insert_at] = HELPER_LINES + ["\n", "\n"]

    raise_index = None
    for i, line in enumerate(lines):
        if "raise RuntimeError(" not in line:
            continue
        nearby = "".join(lines[i:min(i + 5, len(lines))])
        if "FULL/RK4 terminated" in nearby:
            raise_index = i
            break

    if raise_index is None:
        print("ERROR: could not locate FULL/RK4 RuntimeError block.", file=sys.stderr)
        return 1

    line = lines[raise_index]
    indentation = line[: len(line) - len(line.lstrip())]
    diagnostic_lines = [
        f"{indentation}progress(\n",
        f'{indentation}    "FULL/RK4 diagnostic: " + {HELPER_NAME}(full)\n',
        f"{indentation})\n",
    ]
    lines[raise_index:raise_index] = diagnostic_lines

    backup = TARGET.with_suffix(TARGET.suffix + ".diagnostic.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text("".join(lines), encoding="utf-8")

    print(f"Patched: {TARGET}")
    print(f"Backup:  {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
