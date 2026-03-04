"""Check executed notebook for errors."""
import json, re, sys

nb_path = sys.argv[1] if len(sys.argv) > 1 else "notebooks/02_executed.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

found = False
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                found = True
                ename = out.get("ename", "?")
                evalue = out.get("evalue", "?")[:300]
                print(f"CELL {i}: {ename}: {evalue}")

if not found:
    total_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    executed = sum(1 for c in nb["cells"] if c["cell_type"] == "code" and c.get("outputs"))
    print(f"No errors. Code cells: {total_code}, with outputs: {executed}")
