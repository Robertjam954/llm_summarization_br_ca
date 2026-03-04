import json

with open("notebooks/03_executed.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

found = False
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                found = True
                print(f"=== CELL {i} ERROR ===")
                print(f"ename: {out['ename']}")
                print(f"evalue: {out['evalue']}")
                tb = out.get("traceback", [])
                for line in tb[-3:]:
                    # Strip ANSI codes
                    import re
                    clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
                    print(clean[:300])
                print()

if not found:
    print("No errors found in executed notebook cells.")
    # Check if execution stopped early
    total_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    executed = sum(1 for c in nb["cells"] if c["cell_type"] == "code" and c.get("outputs"))
    print(f"Code cells: {total_code}, with outputs: {executed}")
