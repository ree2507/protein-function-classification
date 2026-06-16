import json

with open('04_lstm_v7_kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print()

# Validate all code cells
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = cell['source']
    # Check for broken f-strings
    for j, line in enumerate(src):
        sl = line.rstrip('\n')
        if sl == 'print(f"':
            print(f"ERROR: Cell {i}, line {j}: orphan f-string start")
        # Check for double closing parens at end of line (real syntax error)
        if 'print(f"' in sl and sl.endswith('))'):
            print(f"ERROR: Cell {i}, line {j}: double closing parens at end")
    # Check for empty cells
    if not src or all(s.strip() == '' for s in src):
        print(f"WARNING: Cell {i} is empty")
    # Check for cells with no source
    if not isinstance(src, list):
        print(f"ERROR: Cell {i} has no source list")

print("\nCell summary:")
for i, cell in enumerate(nb['cells']):
    typ = cell['cell_type']
    lines = len(cell['source']) if cell['cell_type'] == 'code' else 0
    src_preview = ''.join(cell['source'][:2]).strip()[:80] if typ == 'code' else (cell['source'][0] if cell['source'] else '').strip()[:80]
    print(f"  {i:2d} [{typ:4s}] L{lines:3d} | {src_preview}")

# Validate all code cells can be compiled (syntax check)
import py_compile, tempfile, os

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    code = ''.join(cell['source'])
    if code.strip() == '':
        continue
    try:
        compile(code, f'<cell_{i}>', 'exec')
    except SyntaxError as e:
        lines_around = code.split('\n')[max(0, e.lineno-3):e.lineno+2]
        print(f"\nSYNTAX ERROR in Cell {i}: {e.msg}")
        print(f"  Line {e.lineno}: {e.text}")
        for ln in lines_around:
            print(f"  | {ln}")

print("\nAll cells syntax-checked!")
