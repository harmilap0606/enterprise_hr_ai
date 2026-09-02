import json, re, sys
sys.stdout.reconfigure(encoding='utf-8')

def strip_ansi(s):
    return re.sub(r'\x1b\[[0-9;]*m', '', s)

def get_text(outputs):
    parts = []
    for out in outputs:
        ot = out.get('output_type','')
        if ot == 'stream':
            parts.append(''.join(out.get('text',[])))
        elif ot in ('execute_result','display_data'):
            d = out.get('data',{})
            if 'text/plain' in d:
                parts.append(''.join(d['text/plain']))
        elif ot == 'error':
            parts.append('ERROR: ' + out.get('ename','') + ': ' + out.get('evalue',''))
            parts.append('\n'.join(strip_ansi(l) for l in out.get('traceback',[])))
    return strip_ansi(''.join(parts)).strip()

with open(r'C:\Users\ASUS\Desktop\enterprise_hr_ai\notebooks\15_employee_intelligence.ipynb','r',encoding='utf-8') as f:
    nb = json.load(f)

errors = 0
lines = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    txt = get_text(cell.get('outputs',[]))
    if 'ERROR:' in txt:
        errors += 1
    lines.append(f'--- CELL {i} ---')
    lines.append(txt if txt else '(no output)')
    lines.append('')

lines.insert(0, f'TOTAL ERRORS: {errors}\n')
out = '\n'.join(lines)
with open(r'C:\Users\ASUS\Desktop\enterprise_hr_ai\nb15_outputs.txt','w',encoding='utf-8') as f:
    f.write(out)
print(f'Written {len(out)} chars, {errors} error(s)')
