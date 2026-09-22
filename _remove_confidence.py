import os
BASE = r'E:\classroom_expression_analysis'
cfg = os.path.join(BASE, 'config.py')
with open(cfg, 'r', encoding='utf-8') as f:
    lines = f.readlines()
new_lines = [l for l in lines if 'CONFIDENCE_THRESHOLD' not in l]
with open(cfg, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('[1/2] config.py: removed CONFIDENCE_THRESHOLD')
