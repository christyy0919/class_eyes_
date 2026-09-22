import os
p = os.path.join(r'E:\classroom_expression_analysis', 'scripts', 'run_all.py')
with open(p, 'r', encoding='utf-8') as f:
    code = f.read()
code = code.replace(
    '    print(f\"    class_stacked_area.png\")\n    print(f\"    class_heatmap.png\")',
    '    print(f\"    class_stacked_area.png\")\n    print(f\"    emotion_fine_trend_line.png  (7-class detail)\")\n    print(f\"    class_heatmap.png\")'
)
with open(p, 'w', encoding='utf-8') as f:
    f.write(code)
print('run_all.py: added fine trend line to output list')
