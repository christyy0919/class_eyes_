import urllib.request, json, urllib.parse

url = 'http://127.0.0.1:8765/api/chart-data?input=' + urllib.parse.quote('e:/classroom_expression_analysis/output')
r = urllib.request.urlopen(url)
d = json.loads(r.read().decode())

print("student_durations keys:", list(d.get('student_durations', {}).keys()))
dd = d['student_durations']
for k, v in dd.items():
    print(f"  {k}: total={v['total_seconds']}s, max={v['max_seconds']}s, blocks={v['block_count']}")
