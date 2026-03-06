import re
with open('debug_output.html', 'r', encoding='utf-8') as f:
    text = f.read()

match = re.search(r'<script>(.*?)</script>', text, re.DOTALL)
if match:
    with open('debug_output.js', 'w') as out:
        out.write(match.group(1))
