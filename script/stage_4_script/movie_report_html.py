"""Create a portable HTML view of this report's Markdown without web assets."""
import base64
import html
from pathlib import Path
import re


def inline(text):
    text = html.escape(text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)', r'<a href="\2">\1</a>', text)


def render(markdown_path):
    markdown_path = Path(markdown_path)
    lines = markdown_path.read_text().splitlines()
    parts = []
    paragraph = []
    in_table = False

    def flush():
        if paragraph:
            parts.append('<p>' + inline(' '.join(paragraph)) + '</p>')
            paragraph.clear()

    for line in lines:
        if not line.startswith('|') and in_table:
            parts.append('</tbody></table>')
            in_table = False
        if not line.strip():
            flush()
        elif line.startswith('#'):
            flush()
            level = len(line) - len(line.lstrip('#'))
            parts.append(f'<h{level}>' + inline(line[level:].strip()) + f'</h{level}>')
        elif line.startswith('!['):
            flush()
            match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            if match:
                alt, relative = match.groups()
                image = markdown_path.parent / relative
                data = base64.b64encode(image.read_bytes()).decode('ascii')
                parts.append(f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{data}"></figure>')
        elif line.startswith('|'):
            flush()
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            if all(re.fullmatch(r':?-+:?', cell) for cell in cells):
                continue
            if not in_table:
                parts.append('<table><thead><tr>' + ''.join('<th>' + inline(cell) + '</th>' for cell in cells) + '</tr></thead><tbody>')
                in_table = True
            else:
                parts.append('<tr>' + ''.join('<td>' + inline(cell) + '</td>' for cell in cells) + '</tr>')
        else:
            paragraph.append(line)
    flush()
    if in_table:
        parts.append('</tbody></table>')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Stage 4 movie classification with LSTM and GRU</title>
<style>
body{max-width:960px;margin:36px auto;padding:0 24px;color:#171717;font:15px/1.5 Arial,sans-serif}
h1{font-size:26px;line-height:1.2}h2{font-size:21px;margin-top:28px}h3{font-size:17px;margin-top:22px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d9d9d9;padding:8px;text-align:right}
th{background:#eaf0f4}th:first-child,td:first-child{text-align:left}figure{margin:16px 0}img{width:100%;height:auto}
code{font-size:.9em;overflow-wrap:anywhere}a{color:#214f77}
@media print{body{font-size:10pt;line-height:1.15;margin:0;max-width:none;padding:0}h1{font-size:16pt}h2{font-size:13pt}h3{font-size:11pt}table{font-size:9pt}figure,table{break-inside:avoid}h2,h3{break-after:avoid}}
</style><main>''' + '\n'.join(parts) + '</main></html>'
    target = markdown_path.with_suffix('.html')
    target.write_text(document)
    return target
