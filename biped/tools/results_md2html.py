#!/usr/bin/env python3
"""RESULTS.md → results/RESULTS.html. 자체완결(외부 의존 0)·다크모드.

★왜 스크립트로 두나 (2026-08-14) — 손으로 HTML 을 쓰면 MD 와 갈라진다.
  실제로 한 번 갈라져서 "3% 일치" 오독이 HTML 에만 남을 뻔했다. **MD 가 원본**이고
  이건 옮기기만 한다.
⚠MD 표 안에 `|` 를 쓰지 말 것 — 셀이 갈라진다(`|±|` 로 한 번 깨졌다).
"""
import re, html, os
HERE = os.path.dirname(os.path.abspath(__file__))
PACE = os.path.join(os.path.dirname(HERE), "emb", "pace")
def esc(t): return html.escape(t, quote=False)
def inline(t):
    t = esc(t)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    return re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', t)
def convert(src):
    out, lines, i = [], src.split("\n"), 0
    while i < len(lines):
        L = lines[i]
        if L.startswith("```"):
            blk = []; i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                blk.append(esc(lines[i])); i += 1
            out.append("<pre>" + "\n".join(blk) + "</pre>"); i += 1; continue
        if L.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")]); i += 1
            rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
            t = ["<table>"]
            for k, r in enumerate(rows):
                tag = "th" if k == 0 else "td"
                cells = []
                for c in r:
                    cls = ""
                    if re.match(r'^[⚠★❌✅]?\s*[-+±−]?[\d.]', c) or c == "—": cls = ' class="n"'
                    if "✅" in c: cls = ' class="ok"'
                    elif "❌" in c: cls = ' class="no"'
                    elif "⚠" in c: cls = ' class="wa"'
                    cells.append(f"<{tag}{cls}>{inline(c)}</{tag}>")
                t.append("<tr>" + "".join(cells) + "</tr>")
            out.append("\n".join(t) + "</table>"); continue
        if L.startswith("### "): out.append(f"<h3>{inline(L[4:])}</h3>")
        elif L.startswith("## "): out.append(f"<h2>{inline(L[3:])}</h2>")
        elif L.startswith("# "):  out.append(f"<h1>{inline(L[2:])}</h1>")
        elif L.startswith("> "):  out.append(f'<div class="quote">{inline(L[2:])}</div>')
        elif L.startswith("---"): pass
        elif L.strip():
            cls = ' class="warn"' if L.lstrip().startswith(("⚠", "★")) else ""
            buf = [L]
            while i + 1 < len(lines) and lines[i+1].strip() and lines[i+1][0] not in "|#>`-":
                i += 1; buf.append(lines[i])
            out.append(f"<p{cls}>{inline(' '.join(buf))}</p>")
        i += 1
    return "\n".join(out)
CSS = """
:root{--bg:#fbfbfa;--surf:#fff;--line:#e6e4e0;--ink:#1c1b19;--ink2:#4a4744;--mut:#8a857e;
 --acc:#3b6ea5;--ok:#15803d;--wa:#b45309;--no:#b91c1c}
@media(prefers-color-scheme:dark){:root{--bg:#161513;--surf:#1e1d1a;--line:#332f2b;
 --ink:#eceae6;--ink2:#c3bdb5;--mut:#8d867d;--acc:#7aa7d4;--ok:#5fbf7f;--wa:#e0a458;--no:#e58a84}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.7 -apple-system,'Noto Sans KR',Pretendard,system-ui,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:40px 22px 90px}
h1{font-size:26px;letter-spacing:-.02em;margin:0 0 26px}
h2{font-size:19px;margin:46px 0 12px;padding-top:20px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:26px 0 6px;color:var(--ink2)}
p{margin:11px 0}
p.warn{background:var(--surf);border:1px solid var(--line);border-left:3px solid var(--wa);
 border-radius:0 6px 6px 0;padding:11px 15px;font-size:14px;color:var(--ink2)}
p.warn b{color:var(--ink)}
.quote{border-left:3px solid var(--acc);background:var(--surf);padding:11px 15px;
 margin:12px 0;border-radius:0 6px 6px 0;font-size:14px;color:var(--ink2)}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px;
 font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--mut);font-size:12px;letter-spacing:.04em;
 padding:0 10px 7px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:7px 10px;border-bottom:1px solid var(--line)}
td.n,th.n{text-align:right}
td.ok{color:var(--ok);font-weight:600}td.no{color:var(--no);font-weight:600}
td.wa{color:var(--wa);font-weight:600}
tr:last-child td{border-bottom:none}
pre{background:var(--surf);border:1px solid var(--line);border-radius:8px;padding:16px 18px;
 overflow-x:auto;font:12.5px/1.75 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink2)}
code{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--surf);
 border:1px solid var(--line);border-radius:4px;padding:1px 5px}
pre code{border:0;background:none;padding:0}
"""
if __name__ == "__main__":
    md = open(os.path.join(PACE, "RESULTS.md"), encoding="utf-8").read()
    out = os.path.join(PACE, "results", "RESULTS.html")
    open(out, "w", encoding="utf-8").write(
        '<!DOCTYPE html>\n<html lang="ko"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'<title>액추에이터 식별 — 최종값</title>\n<style>{CSS}</style></head>\n'
        f'<body><div class="wrap">\n{convert(md)}\n</div></body></html>\n')
    # ★생성 직후 **표 정합성**을 검사한다 (2026-08-14). MD 표 안에 `|` 를 쓰면
    #   셀이 갈라지는데 브라우저는 조용히 렌더한다 — `|±|` 로 한 번 당했다.
    import re as _re
    doc = open(out, encoding="utf-8").read()
    bad = []
    for i, tb in enumerate(_re.findall(r"<table>.*?</table>", doc, _re.S)):
        n = [len(_re.findall(r"<t[hd]", r))
             for r in _re.findall(r"<tr>.*?</tr>", tb, _re.S)]
        if len(set(n)) > 1:
            bad.append(f"표{i+1} 열 수 {n}")
    if bad:
        raise SystemExit("✗ 표가 깨졌다 — MD 셀 안의 `|` 를 확인할 것:\n  "
                         + "\n  ".join(bad))
    print(f"✓ {os.path.relpath(out)}  ({os.path.getsize(out)/1024:.0f} KB) · 표 정합 OK")
