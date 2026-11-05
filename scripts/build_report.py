"""Build Report/Technical_Report.pdf from Report/Technical_Report.md.

Markdown -> HTML (python-markdown, tables + footnotes) -> PDF with a headless
Chromium-based browser (Chrome, Chromium or Edge). Image paths in the
Markdown are relative to Report/.

    python scripts/build_report.py [--browser PATH]
"""
import argparse
import os
import shutil
import subprocess

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "Report")
CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'Segoe UI', 'Noto Sans', 'Noto Sans Telugu', 'Nirmala UI', Arial, sans-serif;
       font-size: 10.5pt; line-height: 1.42; color: #111; }
h1 { font-size: 19pt; margin-bottom: 2pt; } h2 { font-size: 14pt; border-bottom: 1px solid #bbb;
     padding-bottom: 2pt; margin-top: 18pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 12pt; page-break-after: avoid; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #aaa; padding: 3pt 5pt; vertical-align: top; }
th { background: #eee; }
code { font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: 8.8pt; }
pre { background: #f5f5f5; padding: 6pt; font-size: 8.5pt; white-space: pre-wrap; }
img { max-width: 100%; page-break-inside: avoid; }
figure { margin: 6pt 0; page-break-inside: avoid; } figcaption { font-size: 8.8pt; color: #333; }
"""

CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
]


def find_browser(explicit):
    for c in ([explicit] if explicit else []) + CANDIDATES:
        p = c if os.path.isabs(c) else shutil.which(c)
        if p and os.path.exists(p):
            return p
    raise SystemExit("No Chromium-based browser found; pass --browser")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", default=None)
    args = ap.parse_args()
    src = os.path.join(REPORT, "Technical_Report.md")
    body = markdown.markdown(open(src, encoding="utf-8").read(),
                             extensions=["tables", "fenced_code", "footnotes", "toc", "attr_list"])
    html_path = os.path.join(REPORT, "Technical_Report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'><title>Technical Report</title>"
                "<style>{}</style></head><body>{}</body></html>".format(CSS, body))
    pdf = os.path.join(REPORT, "Technical_Report.pdf")
    browser = find_browser(args.browser)
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf=" + pdf, "file:///" + html_path.replace("\\", "/")],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Wrote", pdf)


if __name__ == "__main__":
    main()
