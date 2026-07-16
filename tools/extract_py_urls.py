import re
import os
import sys

URL_RE = re.compile(r'https?://[^\s<>"\'\]\)]+')

exclude_dirs = {"node_modules", ".git", ".venv", ".eggs", "dist", "build",
                "__pycache__", ".ruff_cache", ".pytest_cache"}

extensions = (".py", ".ipynb")

output_dir = sys.argv[1] if len(sys.argv) > 1 else "lychee"
output_file = os.path.join(output_dir, "urls_from_src.md")
os.makedirs(output_dir, exist_ok=True)

all_urls = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    for f in files:
        if not f.endswith(extensions):
            continue
        path = os.path.join(root, f)
        try:
            with open(path) as fh:
                content = fh.read()
        except Exception:
            continue
        for m in URL_RE.finditer(content):
            url = m.group().rstrip(".,;:!?)")
            if url.startswith(("http://", "https://")):
                all_urls.append((path, url))

with open(output_file, "w") as f:
    f.write("# URLs extracted from Python and Jupyter files\n\n")
    for path, url in sorted(set(all_urls)):
        f.write(f"- [{url}]({url})\n")

print(f"Extracted {len(set(all_urls))} URLs from .py and .ipynb files to {output_file}")
