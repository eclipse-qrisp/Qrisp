#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")
DOCS_DIR="$SCRIPT_DIR"

TMPDIR=$(mktemp -d /tmp/qrisp-docs-XXXXXX)
VENV_DIR="$TMPDIR/.venv"

echo "Temporary doc env: $VENV_DIR"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip

cd "$DOCS_DIR"
"$VENV_DIR/bin/pip" install -r requirements_doc.txt

PANDOC_PATH=$("$VENV_DIR/bin/python" -c 'import pypandoc; print(pypandoc.get_pandoc_path())')
ln -sf "$PANDOC_PATH" "$VENV_DIR/bin/pandoc"

PATH="$VENV_DIR/bin:$PATH" make html

echo
echo "Build succeeded."
echo "HTML output: $DOCS_DIR/build/html"
echo "Temporary env kept at: $VENV_DIR"