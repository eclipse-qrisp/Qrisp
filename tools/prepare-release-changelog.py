"""Prepare changelog for a new Qrisp release.

Workflow
--------
1. Manually bump the version in ``pyproject.toml`` to the release version
   (e.g. 0.8.2 → 0.9.0).
2. Run ``make release-notes`` (or ``python tools/prepare-release-changelog.py``).
3. Review with ``git diff`` and commit.

The script reads the current version from ``pyproject.toml``, archives
``changelog-dev.rst`` as ``<version>.rst``, inserts it into the
``index.rst`` toctree in descending order, and writes a fresh
``changelog-dev.rst`` for the next minor release.

Run this script only once per release. If something goes wrong partway
through, fix the issue and restore ``changelog-dev.rst`` from git before
re-running.
"""

import re
import shutil
import tomllib
from pathlib import Path


# Path to the changelog RST files, relative to the repo root
CHANGELOG_DIR = Path("documentation/source/general/changelog")


def rename_dev_changelog(version):
    """Step 1: Archive the current dev changelog under the release version.

    The file changelog-dev.rst becomes <version>.rst so the entry
    survives in the versioned toctree alongside 0.1.rst, 0.2.rst, etc.
    """
    dev = CHANGELOG_DIR / "changelog-dev.rst"
    release = CHANGELOG_DIR / f"{version}.rst"
    if not dev.exists():
        print(f"ERROR: {dev} not found")
        return False
    shutil.move(str(dev), str(release))
    print(f"  Renamed: changelog-dev.rst -> {version}.rst")
    return True


def update_index_rst(version):
    """Step 2: Register the new release in the toctree of index.rst.

    The toctree lists versions in descending order (newest first), with
    changelog-dev.rst always at the end.  We scan the existing numeric
    entries and insert the new version before the first one that is
    numerically smaller (or at the end if it is the smallest).
    """
    index_file = CHANGELOG_DIR / "index.rst"
    content = index_file.read_text()
    lines = content.split("\n")

    # Locate the toctree directive and find where the entries begin
    entry_start = None
    for i, line in enumerate(lines):
        if entry_start is None and ".. toctree::" in line:
            entry_start = i + 1
        if entry_start is not None and ":maxdepth:" in line:
            entry_start = i + 1
            break

    assert entry_start is not None, "Could not find toctree in index.rst"

    # Collect the existing version entries (numeric only) up to changelog-dev
    entry_lines = []
    j = entry_start
    while j < len(lines):
        stripped = lines[j].strip()
        if not stripped:
            j += 1
            continue
        if stripped == "changelog-dev":
            break
        if re.match(r"^[\d.]+$", stripped):
            entry_lines.append((stripped, j))
        j += 1

    changelog_dev_line = j  # line index of "changelog-dev"

    # Walk the list from newest → oldest and insert before the first
    # entry that is *older* than the new version.  This preserves the
    # descending sort without requiring us to sort the whole list.
    # Convert dotted strings to tuples so we can compare numerically
    # (e.g. "0.10" > "0.9" because (0, 10) > (0, 9)).
    new_ver = tuple(int(x) for x in version.split("."))
    inserted = False
    for entry_ver_str, line_idx in entry_lines:
        if entry_ver_str == version:
            print(f"  Version {version} already in index.rst, skipping")
            return True
        if tuple(int(x) for x in entry_ver_str.split(".")) < new_ver:
            lines.insert(line_idx, f"   {version}")
            inserted = True
            break

    # Fallback: new version is the oldest → insert right before dev
    if not inserted:
        lines.insert(changelog_dev_line, f"   {version}")

    index_file.write_text("\n".join(lines))
    print(f"  Updated index.rst (added {version})")
    return True


def create_new_dev_changelog(version):
    """Step 3: Scaffold a blank changelog-dev.rst for the next release.

    The new file uses the minor-bumped version (e.g. 0.9 after a 0.8.2
    release) and mirrors the section headers from changelog-dev.rst.
    Contributors fill in entries under the relevant headings during the
    development cycle.
    """
    # Bump to the next minor version for the new dev cycle.
    # Dropping the patch segment keeps labels clean (e.g. 0.8.2 → 0.9).
    major, minor, _ = version.split(".")
    next_label = f"{major}.{int(minor) + 1}"

    # The RST underline must match the title width.
    # Title is "Qrisp <label>", underline is '=' repeated.
    title = f"Qrisp {next_label}"
    underline = "=" * len(title)
    label = f".. _v{next_label}:"

    content = f"""\
{label}

{title}
{underline}

{title} continues to push the boundaries of high-level quantum programming.

.. Add introductory paragraph above this line

Other New Features
------------------

.. Add other new features above this line

Compatibility
-------------

.. Add compatibility notes above this line

New Tutorials
-------------

.. Add new tutorials above this line

UI Changes
----------

.. Add UI changes above this line

First Time Contributors 🎉
--------------------------

.. Add new contributors above this line
"""

    (CHANGELOG_DIR / "changelog-dev.rst").write_text(content)
    print(f"  Created new changelog-dev.rst for v{next_label}")
    return True


def main():
    """Orchestrate the three steps in order.

    Each step is a function that accepts the version string and returns
    True on success.  A failure in any step aborts the whole process so
    the repo isn't left in a half-applied state.
    """
    # Read the version that was set in pyproject.toml before running this script
    with open("pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]
    print(f"Current version: {version}\n")

    for step in [rename_dev_changelog, update_index_rst, create_new_dev_changelog]:
        if not step(version):
            return 1

    print("\nDone. Review with `git diff` and commit when ready.")
    return 0


if __name__ == "__main__":
    exit(main())
