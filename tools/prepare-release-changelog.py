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

import shutil
from pathlib import Path

# toml file parser
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# Path to the changelog RST files, relative to the repo root.
# All three steps read or write files under this directory.
CHANGELOG_DIR = Path("documentation/source/general/changelog")


def rename_dev_changelog(version):
    """Step 1: Archive the current dev changelog under the release version.

    The file changelog-dev.rst becomes <version>.rst so the entry
    survives in the versioned toctree alongside 0.1.rst, 0.2.rst, etc.
    """
    dev = CHANGELOG_DIR / "changelog-dev.rst"
    release = CHANGELOG_DIR / f"{version}.rst"

    # Ensure the source dev changelog exists before moving.
    if not dev.exists():
        print(f"ERROR: {dev} not found")
        return False

    # Guard against overwriting a previous release's file.  The user should
    # restore changelog-dev.rst from git before re-running after a failure.
    if release.exists():
        print(f"ERROR: {release} already exists (restore from git before re-running)")
        return False

    # Rename changelog-dev.rst -> <version>.rst within the changelog directory.
    shutil.move(str(dev), str(release))
    print(f"  Renamed: changelog-dev.rst -> {version}.rst")
    return True


def update_index_rst(version):
    """Step 2: Register the new release in the toctree of index.rst.

    The toctree lists versions in descending order (newest first), with
    changelog-dev.rst always at the end.  This function inserts the new
    version at the correct sorted position without rewriting entries that
    are already there.
    """
    index_file = CHANGELOG_DIR / "index.rst"
    lines = index_file.read_text().split("\n")

    # Locate the first line containing ":maxdepth:" — entries start on the
    # line immediately after this option directive.
    for i, line in enumerate(lines):
        if ":maxdepth:" in line:
            start = i + 1
            break
    else:
        print("ERROR: could not find ':maxdepth:' in index.rst")
        return False

    # Walk forward from `start` to find the "changelog-dev" line.  This sets
    # the exclusive upper bound of the toctree entry section.
    for end in range(start, len(lines)):
        if lines[end].strip() == "changelog-dev":
            break
    else:
        print("ERROR: could not find 'changelog-dev' entry")
        return False

    # Collect non-blank lines in the toctree block together with their real
    # line indices.  We keep both because blank lines (e.g. the mandatory
    # blank after the :maxdepth: option) are skipped when building the
    # `entries` list, but the raw positions in `entry_positions` are needed
    # later to insert new lines at the correct spot in `lines`.
    entries = []
    entry_positions = []
    for i in range(start, end):
        if lines[i].strip():
            entries.append(lines[i].strip())
            entry_positions.append(i)

    # Bail early if this version was already added (e.g. on a re-run).
    if version in entries:
        print(f"  Version {version} already in index.rst, skipping")
        return True

    # Parse the incoming version as a numeric tuple for comparison.
    vtuple = tuple(int(x) for x in version.split("."))

    # Scan entries in order (already descending) to find the first entry
    # that is *smaller* than the new version — that's where we insert.
    for idx, entry in enumerate(entries):
        # Safely parse the toctree entry.  Non-numeric suffixes (e.g.
        # "0.9.0rc1") produce an empty tuple and sort below genuine
        # dotted-integer versions.
        try:
            entry_tuple = tuple(int(x) for x in entry.split("."))
        except ValueError:
            entry_tuple = ()

        if vtuple > entry_tuple:
            lines.insert(entry_positions[idx], f"   {version}")
            break
    else:
        # New version is the smallest seen — insert right before
        # changelog-dev.
        lines.insert(end, f"   {version}")

    # Persist the modified lines back to disk.
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
    parts = version.split(".")
    major, minor = parts[0], parts[1]
    next_label = f"{major}.{int(minor) + 1}"

    # RST rules: the underline characters must repeat the title length.
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

Bug Fixes
---------

.. Add bug fixes above this line

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
    # Confirm we're running from the repo root (CHANGELOG_DIR is relative).
    if not CHANGELOG_DIR.exists():
        print("ERROR: run this script from the repository root")
        return 1

    # Read the version that was set in pyproject.toml before running this script.
    # tomllib.load returns a dict; we reach into [project] for the version field.
    with open("pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]
    print(f"Current version: {version}\n")

    # Steps are executed in order.  Each returns False on failure; the
    # short-circuit ensures we don't proceed with a half-applied state.
    for step in [rename_dev_changelog, update_index_rst, create_new_dev_changelog]:
        if not step(version):
            return 1

    print("\nDone. Review with `git diff` and commit when ready.")
    return 0


if __name__ == "__main__":
    exit(main())
