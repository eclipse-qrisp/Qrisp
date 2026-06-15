.. _DevGuideReleasing:

Making a Qrisp Release
======================

This page documents the step-by-step process for cutting a new Qrisp release,
from bumping the version number to publishing on PyPI and creating a GitHub
Release.

Prerequisites
-------------

Before you can publish a release you need:

* **PyPI credentials** with upload permissions for the ``qrisp`` package.
  `PyPI account <https://pypi.org/account/register/>`_ and
  `token-based authentication <https://pypi.org/help/#apitoken>`_ are
  recommended.
* **``twine``** installed — ``pip install twine``
* **Write access** to the repository (for pushing tags).

Step-by-step instructions
-------------------------

#. **Ensure the working tree is clean.**

   Check that there are no uncommitted changes and that you are on ``main``
   with the latest upstream commits:

   .. code-block:: bash

       git checkout main
       git pull --ff-only
       git status

#. **Update the version number.**

   The canonical version is defined in :file:`setup.cfg` under ``[metadata]``:

   .. code-block:: ini

       [metadata]
       name = qrisp
       version = 0.8.2   # ← bump this

   Increment according to `Semantic Versioning <https://semver.org/>`_:

   * **Patch** (0.8.2 → 0.8.3) — bug fixes, minor improvements, no API changes.
   * **Minor** (0.8.2 → 0.9.0) — new features, deprecations, backward-compatible API additions.
   * **Major** (0.8.2 → 1.0.0) — breaking changes to the public API.

#. **Update CHANGELOG.md.**

   Ensure :file:`CHANGELOG.md` reflects all notable changes since the last
   release.  If the :doc:`changelog workflow <index:workflows/changelog.yml>`
   is active, it will have kept the ``Unreleased`` section up-to-date on every
   push to ``main``.

   Move the ``Unreleased`` entries under a new heading for the version being
   released (e.g. ``## [v0.8.3] - 2026-06-15``) and commit the result.

#. **Commit the version bump and changelog update.**

   .. code-block:: bash

       git add setup.cfg CHANGELOG.md
       git commit -m "chore: bump version to 0.8.3"

#. **Create a signed tag.**

   Tag the commit with the version string (``v`` prefix).  Annotated tags are
   preferred because they carry a message:

   .. code-block:: bash

       git tag -a v0.8.3 -m "v0.8.3"
       git push origin v0.8.3

   Pushing the tag triggers the :doc:`changelog workflow <index:workflows/changelog.yml>`,
   which automatically creates a GitHub Release with notes extracted from
   :file:`CHANGELOG.md`.

#. **Build the distribution packages.**

   .. code-block:: bash

       rm -rf dist/          # clean previous builds
       python -m build       # produces dist/qrisp-0.8.3.tar.gz and dist/qrisp-0.8.3-py3-none-any.whl

   :command:`python -m build` uses the configuration in :file:`pyproject.toml`
   (``setuptools.build_meta``) and :file:`setup.py`.

#. **Upload to PyPI.**

   Use ``twine`` to publish the packages.  Token-based auth is strongly
   recommended over username + password:

   .. code-block:: bash

       twine upload dist/*

   If you are publishing a **pre-release** (e.g. ``v0.9.0-alpha.1``), upload
   to `Test PyPI <https://test.pypi.org/>`_ first to verify:

   .. code-block:: bash

       twine upload --repository testpypi dist/*

#. **Verify the release.**

   * On `PyPI <https://pypi.org/project/qrisp/>`_ — check that the new
     version appears and the description renders correctly.
   * On `GitHub Releases <https://github.com/eclipse-qrisp/Qrisp/releases>`_ —
     confirm the release has the correct notes, is marked as the latest, and
     (if applicable) as a pre-release.
   * Install from PyPI in a fresh environment:

     .. code-block:: bash

         pip install qrisp==0.8.3
         python -c "import qrisp; print(qrisp.__version__)"

#. **Announce.**

   Post about the release on:

   * Our `Discord server <https://discord.gg/v5np7DeBaq>`_ (``#releases``
     channel).
   * Relevant social media / mailing lists.

Automated workflows reference
-----------------------------

This release process is supported by the following CI workflows:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Workflow
     - What it does
   * - :file:`.github/workflows/changelog.yml`
     - On push to ``main``, rebuilds :file:`CHANGELOG.md` from git history.
       On tag push (``v*``), creates a GitHub Release with auto-generated
       notes. See the inline comments in the file for a detailed walkthrough.
   * - :file:`.github/workflows/qrisp_test.yml`
     - Runs ``pytest`` on every push and pull request to ``main``. Releases
       should never be cut from a branch that fails tests.
