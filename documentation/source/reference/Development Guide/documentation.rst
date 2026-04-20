.. _DevGuideDocumentation:

Documentation
=============

Qrisp's documentation is built with `Sphinx <https://www.sphinx-doc.org/>`_
and hosted at `qrisp.eu <https://www.qrisp.eu>`_. Contributions that add or
modify public functionality should include corresponding documentation updates.

Building the docs locally
--------------------------

All documentation source files live under ``documentation/source/``. To build
and preview them:

.. code-block:: bash

    # From the repository root
    cd documentation

    # One-shot build — output lands in documentation/build/html/
    make html

Open ``documentation/build/html/index.html`` in your browser to inspect the
result. Always verify that pages render correctly before submitting a pull
request — formatting that looks fine in raw reStructuredText may not render
well in Sphinx.

Docstrings
----------

Public functions and classes should have docstrings that use section headers
separated by dashes (``Parameters``, ``Returns``, ``Examples``, etc.),
following the `NumPy docstring style <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
This is the pattern already used in the existing codebase.

Checklist for documentation contributions:

- All docstring ``Examples`` blocks execute correctly.
- New pages are registered in the appropriate ``index.rst`` ``toctree``.
- Cross-references (e.g., ``:ref:`QuantumFloat```) resolve without warnings.
- The build produces zero new Sphinx warnings.

Useful references
-----------------

- Sphinx documentation: https://www.sphinx-doc.org/
- NumPy docstring style: https://numpydoc.readthedocs.io/en/latest/format.html
- Google docstring style: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
- reStructuredText primer: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
