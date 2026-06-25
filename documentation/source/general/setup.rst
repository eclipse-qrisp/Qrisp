.. _setup:

Setup
=====

Qrisp is written in pure Python implying it can be installed conveniently with `PyPi <https://pypi.org/>`_. Currently Python version 3.11 - 3.13 have been confirmed to work with Qrisp.

Simply execute:

::

   pip install qrisp

Additional backends and features require extra dependencies:

::

   pip install qrisp[aqt]     # AQT quantum hardware
   pip install qrisp[cirq]    # Cirq simulator
   pip install qrisp[qiskit]  # Qiskit Aer simulator + IBM Quantum Runtime
   pip install qrisp[iqm]     # IQM quantum hardware
   pip install qrisp[catalyst] # Catalyst JIT compilation
   pip install qrisp[xdsl]    # xDSL IR framework


Visit our :ref:`tutorial` page to get started with building your first algorithms!