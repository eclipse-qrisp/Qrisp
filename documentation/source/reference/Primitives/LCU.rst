.. _LCU:

Linear Combination of Unitaries (LCU)
=====================================

The Linear Combination of Unitaries (LCU) algorithm is a foundational quantum algorithmic primitive that enables the implementation of non-unitary operators by expressing them as a weighted sum of unitaries. This approach is central to quantum algorithms for Hamiltonian simulation, Quantum Linear Systems (e.g., HHL algorithm), Quantum Signal Processing (QSP), and Quantum Singular Value Transformation (QSVT).

This module provides the following implementations:

- **inner_LCU**: The core implementation of the LCU protocol without the Repeat-Until-Success (RUS) protocol.
- **LCU**: The full implementation of the LCU algorithm using the RUS protocol.
- **view_LCU**: A utility function to generate and return the quantum circuit for visualization.

.. currentmodule:: qrisp

Contents
--------

.. autosummary::
   :toctree: generated/

   inner_LCU
   LCU
   view_LCU

Function Details
----------------

.. autofunction:: inner_LCU

.. autofunction:: LCU

.. autofunction:: view_LCU