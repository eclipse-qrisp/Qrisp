.. _LCU:

Linear Combination of Unitaries (LCU)
=====================================

The Linear Combination of Unitaries (LCU) algorithm is a foundational quantum algorithmic primitive that enables the implementation of non-unitary operators by expressing them as a weighted sum of unitaries. This approach is central to quantum algorithms for `Hamiltonian simulation <https://www.taylorfrancis.com/chapters/edit/10.1201/9780429500459-11/simulating-physics-computers-richard-feynman>`_, `Linear Combination of Hamiltonian Simulation (LCHS) <https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.131.150603>`_, Quantum Linear Systems (e.g. `HHL algorithm <https://pennylane.ai/qml/demos/linear_equations_hhl_qrisp_catalyst>`_), `Quantum Signal Processing (QSP) <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020368>`_, and `Quantum Singular Value Transformation (QSVT) <https://dl.acm.org/doi/abs/10.1145/3313276.3316366>`_.

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
