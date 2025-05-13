.. _LCU:

Linear Combination of Unitaries (LCU)
=====================================

The LCU method is a foundational quantum algorithmic
primitive that enables the application of a non-unitary operator $A$, expressed as a weighted
sum of unitaries $U_i$ as $A=\sum_i\alpha_i U_i$, to a quantum state, by embedding $A$ into a larger unitary circuit. 

This is
central to quantum algorithms for `Hamiltonian simulation <https://www.taylorfrancis.com/chapters/edit/10.1201/9780429500459-11/simulating-physics-computers-richard-feynman>`_, `Linear Combination of Hamiltonian Simulation (LCHS) <https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.131.150603>`_, Quantum Linear Systems (e.g. `HHL algorithm <https://pennylane.ai/qml/demos/linear_equations_hhl_qrisp_catalyst>`_), `Quantum Signal
Processing (QSP) <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020368>`_, and `Quantum Singular Value Transformation (QSVT) <https://dl.acm.org/doi/abs/10.1145/3313276.3316366>`_.

This function implements the prepare-select-unprepare structure, also known as block encoding:

.. math::
   \mathrm{LCU} = \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE}

- **PREPARE**: Prepares an ancilla variable in a superposition encoding the normalized coefficients $\alpha_i$ of the target operator

.. math::
   \mathrm{PREPARE}|0\rangle=\sum_i\sqrt{\frac{\alpha_i}{\lambda}}|i\rangle

- **SELECT**: Applies the unitary $U_i$ to the target variable, controlled on the ancilla variable being in state $|i\rangle$. 

.. math::
   \mathrm{SELECT}|i\rangle|\psi\rangle=|i\rangle U_i|\psi\rangle

- **PREPARE$^{\\dagger}$**: Applies the inverse prepartion to the ancilla.

.. note::

   The LCU protocol is deemed successful only if the ancilla variable is measured in the :math:`|0\rangle` state, which occurs with a probability proportional to :math:`\frac{\langle\psi|A^{\dagger}A|\psi\rangle}{\lambda^2}` where $\lambda=\sum_i\alpha_i$.  
   This function does not perform the measurement; it returns the ancilla variable and the transformed target variable.

The success probability depends on the LCU coefficients and the initial state's properties. Said success probability can be further improved using `Oblibious Amplitude Amplification <https://arxiv.org/pdf/1312.1414>`_, which applies a series of reflections and controlled operations to amplify the $\ket{0}$ ancilla component without prior knowledge of the initial state. For implementation details in Qrisp, see :func:`qrisp.amplitude_amplification`

For a complete implementation of LCU with the Repeat-Until-Success protocol, see :func:`qrisp.LCU`.

For more details on the LCU protocol, refer to `Childs and Wiebe (2012) <https://arxiv.org/abs/1202.5822>`_, or `related seminars provided by Nathan Wiebe <https://www.youtube.com/watch?v=irMKrOIrHP4>`_. 

.. caution::
   Nathan's a cool guy, but **do not** call the SELECT unitary a *qswitch* in his presence, or else he might perform the Touch of Death (in reference to the movie The Men Who Stare at Goats). While on the topic, refer to :func:`qrisp.qswitch` for more information about how the SELECT operator is implemented efficiently in Qrisp (as :func:`qrisp.qswitch`).

.. currentmodule:: qrisp

Contents
--------

.. autosummary::
   :toctree: generated/

   inner_LCU
   LCU
   view_LCU
