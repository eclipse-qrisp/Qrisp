.. _prefix_arithmetic:

Prefix Arithmetic
=================

While syntactically more compact/elegant, infix arithmetic expressions allow barely any customization regarding precision, method etc. which can be detrimental when efficiency is a concern. Because of this we expose the underlying prefix arithmetic functions:

.. currentmodule:: qrisp

In-place adders
---------------

.. autosummary::
   :toctree: generated/
   
   fourier_adder
   cuccaro_adder
   gidney_adder
   qcla

QuantumFloat processing
-----------------------

.. autosummary::
   :toctree: generated/

   sbp_mult
   sbp_add
   sbp_sub
   hybrid_mult
   inpl_add
   inpl_mult
   q_divmod
   q_div
   qf_inversion
   
QuantumArray processing
-----------------------

.. autosummary::
   :toctree: generated/

   q_matmul
   semi_classic_matmul
   inplace_matrix_app
   dot
   tensordot