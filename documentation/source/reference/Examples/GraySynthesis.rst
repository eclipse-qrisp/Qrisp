.. _GraySynthesis:

Gray Synthesis
==============

Qrisp's logic synthesis module converts truth tables into quantum circuits using Gray code synthesis.
The user interface is through :class:`TruthTable <qrisp.alg_primitives.logic_synthesis.TruthTable>` and :meth:`gray_logic_synth <qrisp.alg_primitives.logic_synthesis.gray_logic_synth>`.

Example
-------

Create a truth table from a list of output bitstrings and synthesize a circuit for each input:

>>> from qrisp import h
>>> from qrisp.alg_primitives.logic_synthesis import TruthTable, gray_logic_synth
>>> from qrisp.core import QuantumVariable
>>> from qrisp.misc import int_encoder
>>> tt = TruthTable(["00010101", "01001101", "01011100"])

For each input index, create input and output variables, encode the index, and synthesize:

>>> for index in range(tt.shape[0]):
...     input_var = QuantumVariable(tt.bit_amount)
...     int_encoder(input_var, index)
...     output_var = QuantumVariable(tt.shape[1])
...     gray_logic_synth(input_var, output_var, tt, phase_tolerant = False)
...     meas = list(output_var.get_measurement().keys())[0]
...     print("Index", index, ": measured", meas)
Index 0 : measured 000
Index 1 : measured 011
Index 2 : measured 000
Index 3 : measured 101
Index 4 : measured 011
Index 5 : measured 111
Index 6 : measured 000
Index 7 : measured 110

With a superposition input, the synthesis produces a superposition of all outputs:

>>> input_var = QuantumVariable(tt.bit_amount)
>>> h(input_var)
>>> output_var = QuantumVariable(tt.shape[1])
>>> gray_logic_synth(input_var, output_var, tt, phase_tolerant = False)
>>> print(input_var.qs.compile().depth())
23
