"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************/
"""


from qrisp.environments.quantum_environments import QuantumEnvironment
from qrisp.misc.GMS_tools import GXX_converter


# Environments that allows quick and easy access to the GMS_converter
# To use it make sure every gate that happens inside is either phase or cphase
class GMSEnvironment(QuantumEnvironment):
    """
    This environment provides a convenient interface for constructing quantum algorithms
    using the Ion-trap native GMS gates. GMS gates allow entangling more than 2 qubits
    in a single step and can therefore boost performance in many situations. For more
    information on GMS gates consult https://arxiv.org/abs/quant-ph/9810040 . The
    techniques for converting the circuits presented to this environment are mostly
    based on https://ieeexplore.ieee.org/document/9815035 .

    This environment allows to code blocks of phase-only gates as we are used to but
    compiles these blocks to GMS gates.

    Examples
    --------


    We create a function performing the quantum Fourier-transform using GMS gates: ::

        from qrisp import QuantumEnvironment, GMSEnvironment, h, cp, swap
        import numpy as np

        def QFT(qv, use_gms = False):

            n = qv.size

            if use_gms:
                env = GMSEnvironment
            else:
                env = QuantumEnvironment


            for i in range(n):
                h(qv[i])

                if i == n-1:
                    break


                with env():

                    #This is the block which converted to GMS gates
                    #We can only use the gates p, cp and rz in here

                    for k in range(n-i-1):
                        cp(2*np.pi/2**(k+2), qv[k+i+1], qv[i])

            for i in range(n//2):
                swap(qv[i], qv[n-i-1])


    We inspect the resulting quantum circuit:

    >>> from qrisp import QuantumFloat, invert
    >>> qf = QuantumFloat(5)
    >>> qf[:] = 13
    >>> QFT(qf, use_gms = True)
    >>> print(qf.qs)
    QuantumCircuit:
    ---------------
          ┌───┐┌───┐┌─────────────────────┐                                 »
    qf.0: ┤ X ├┤ H ├┤0                    ├─────────────────────────────────»
          └───┘└───┘│                     │┌───┐┌─────────────────────┐     »
    qf.1: ──────────┤1                    ├┤ H ├┤0                    ├─────»
          ┌───┐     │                     │└───┘│                     │┌───┐»
    qf.2: ┤ X ├─────┤2 GXX converted gate ├─────┤1                    ├┤ H ├»
          ├───┤     │                     │     │  GXX converted gate │└───┘»
    qf.3: ┤ X ├─────┤3                    ├─────┤2                    ├─────»
          └───┘     │                     │     │                     │     »
    qf.4: ──────────┤4                    ├─────┤3                    ├─────»
                    └─────────────────────┘     └─────────────────────┘     »
    «
    «qf.0: ─────────────────────────────────────────────────────────X─
    «                                                               │
    «qf.1: ─────────────────────────────────────────────────────X───┼─
    «      ┌─────────────────────┐                              │   │
    «qf.2: ┤0                    ├──────────────────────────────┼───┼─
    «      │                     │┌───┐┌─────────────────────┐  │   │
    «qf.3: ┤1 GXX converted gate ├┤ H ├┤0                    ├──X───┼─
    «      │                     │└───┘│  GXX converted gate │┌───┐ │
    «qf.4: ┤2                    ├─────┤1                    ├┤ H ├─X─
    «      └─────────────────────┘     └─────────────────────┘└───┘
    Live QuantumVariables:
    ----------------------
    QuantumFloat qf

    Now we check that the GMS version indeed performs the same operation as the CNOT
    version by performing the inverse of the CNOT version.

    >>> with invert(): QFT(qf, use_gms = False)
    >>> print(qf)
    {13: 1.0}

    """

    # We only need to modify the compile method of the base environment class
    def compile(self):
        # Temporarily store the data of the quantum session
        temp_data = list(self.env_qs.data)
        self.env_qs.clear_data()

        QuantumEnvironment.compile(self)

        copied_circ = self.env_qs.copy()
        depth_dic = self.env_qs.get_depth_dic()

        i = 0
        while i < len(copied_circ.qubits):
            if depth_dic[copied_circ.qubits[i]]:
                i += 1
            else:
                copied_circ.qubits.pop(i)

        if not len(copied_circ.qubits):
            return

        # Convert the circuit
        # converted_gate, qubit_map = GMS_converter(self.env_qs, True)
        converted_gate = GXX_converter(copied_circ).to_gate()

        converted_gate.name = "GXX converted gate"

        self.env_qs.clear_data()
        # Recover original circuit
        self.env_qs.data.extend(temp_data)

        # Apply original circuit
        self.env_qs.append(converted_gate, copied_circ.qubits)
