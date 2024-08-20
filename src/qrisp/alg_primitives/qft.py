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

import numpy as np

from qrisp.core import p, h, cp, cx, x, s, swap

def QFT_inner(qv, exec_swap=True, qiskit_endian=True, inplace_mult=1, use_gms=False, inpl_adder = None):
    from qrisp.misc import is_inv

    qv = list(qv)
    n = len(qv)

    if qiskit_endian:
        qv = qv[::-1]

    if not use_gms:
        from qrisp.environments.quantum_environments import QuantumEnvironment

        env = QuantumEnvironment

    else:
        from qrisp.environments.GMS_environment import GMSEnvironment

        env = GMSEnvironment

    if not is_inv(inplace_mult, n):
        raise Exception(
            "Tried to perform non-invertible inplace multiplication"
            "during Fourier-Transform"
        )


    if inpl_adder is None:
        accumulated_phases = np.zeros(n)
        for i in range(n):
            if accumulated_phases[i] and not use_gms:
                p(accumulated_phases[i], qv[i])
                accumulated_phases[i] = 0
            
            h(qv[i])
    
            if i == n - 1:
                break
    
            with env():
                for k in range(n - i - 1):
                    # cp(inplace_mult * 2 * np.pi / 2 ** (k + 2), qv[k + i + 1], qv[i])
                    
                    if use_gms:
                        cp(inplace_mult * 2 * np.pi / 2 ** (k + 2), qv[i], qv[k + i + 1])
                    else:
                        phase = inplace_mult * 2 * np.pi / 2 ** (k + 2)
                        
                        # cx(qv[k + i + 1], qv[i])
                        # p(-phase/2, qv[i])
                        # cx(qv[k + i + 1], qv[i])
                        
                        
                        cx(qv[i], qv[k + i + 1])
                        p(-phase/2, qv[k + i + 1])
                        cx(qv[i], qv[k + i + 1])
                        
                        
                        accumulated_phases[i] += phase/2
                        accumulated_phases[k + i + 1] += phase/2
        
        
                    
        for i in range(n):
            if accumulated_phases[i] and not use_gms:
                p(accumulated_phases[i], qv[i])
                accumulated_phases[i] = 0
                
    else:
        
        from qrisp import QuantumFloat, conjugate
        reservoir = QuantumFloat(n+1)
        
        def prepare_reservoir(reservoir):
            n = len(reservoir)
            h(reservoir)
            for i in range(n):
                p(np.pi*2**(i-n+1), reservoir[i])
        
        
        with conjugate(prepare_reservoir)(reservoir):

            for i in range(n):
                
                h(qv[i])
        
                if i == n - 1:
                    break
        
                phase_qubits = []
                for k in range(n - i - 1):
                    cx(qv[i], qv[k + i + 1])
                    phase_qubits.append(qv[k + i + 1])
                
                inpl_adder(phase_qubits[::-1], reservoir[-len(phase_qubits)-2:])
                    
                for k in range(n - i - 1):
                    cx(qv[i], qv[k + i + 1])
                
                x(reservoir)
                inpl_adder(phase_qubits[::-1], reservoir[-len(phase_qubits)-2:])
                x(reservoir)
            
            s(qv)
            inpl_adder(qv, reservoir[-n-1:])
        
        reservoir.delete()

    if exec_swap:
        for i in range(n // 2):
            swap(qv[i], qv[n - i - 1])

    return qv



def QFT(
    qv, inv=False, exec_swap=True, qiskit_endian=True, inplace_mult=1, use_gms=False, inpl_adder=None
):
    """
    Performs the quantum fourier transform on the input.

    Parameters
    ----------
    qv : QuantumVariable
        QuantumVariable to transform (in-place).
    inv : bool, optional
        If set to True, the inverse transform will be applied. The default is False.
    exec_swap : bool, optional
        If set to False, the swaps at the end of the transformation will be skipped.
        The default is True.
    qiskit_endian : bool, optional
        If set to False the order of bits will be reversed. The default is True.
    inplace_mult : int, optional
        Allows multiplying the QuantumVariable with an extra factor during the
        transformation. For more information check `the publication
        <https://ieeexplore.ieee.org/document/9815035>`_. The default is 1.
    use_gms : bool, optional
        If set to True, the QFT will be compiled using only GMS gates as entangling
        gates. The default is False.
    inpl_adder : callable, optional
        Uses an adder and a reservoir state to perform the QFT. Read more about 
        it :ref:`here <adder_based_qft>`. The default is None


    """
    from qrisp import gate_wrap, invert

    name = "QFT"
    if not exec_swap:
        name += " no swap"
    if inplace_mult != 1:
        name += " inpl mult " + str(inplace_mult)
    if inpl_adder is not None:
        name += "_adder_supported"

    if inv:
        with invert():
            gate_wrap(permeability=[], is_qfree=False, name=name)(QFT_inner)(
                qv,
                exec_swap=exec_swap,
                qiskit_endian=qiskit_endian,
                inplace_mult=inplace_mult,
                use_gms=use_gms,
                inpl_adder=inpl_adder
            )
    else:
        gate_wrap(permeability=[], is_qfree=False, name=name)(QFT_inner)(
            qv,
            exec_swap=exec_swap,
            qiskit_endian=qiskit_endian,
            inplace_mult=inplace_mult,
            use_gms=use_gms,
            inpl_adder=inpl_adder
        )

    return qv