"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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
********************************************************************************
"""

import itertools

import numpy as np

from qrisp.core import QuantumVariable, recursive_qv_search
from qrisp.misc import bin_rep, custom_qv, int_as_array


class QuantumDictionary(dict):
    r"""
    This class can be used for loading data relations into the quantum computer which
    are not based on a quantum algorithm.

    As an inheritor of the Python dictionary it has all the functionality we are used to

    >>> from qrisp import QuantumDictionary, QuantumVariable, multi_measurement
    >>> qd = QuantumDictionary()
    >>> qd[1] = 2
    >>> qd[42] = (3,4)
    >>> qd["hello"] = "hallo"
    >>> qd["world"] = "welt"
    >>> print(qd[42])
    (3,4)

    The key difference is that the QuantumDictionary can also recieve
    :ref:`QuantumVariables <QuantumVariable>` as keys and return the corresponding
    values as an entangled QuantumVariable.

    We demonstrate this by preparing a QuantumVariable which has the keys of ``qd`` as
    outcome labels:

    >>> key_qv = QuantumVariable.custom([1, 42, "hello", "world"])
    >>> key_qv[:] = {"hello" : 1, "world" : 1}
    >>> print(key_qv)
    {'hello': 0.5, 'world': 0.5}

    We now load the data from the QuantumDictionary into the quantum algorithm by
    dereferencing ``qd`` with ``key_qv``:

    >>> value_qv = qd[key_qv]
    >>> multi_measurement([key_qv, value_qv])
    {('hello', 'hallo'): 0.5, ('world', 'welt'): 0.5}

    We see that the states of ``key_qv`` are now entangled with the states of the
    values.

    QuantumDictionaries can also load from tuples of QuantumVariables:

    >>> qd[(1,2)] = 73
    >>> qd[(0,2)] = 37
    >>> qf1 = QuantumFloat(1)
    >>> qf2 = QuantumFloat(2)
    >>> h(qf1)
    >>> qf2[:] = 2
    >>> res = qd[(qf1, qf2)]
    >>> multi_measurement([qf1, qf2, res])
    {(0, 2, 37): 0.5, (1, 2, 73): 0.5}

    Mathematically we have

    .. math::

        U_{\text{qd}} \left( \sum_{x \in \text{labels}} a_x \ket{x} \right)
        \ket{0} = \sum_{x \in \text{labels}} a_x \ket{x} \ket{\text{qd}[x]}

    Note that this quantum operation is realized through quantum logic synthesis, which
    scales rather badly compared to algorithmic generation of data relations.
    Therefore, performing as much logic on the quantum computer is preferable over
    performing the logic on the classical computer and inserting the results using
    QuantumDictionaries.


    **Specifying the return type**

    The returned QuantumVariable ``value_qv`` is (similar to ``key_qv``) a
    CustomQuantumVariable:

    >>> print(type(value_qv))
    <class 'qrisp.misc.misc_functions.custom_qv.<locals>.CustomQuantumVariable'>

    If we want to apply further processing this might not be helpfull since custom
    QuantumVariables lack many methods that are available in more specific quantum
    types. In this case we can supply the QuantumDictionary with a return type:

    >>> from qrisp import QuantumFloat
    >>> qtype = QuantumFloat(4, -2, signed = True)
    >>> float_qd = QuantumDictionary(return_type = qtype)

    We fill again with some example values

    >>> float_qd["hello"] = 0.5
    >>> float_qd["world"] = -1

    And retrieve the value:

    >>> value_qv = float_qd[key_qv]
    >>> print(type(value_qv))
    <class 'qrisp.qtypes.quantum_float.QuantumFloat'>

    Since ``value_qv`` is a QuantumFloat now, we can use the established methods for
    arithmetic - for instance the inplace addition:

    >>> value_qv += 1.5
    >>> print(multi_measurement([key_qv, value_qv]))
    {('hello', 2.0): 0.5, ('world', 0.5): 0.5}


    **Advanced usage**

    In some cases, (such as manual uncomputation) it is required to specify into which
    variable the QuantumDictionary should load. We do this with the
    :meth:`load <qrisp.QuantumDictionary.load>` method:

    >>> qf = qtype.duplicate()
    >>> float_qd.load(key_qv, qf)
    >>> print(qf)
    {0.5: 0.5, -1.0: 0.5}

    The ``load`` method furthermore allows to specify which logic synthesis algorithm
    should be used.

    >>> qf2 = qtype.duplicate()
    >>> float_qd.load(key_qv, qf2, synth_method = "gray")

    """

    def __init__(self, init_dict={}, return_type=None):
        super().__init__(init_dict)

        self.return_type = return_type

    def __getitem__(self, key):
        return self.load(key)

    def load(self, key, value_qv=None, synth_method="gray"):
        """
        Loads the values of the QuantumDictionary into a given QuantumVariable.

        Parameters
        ----------
        key_qv : QuantumVariable
            A QuantumVariable with a decoder supporting the keys of this
            QuantumDictionary.
        value_qv : QuantumVariable, optional
            The QuantumVariable to load the values into. If given None, a new
            QuantumVariable is generated.
        synth_method : string, optional
            The method of logic synthesis to use for loading. Currently available are
            "gray", "gray_pt", "pprm" and "pprm_pt". The default is "gray".

        Raises
        ------
        Exception
            Tried to load value from empty dictionary.

        Examples
        --------

        We create a QuantumDictionary with return type QuantumFloat

        >>> from qrisp import QuantumDictionary, QuantumFloat, h
        >>> qtype = QuantumFloat(4, -2)
        >>> float_qd = QuantumDictionary(return_type = qtype)
        >>> float_qd[0] = 1
        >>> float_qd[1] = 2

        Create the key and the value variable:

        >>> key_qv = QuantumFloat(1, signed = False)
        >>> h(key_qv)
        >>> value_qv = qtype.duplicate()

        And load the values

        >>> float_qd.load(key_qv, value_qv, synth_method = "pprm")
        >>> print(value_qv)
        {1.0: 0.5, 2.0: 0.5}

        """
        from qrisp.alg_primitives.logic_synthesis import TruthTable

        qv_list = recursive_qv_search(key)
        if not len(qv_list):
            return dict.__getitem__(self, key)

        if not isinstance(key, tuple):
            key = (key,)

        labels = []

        for qv in key:
            if not isinstance(qv, QuantumVariable):
                raise Exception(
                    "Tried to deref QuantumDictionary with mixed"
                    "(classical + quantum) input."
                )

            labels.append([qv.decoder(i) for i in range(2**qv.size)])

        quantum_key = key

        if len(labels) == 1:
            constellations = labels[0]
        else:
            constellations = itertools.product(*labels)

        relevant_dic = {}

        for const in constellations:
            if const in self:
                relevant_dic[const] = dict.__getitem__(self, const)

        if value_qv is None:
            if isinstance(self.return_type, type(None)):
                value_qv = custom_qv(list(relevant_dic.values()))
            else:
                value_qv = self.return_type.duplicate()

        n = sum([qv.size for qv in key])

        tt_array = np.zeros((2**n, value_qv.size))

        for k in relevant_dic:
            bin_string = ""

            for i in range(len(quantum_key)):
                qv = quantum_key[i]

                if isinstance(k, tuple):
                    bin_string += bin_rep(qv.encoder(k[i]), qv.size)[::-1]
                else:
                    bin_string += bin_rep(qv.encoder(k), qv.size)[::-1]

            row_number = int(bin_string[::-1], 2)
            tt_array[row_number, :] = int_as_array(
                value_qv.encoder(relevant_dic[k]), value_qv.size
            )[::-1]

        tt = TruthTable(tt_array)

        qv_temp_0 = QuantumVariable(n)
        qv_temp_1 = QuantumVariable(value_qv.size, qs=qv_temp_0.qs)

        tt.q_synth(qv_temp_0, qv_temp_1, method=synth_method)

        if len(qv_temp_0.qs.qubits) != qv_temp_0.size + qv_temp_1.size:
            synth_ancilla = QuantumVariable(
                len(qv_temp_0.qs.qubits) - qv_temp_0.size + qv_temp_1.size
            )
            quantum_key.append(synth_ancilla)

        res_gate = qv_temp_0.qs.data[-1].op
        # res_gate = qv_temp_0.qs.to_gate("q_load")
        res_gate.is_qfree = True
        res_gate.permeability = {i: i < n for i in range(res_gate.num_qubits)}

        quantum_key[0].qs.append(
            res_gate, sum([qv.reg for qv in quantum_key], []) + value_qv.reg
        )

        if len(qv_temp_0.qs.qubits) != qv_temp_0.size + qv_temp_1.size:
            synth_ancilla.delete()

        return value_qv
