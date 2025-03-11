"""
\********************************************************************************
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
********************************************************************************/
"""

import copy
from itertools import product

import numpy as np
import jax.numpy as jnp
import jax


from qrisp.circuit import transpile
from qrisp.core import QuantumVariable, qompiler, QuantumSession, merge
from qrisp.misc import bin_rep
from qrisp.jasp import check_for_tracing_mode, q_fori_loop, jrange, create_qubits, DynamicQubitArray, TracingQuantumSession


class QuantumArray(np.ndarray):
    """
    This class allows the convenient management of multiple QuantumVariables of one
    type. As a subclass of `numpy's ndarray
    <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, the
    QuantumArray supports many convenient array manipulation methods. Similar to the
    numpy equivalent, creating a QuantumArray can be achieved by specifying a shape and
    a ``qtype``:

    >>> import numpy as np
    >>> from qrisp import QuantumArray, QuantumFloat
    >>> qtype = QuantumFloat(5, -2)
    >>> q_array = QuantumArray(qtype = qtype, shape = (2, 2, 2))

    Note that ``qtype`` is not a type object but a QuantumVariable which serves as an
    "example".

    To retrieve the entries (i.e. QuantumVariables) from the QuantumArray, we simply
    index as with regular numpy arrays:

    >>> from qrisp import h
    >>> qv = q_array[0,0,1]
    >>> h(qv[0])
    >>> print(q_array)
    {OutcomeArray([[[0., 0.],
                    [0., 0.]],
                   [[0., 0.],
                    [0., 0.]]]): 0.5,
     OutcomeArray([[[0.  , 0.25],
                    [0.  , 0.  ]],
                   [[0.  , 0.  ],
                    [0.  , 0.  ]]]): 0.5}

    We see the value 0.25 in the second entry because we applied an H-gate onto the 0-th
    qubit of the QuantumVariable at position (0,0,1). Since the type of this array is a
    QuantumFloat, with exponent -2, the significance of this qubit is 0.25.

    Note that the keys of the dictionary returned by the get_measurement method are no
    regular numpy arrays, as key objects need to be hashable. Instead, they are objects
    of an immutable subclass of np.ndarray called OutcomeArray, that supports hashing.

    For QuantumArrays, many methods known from numpy arrays work here too:

    >>> q_array = q_array.reshape(2,4)

    Not only do the ndarray methods work but also many other convenience functions from
    the numpy module:

    >>> q_array_swap = np.swapaxes(q_array, 0, 1)
    >>> print(q_array_swap)
    {OutcomeArray([[0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.]]): 0.5,
     OutcomeArray([[0.  , 0.  ],
                   [0.25, 0.  ],
                   [0.  , 0.  ],
                   [0.  , 0.  ]]): 0.5}

    To initiate the array, we use the :meth:`encode <qrisp.QuantumArray.encode>` method.
    Similar to QuantumVariables, we can also use the slicing operator, but this time
    non-trivial slices are possible as well:

    >>> q_array[1:,:] = 2*np.ones((1,4))
    >>> print(q_array)
    {OutcomeArray([[0., 0., 0., 0.],
                   [2., 2., 2., 2.]]): 0.5,
     OutcomeArray([[0.  , 0.25, 0.  , 0.  ],
                   [2.  , 2.  , 2.  , 2.  ]]): 0.5}


    **Quantum indexing**

    QuantumArrays can be dereferenced by :ref:`QuantumFloats <QuantumFloat>`. This
    returns a :ref:`QuantumEnvironment` in which the corresponding entry is avaliable as
    a QuantumVariable. ::

        from qrisp import QuantumBool, QuantumArray, QuantumFloat, h, multi_measurement

        q_array = QuantumArray(QuantumFloat(1), shape = (4,4))
        index_0 = QuantumFloat(2)
        index_1 = QuantumFloat(2)


        index_0[:] = 2
        index_1[:] = 1

        h(index_0[0])

        with q_array[index_0, index_1] as entry:
            x(entry)

    >>> print(multi_measurement([index_0, index_1, q_array]))
    {(2, 1, OutcomeArray([[0., 0., 0., 0.],
                          [0., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 0., 0.]])): 0.5,
     (3, 1, OutcomeArray([[0., 0., 0., 0.],
                          [0., 0., 0., 0.],
                          [0., 0., 0., 0.],
                          [0., 1., 0., 0.]])): 0.5}

    .. note::
        This only works for arrays which have a size of an integer power of 2.

    **Matrix multiplication**

    For QuantumArrays with ``qtype`` QuantumFloat, matrix multiplication is available.

    >>> q_array_1 = QuantumArray(qtype, (2,2))
    >>> q_array_2 = QuantumArray(qtype, (2,2))
    >>> q_array_1[:] = 2*np.eye(2)
    >>> q_array_2[:] = [[1,2],[3,4]]
    >>> print(q_array_1 @ q_array_2)
    {OutcomeArray([[2., 4.],
                   [6., 0.]]): 1.0}

    .. note::
        By default, the output matrix will have the same ``qtype`` as the first input
        matrix. Here, the ``qtype`` is a QuantumFloat with 5 mantissa bits and exponent
        -2, implying that the result 8 yields overflow. Since qrisps unsigend arithmetic
        is modular, we get a 0.

    It is also possible to multiply classical and quantum matrices

    >>> q_array = QuantumArray(qtype, (2,2))
    >>> q_array[:] = 3*np.eye(2)
    >>> cl_array = np.array([[1,2],[3,4]])
    >>> print(q_array @ cl_array)
    {OutcomeArray([[3., 6.],
                   [1., 4.]]): 1.0}
    """
    
    def __new__(subtype, qtype, shape=0, qs=None, name=None):
        if isinstance(shape, QuantumVariable):
            raise Exception

        obj = super().__new__(subtype, shape, dtype="object")

        if shape == 0:
            obj.shape_specified = False
        else:
            obj.shape_specified = True

        from qrisp.misc import find_calling_line

        if name is None:
            if type(obj) is QuantumArray:
                line = find_calling_line(1)
            else:
                line = find_calling_line(2)

            split_line = line.split("=")

            if len(split_line) == 2 and False:
                python_var_name = split_line[0]
                python_var_name = python_var_name.split(" ")[0]
                python_var_name = python_var_name.split(" ")[-1]
                name = python_var_name
            else:
                from qrisp import QuantumBool, QuantumChar, QuantumFloat

                if isinstance(qtype, QuantumFloat):
                    name = qtype.get_unique_name("qf_array")
                elif isinstance(qtype, QuantumBool):
                    name = qtype.get_unique_name("qbl_array")
                elif isinstance(qtype, QuantumChar):
                    name = qtype.get_unique_name("qch_array")
                else:
                    name = qtype.get_unique_name("qv_array")

        else:
            if name[-1] == "*":
                name = qtype.get_unique_name(name[:-1])

        obj.name = qtype.name

        # Set data size
        obj.msize = qtype.size
        obj.qtype = qtype

        if qs is None:
            from qrisp.core import QuantumSession

            qs = QuantumSession()

        if isinstance(shape, int):
            shape = (shape,)
        indices = list(product(*[list(range(i)) for i in shape]))

        for i in indices:

            temp_dup = qtype.duplicate(name=obj.name + "*", qs=qs)
            np.ndarray.__.__(obj, i, temp_dup)

        obj.qs = qs

        return obj

    def decoder(self, code_int):
        """
        The decoder method specifies how a QuantumArray turns the outcomes of
        measurements into human-readable values. It recieves an integer i and returns an
        OutcomeArray.

        Parameters
        ----------
        i : int
            Integer representing the outcome of a measurement of the qubits of this
            QuantumArray.

        Returns
        -------
        res : np.ndarray
            An array with entries of the type of the results of the .decoder of the
            qtype of this array.

        Examples
        --------

        We create a QuantumFloat and inspect its decoder:

        >>> from qrisp import QuantumArray, QuantumFloat
        >>> qtype = QuantumFloat(3)
        >>> q_array = QuantumArray(qtype, (2,2))
        >>> print(q_array.decoder(1))
        [[0 0]
         [0 1]]

        """

        flattened_array = self.flatten()

        from qrisp.qtypes.quantum_float import QuantumFloat

        if isinstance(self.qtype, QuantumFloat):
            if self.qtype.exponent >= 0:
                res = np.zeros(len(flattened_array), dtype=np.int32)
            else:
                res = np.zeros(len(flattened_array))
        else:
            res = np.zeros(len(flattened_array))

        n = len(self.qtype)

        bin_string = bin_rep(code_int, len(flattened_array) * n)

        for i in range(len(flattened_array)):
            if isinstance(self.qtype, QuantumFloat):
                res[i] = self.qtype.decoder(int(bin_string[i * n : (i + 1) * n], 2))
            else:
                res = res.astype("object")
                res[i] = self.qtype.decoder(int(bin_string[i * n : (i + 1) * n], 2))

        return OutcomeArray(res.reshape(self.shape))

    def encoder(self, encoding_array):
        """
        The encoder reverses the decoder, it turns arrays into integers based on the
        encoder of the ``qtype`` of this array.

        Parameters
        ----------
        encoding_array :
            An array where the entries can be read by the decoder of qtype.

        Raises
        ------
        Exception
            Tried to call encoder on array with mismatching shape.

        Returns
        -------
        i : int
            The integer encoding the given array.

        Examples
        --------

        We create a QuantumArray and inspect it's encoder:

        >>> from qrisp import QuantumArray, QuantumFloat
        >>> qtype = QuantumFloat(3)
        >>> q_array = QuantumArray(qtype, (2,2))
        >>> print(q_array.encoder(np.eye(2)))
        513
        """

        if isinstance(encoding_array, list):
            encoding_array = np.array(encoding_array)

        if self.shape != encoding_array.shape:
            raise Exception("Tried to call encoder on array with mismatching shape")

        flattened_encoding_array = encoding_array.flatten()
        flattened_quantum_array = self.flatten()

        encoding_int_str = ""

        for i in range(len(flattened_encoding_array)):
            qv = flattened_quantum_array[i]
            encoding_int_str += bin_rep(
                qv.encoder(flattened_encoding_array[i]), qv.size
            )[::-1]

        return int(encoding_int_str[::-1], 2)

    def set_shape(self, shape):
        """
        Method to specify a shape for arrays which have been created without one.

        Parameters
        ----------
        shape : tuple
            Desired shape.

        Raises
        ------
        Exception
            QuantumArray already has a shape.

        """

        if self.shape_specified:
            raise Exception(
                "Tried to set shape of QuantumArray, which already has a shape"
            )

        if isinstance(shape, int):
            shape = (shape,)

        self.resize(shape, refcheck=False)

        indices = list(product(*[list(range(i)) for i in shape]))

        for i in indices:
            temp_dup = self.qtype.duplicate(name=self.qtype.name + "_duplicate*", qs=self.qs)
            np.ndarray.__setitem__(self, i, temp_dup)

        self.shape_specified = True

    def __setitem__(self, key, value):
        if not self.shape_specified:
            if not hasattr(value, "shape"):
                value = np.array(value)
            self.set_shape(value.shape)

        if isinstance(value, QuantumArray):
            self[key].init_from(value)
            return

        if isinstance(value, QuantumVariable):
            return
        self[key].encode(value)
        return self

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ == "concatenate":
            from qrisp import merge

            if id(args[0][0].qtype) != id(args[0][1].qtype):
                raise Exception(
                    "Tried to concatenate arrays with differing qtype objects"
                )

            merge(args)

            ndarray_res = np.ndarray.__array_function__(self, func, types, args, kwargs)

            res = QuantumArray(self.qtype, shape=ndarray_res.shape)

            indices = product(*[list(range(i)) for i in ndarray_res.shape])
            for i in indices:
                np.ndarray.__setitem__(res, i, ndarray_res[i])

            res.qs = args[0][0].qs
            res.qtype = args[0][0].qtype

            return res

        else:
            return np.ndarray.__array_function__(self, func, types, args, kwargs)

    # Retrieves a measurement of the arrays state
    # Returns a list of tuples of the type (array, count)
    # ie. [(array([1,1,0]), 232), (array([1,1,3]), 115), ...]
    def get_measurement(
        self,
        backend=None,
        shots=None,
        compile=True,
        compilation_kwargs={},
        subs_dic={},
        circuit_preprocessor=None,
        precompiled_qc=None,
    ):
        """
        Method for acquiring measurement results for the given array. The semantics are
        similar to the :meth:`get_measurement <qrisp.QuantumVariable.get_measurement>`
        method of QuantumVariable. The results are returned as a dictionary of another
        numpy subtype called OutcomeArray.

        Parameters
        ----------
        backend : BackendClient, optional
            The backend on which to evaluate the quantum circuit. The default can be
            specified in the file default_backend.py.
        shots : integer, optional
            The amount of shots to evaluate the circuit. The default is given by the backend used.
        compile : bool, optional
            Boolean indicating if the .compile method of the underlying QuantumSession
            should be called before. The default is True.
        compilation_kwargs  : dict, optional
            Keyword arguments for the compile method. For more details check
            :meth:`QuantumSession.compile <qrisp.QuantumSession.compile>`.
            The default is ``{}``.
        subs_dic : dict, optional
            A dictionary of sympy symbols and floats to specify parameters in the case
            of a circuit with unspecified, abstract parameters. The default is {}.
        circuit_preprocessor : Python function, optional
            A function which recieves a QuantumCircuit and returns one, which is applied
            after compilation and parameter substitution. The default is None.

        Raises
        ------
        Exception
            Tried to get measurement within open environment.

        Returns
        -------
        list of tuples
            The measurement results in the form [(outcome_label, probability), ...].

        Examples
        --------

        >>> from qrisp import QuantumFloat, QuantumArray
        >>> qtype = QuantumFloat(3)
        >>> q_array = QuantumArray(qtype)
        >>> q_array[:] = [[1,0],[0,1]]
        >>> res = q_array.get_measurement()
        >>> print(res)
        {OutcomeArray([[1, 0],
                       [0, 1]]): 1.0}
        """

        for qv in self.flatten():
            if qv.is_deleted():
                raise Exception(
                    "Tried to measure QuantumArray containing deleted QuantumVariables"
                )

        if not self.shape_specified:
            raise Exception("Tried to measure QuantumArray without shape specification")

        if backend is None:
            if self.qs.backend is None:
                from qrisp.default_backend import def_backend

                backend = def_backend
            else:
                backend = self.qs.backend

        if len(self.qs.env_stack) != 0:
            raise Exception("Tried to get measurement within open environment")

        qubits = sum([qv.reg for qv in self.flatten()[::-1]], [])
        # Copy circuit in over to prevent modification
        # from qrisp.quantum_network import QuantumNetworkClient

        if precompiled_qc is None:
            if compile:
                qc = qompiler(
                    self.qs, intended_measurements=qubits, **compilation_kwargs
                )
            else:
                qc = self.qs.copy()

            # Transpile circuit
            qc = transpile(qc)
        else:
            qc = precompiled_qc.copy()

        # Bind parameters
        if subs_dic:
            qc = qc.bind_parameters(subs_dic)
            from qrisp.core.compilation import combine_single_qubit_gates

            qc = combine_single_qubit_gates(qc)

        # Execute user specified circuit_preprocessor
        if circuit_preprocessor is not None:
            qc = circuit_preprocessor(qc)

        from qrisp.misc import get_measurement_from_qc

        counts = get_measurement_from_qc(qc, qubits, backend, shots)

        # Insert outcome labels (if available and hashable)
        new_counts_dic = {}
        for key in counts.keys():
            outcome_label = self.decoder(key)

            new_counts_dic[outcome_label] = counts[key]

        counts = new_counts_dic

        # Sort keys
        sorted_key_list = list(counts.keys())
        sorted_key_list.sort(key=lambda x: -counts[x])
        counts = {key: counts[key] for key in sorted_key_list}

        return counts

    def encode(self, encoding_array):
        """
        The encode method allows to quickly bring a QuantumArray in a desired
        computational basis state. For this, it performs a circuit, bringing fresh
        qubits into the integer state specified by the encoder.

        A shorthand for this method is given by the ``[:]`` operator.

        Note that the qubits to initialize have to be fresh (i.e. no operations
        performed on them).

        Parameters
        ----------
        value :
            A value supported by the encoder.

        Examples
        --------

        We create a QuantumArray and encode the identity matrix.

        >>> from qrisp import QuantumArray, QuantumFloat
        >>> qtype = QuantumFloat(5)
        >>> q_array = QuantumArray(qtype, (4,4))
        >>> q_array.encode(np.eye(4))
        >>> print(q_array)
        {OutcomeArray([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]): 1.0}

        Using the slice operator we can also encode on slices of QuantumArrays

        >>> q_array = QuantumArray(qtype, (4,4))
        >>> q_array[:,:2] = np.ones((4,2))
        >>> print(q_array)
        {OutcomeArray([[1, 1, 0, 0],
                       [1, 1, 0, 0],
                       [1, 1, 0, 0],
                       [1, 1, 0, 0]]): 1.0}
        """

        flattened_array = self.flatten()

        qubit_list = sum([qv.reg for qv in flattened_array], [])

        from qrisp.misc import check_if_fresh, int_encoder

        if not check_if_fresh(qubit_list, self.qs):
            raise Exception("Tried to initialize qubits which are not fresh anymore.")

        int_encoder(qubit_list, self.encoder(encoding_array))

        return

    def delete(self, verify=False):
        r"""
        Performs the :meth:`delete <qrisp.QuantumVariable.delete>` method on all
        QuantumVariables in this array.

        Parameters
        ----------
        verify : bool, optional
            If this keyword is set to true, a simulator is queried to check if the
            deleted qubits are in the $\ket{0}$ state. The default is False.

        """
        for qv in self.flatten():
            qv.delete(verify=verify)

    def __repr__(self):
        return str(self.get_measurement())

    def __str__(self):
        return str(self.get_measurement())

    def __matmul__(self, other):
        from qrisp import QuantumFloat

        if isinstance(self.qtype, QuantumFloat):
            if isinstance(other, QuantumArray):
                from qrisp.alg_primitives.arithmetic import q_matmul

                return q_matmul(self, other)

            elif isinstance(other, np.ndarray):
                from qrisp.alg_primitives.arithmetic import semi_classic_matmul

                return semi_classic_matmul(self, other)

        raise Exception("Matrix multiplication for non-float types not implemented")

    def __rmatmul__(self, other):
        from qrisp import QuantumFloat

        if isinstance(self.qtype, QuantumFloat):
            return (self.transpose() @ other.transpose()).transpose()

    def __getitem__(self, index):
        from qrisp.environments import conjugate

        if isinstance(index, QuantumVariable):
            return conjugate(manipulate_array)(self, index)

        if isinstance(index, tuple):
            if isinstance(index[0], QuantumVariable):
                return conjugate(manipulate_array)(self, index)

        return np.ndarray.__getitem__(self, index)

    def init_state(self, tuple_list):
        """
        Method to initialize arbitrary quantum states in this array. The semantics are
        similar to the :meth:`QuantumVariable equivalent
        <qrisp.QuantumVariable.init_state>` of this method.

        The given state will be normalized.

        Parameters
        ----------
        tuple_list : list of tuples
            The list of tuples describing the quantum state. The first component of the
            tuples needs to represent the array and the second the required amplitude.

        Raises
        ------
        Exception
            Tried to initialize quantum state on qubits which are not fresh anymore.

        Examples
        --------

        We initialize a quantum state on an array and evaluate the measurement
        probabilities.

        >>> import numpy as np
        >>> from qrisp import QuantumArray, QuantumFloat
        >>> qtype = QuantumFloat(3)
        >>> q_array = QuantumArray(qtype, shape = 3)
        >>> q_array.init_state([(np.array([1, 2, 3]), 1), (np.array([1, 2, 2]), 0.5j)])
        >>> print(q_array)
        {OutcomeArray([1, 2, 3]): 0.8, OutcomeArray([1, 2, 2]): 0.2}

        """

        if isinstance(tuple_list, dict):
            tuple_list = [(k, v) for k, v in tuple_list.items()]

        flattened_array = self.flatten()
        qubit_list = []

        for qv in flattened_array:
            qubit_list += qv.reg
            from qrisp.misc import check_if_fresh

            if not check_if_fresh(qv.reg, qv.qs):
                raise Exception(
                    "Tried to initialize quantum state on qubits,"
                    "which are not fresh anymore"
                )

        target_array = np.zeros(2 ** len(qubit_list), dtype=np.complex128)

        for i in range(len(tuple_list)):
            target_array[self.encoder(tuple_list[i][0])] = tuple_list[i][1]

        target_array = target_array / np.vdot(target_array, target_array) ** 0.5

        from qrisp import init_state

        init_state(self, target_array)

    def init_from(self, other):
        """
        Performs the :meth:`init_from <qrisp.QuantumVariable.init_from>` method on all
        containing QuantumVariables based on their equivalent in the QuantumArray other.
        Note that this method does not copy the quantum state. For more details check
        the documentation of the init_from method of QuantumVariable.

        A shorthand for this method is the slicing operator ``[:]``.

        Parameters
        ----------
        other : QuantumArray
            The QuantumArray from which to initiate.

        Raises
        ------
        Exception
            Tried to initialize from array of invalied shape or qtype.

        Examples
        --------

        We create a QuantumArray and bring it into superposition

        >>> import numpy as np
        >>> from qrisp import QuantumArray, QuantumFloat, h, multi_measurement
        >>> qtype = QuantumFloat(5)
        >>> q_array_a = QuantumArray(qtype)
        >>> q_array_a[:] = np.eye(3)
        >>> h(q_array_a[0,0][0])
        >>> print(q_array_a)
        {OutcomeArray([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]): 0.5,
         OutcomeArray([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]): 0.5}

        We now duplicate this array and initiate

        >>> q_array_b = q_array_a.duplicate()
        >>> q_array_b.init_from(q_array_a)
        >>> print(multi_measurement([q_array_a, q_array_b]))
                {(OutcomeArray([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]),
                  OutcomeArray([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])): 0.5,
                 (OutcomeArray([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]),
                  OutcomeArray([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])): 0.5}


        We can achieve the same with the slicing operator

        >>> q_array_c = QuantumArray(qtype, shape = (3, 3))
        >>> q_array_c[1,:] = q_array_a[0,:]
        >>> print(multi_measurement([q_array_b, q_array_c]))
        {(OutcomeArray([[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]]),
          OutcomeArray([[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]])): 0.5,
         (OutcomeArray([[0, 0, 0],
                        [1, 0, 0],
                        [0, 0, 0]]),
          OutcomeArray([[0, 0, 0],
                        [1, 0, 0],
                        [0, 0, 0]])): 0.5}
        """
        if id(self.qtype) != id(other.qtype) or self.shape != other.shape:
            raise Exception("Tried to initialize from array of invalid shape or qtype")

        self_fl = self.flatten()
        other_fl = other.flatten()

        for i in range(len(self_fl)):
            self_fl[i].init_from(other_fl[i])

    def duplicate(self, init=False, qs=None):
        """
        This method returns a fresh QuantumArray, with equal ``qtype`` and shape.

        Parameters
        ----------
        init : bool, optional
            If set to True, the :meth:`init_from <qrisp.QuantumArray.init_from>` method
            will be called after creation. The default is False.
        qs : QuantumSession, optional
            The QuantumSession where the duplicate should be registered. By default,
            the duplicate will be registered in a new QuantumSession.

        Returns
        -------
        res : QuantumArray
            The duplicated array.

        Examples
        --------

        We duplicate a QuantumArray consisting of QuantumFloats with and without
        initiation.


        >>> from qrisp import QuantumArray, QuantumFloat
        >>> qtype = QuantumFloat(4)
        >>> q_array_0 = QuantumArray(qtype, (2,2))
        >>> q_array_0[:] = np.ones((2,2))
        >>> print(q_array_0)
        {OutcomeArray([[1, 1],
                       [1, 1]]): 1.0}
        >>> q_array_1 = q_array_0.duplicate()
        >>> print(q_array_1)
        {OutcomeArray([[0, 0],
                       [0, 0]]): 1.0}

        Note that no values have been carried over:

        >>> q_array_2 = q_array_0.duplicate(init = True)
        >>> print(q_array_2)
        {OutcomeArray([[1, 1],
                       [1, 1]]): 1.0}

        Now the values have been carried over. Note that this does NOT copy the state.
        For more information on this check the documentation of the
        :meth:`init_from <qrisp.QuantumVariable.init_from>` method of QuantumVariable.
        """

        res = self.copy()

        if qs is None:
            from qrisp import QuantumSession

            qs = QuantumSession()

        indices = product(*[list(range(i)) for i in self.shape])

        for i in indices:
            temp_dup = self.qtype.duplicate(qs=qs)
            np.ndarray.__setitem__(res, i, temp_dup)

        res.msize = self.qtype.size
        res.qtype = self.qtype
        res.qs = qs
        if init:
            res.init_from(self)

        return res

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.shape_specified = bool(obj.shape_specified)
        self.qtype = obj.qtype
        self.qs = obj.qs

    def most_likely(self, **kwargs):
        """
        Performs a measurement and returns the most likely outcome.

        Parameters
        ----------
        **kwargs : Keyword arguments for the get_measurement call.

        Examples
        --------

        >>> from qrisp import QuantumFloat, QuantumArray, ry
        >>> import numpy as np
        >>> qa = QuantumArray(QuantumFloat(3), shape = 4)
        >>> ry(np.pi*9/8, qa[0][0])
        >>> print(qa)
        {OutcomeArray([1, 0, 0, 0]): 0.9619, OutcomeArray([0, 0, 0, 0]): 0.0381}
        >>> qa.most_likely()
        OutcomeArray([1, 0, 0, 0])
        """
        return list(self.get_measurement(**kwargs))[0]


def manipulate_array(q_array, index):
    from qrisp import QuantumFloat, demux, invert

    if isinstance(index, tuple):
        if len(q_array.shape) != len(index):
            raise Exception(
                "Tried to quantum deref QuantumArray with index of mismatching shape"
            )

        for qf in index:
            if isinstance(qf, QuantumFloat):
                if qf.signed:
                    raise Exception("Tried to quantum deref with a signed QuantumFloat")
                if qf.exponent != 0:
                    raise Exception("Tried to quantum deref with a non-integer")

        index = sum([qv.reg for qv in index[::-1]], [])
        q_array = q_array.flatten()

    with invert():
        demux(q_array[0], index, q_array)

    return q_array[0]


class OutcomeArray(np.ndarray):
    def __new__(subtype, ndarray):
        if isinstance(ndarray, list):
            ndarray = np.array(ndarray)

            if ndarray.dtype == np.int64:
                ndarray = np.array(ndarray, dtype=np.int32)

        obj = super().__new__(subtype, ndarray.shape, dtype=ndarray.dtype)
        indices = product(*[list(range(i)) for i in ndarray.shape])
        for i in indices:
            np.ndarray.__setitem__(obj, i, ndarray[i])

        obj.flags.writeable = False
        return obj

    def __hash__(self):
        return hash(self.ravel().data.tobytes())
        # return hash(str(self))

    def __eq__(self, other):
        return np.array_equal(self, other)

    def __repr__(self):
        res = np.ndarray.__repr__(self).replace(", dtype=int32", "")

        return res


class QuantumArray:
    
    def __init__(self, qtype, shape, qs = None):
        
        if isinstance(shape, int):
            shape = (shape,)
        size = 1
        for s in shape:
            if not isinstance(s, int):
                raise Exception(f"Tried to create QuantumArray with non-integer tuple {shape}")
            size *= s
        
        self.ind_array = jnp.arange(size)
        self.ind_array = self.ind_array.reshape(shape)
        self.qtype = qtype
        
        if check_for_tracing_mode():
            qs = qtype.qs
            self.qs = qs
            qs.abs_qc, qb_array_tracer = create_qubits(qs.abs_qc, size*qtype.size)
            self.qb_array = DynamicQubitArray(qb_array_tracer)
            self.qtype_size = qtype.size
        else:
            if qs is None:
                self.qs = QuantumSession()
            else:
                self.qs = qs
            self.qv_list = []
            for i in range(size):
                self.qv_list.append(qtype.duplicate(name=self.qtype.name + "*", qs=self.qs))
    
    @property
    def shape(self):
        return self.ind_array.shape
    
    @property
    def size(self):
        return self.ind_array.size
    
    @property
    def ndim(self):
        return self.ind_array.ndim
        
    def __getitem__(self, key):
        
        from qrisp.environments import conjugate

        if isinstance(key, QuantumVariable):
            merge([self.qs, key.qs])
            return conjugate(manipulate_array)(self, key)

        if isinstance(key, tuple):
            if all(isinstance(index, QuantumVariable) for index in key):
                merge([self.qs, key[0].qs])
                return conjugate(manipulate_array)(self, key)
        
        if not isinstance(key, tuple):
            key = (key,)
        
        sliced_ind_array = self.ind_array[key]
        
        if len(sliced_ind_array.shape):
            res = copy.copy(self)
            res.ind_array = sliced_ind_array
            return res
        else:
            index = sliced_ind_array
            if check_for_tracing_mode():
                qv = copy.copy(self.qtype)
                s = self.qtype_size
                qv.reg = self.qb_array[index*s:(index+1)*s]
                qv.qs = self.qs
                return qv
            else:
                for i in range(len(key)):
                    if key[i] >= self.shape[i]:
                        raise Exception(f"Index {key} out of bounds for QuantumArray with shape {self.shape}")
                return self.qv_list[index]
    
    def __setitem__(self, key, value):
        if isinstance(value, QuantumVariable):
            return
        sliced_array = self[key]
        sliced_array.encode(value)
        
    
    def encode(self, value):
        
        if isinstance(value, list):
            value = np.array(value, dtype = "object")
        
        if not value.shape == self.shape:
            raise Exception("Tried to initialize a QuantumArray with incompatible shape")
        
        flattened_value_array = value.flatten()
        flat_self = self.flatten()
        
        if check_for_tracing_mode() and isinstance(flattened_value_array, (np.ndarray, list)):
            flattened_value_array = jnp.array(flattened_value_array)
        
        for i in jrange(self.size):
            flat_self[i][:] = flattened_value_array[i]
    
    def reshape(self, *args):
        if isinstance(args[0], (tuple, list)):
            shape = args[0]
        else:
            shape = args
        res = copy.copy(self)
        res.ind_array = self.ind_array.reshape(shape)
        return res
    
    def flatten(self):
        return self.reshape(self.size)
    
    def ravel(self):
        return self.flatten()
    
    def transpose(self, *args):
        res = copy.copy(self)
        res.ind_array = self.ind_array.transpose(*args)
        return res
    
    def swapaxes(self, *args, **kwargs):
        res = copy.copy(self)
        res.ind_array = self.ind_array.swapaxes(*args, **kwargs)
        return res
    
    def delete(self, verify = False):
        if check_for_tracing_mode():
            self.qs.clear_qubits(self.qb_array)
        else:
            for i in range(self.size):
                self.qv_list[i].delete(verify = verify)
        
    def measure(self):
        from qrisp import measure
        
        dtype = self.qtype.decoder(jnp.zeros(1)[0]).dtype
        
        meas_res = jnp.zeros(self.size, dtype = dtype)
        
        flattened_qa = self.flatten()
        
        
        def body_fun(i, val):
            meas_res, flattened_qa = val
            meas_res = meas_res.at[i].set(measure(flattened_qa[i]))
            return (meas_res, flattened_qa)
        
        meas_res, flattened_qa = q_fori_loop(0, 
                                             flattened_qa.size, 
                                             body_fun,
                                             (meas_res, flattened_qa))
        
        return meas_res.reshape(self.shape)
    
    # Retrieves a measurement of the arrays state
    # Returns a list of tuples of the type (array, count)
    # ie. [(array([1,1,0]), 232), (array([1,1,3]), 115), ...]
    def get_measurement(
        self,
        backend=None,
        shots=None,
        compile=True,
        compilation_kwargs={},
        subs_dic={},
        circuit_preprocessor=None,
        precompiled_qc=None,
    ):
        """
        Method for acquiring measurement results for the given array. The semantics are
        similar to the :meth:`get_measurement <qrisp.QuantumVariable.get_measurement>`
        method of QuantumVariable. The results are returned as a dictionary of another
        numpy subtype called OutcomeArray.

        Parameters
        ----------
        backend : BackendClient, optional
            The backend on which to evaluate the quantum circuit. The default can be
            specified in the file default_backend.py.
        shots : integer, optional
            The amount of shots to evaluate the circuit. The default is given by the backend used.
        compile : bool, optional
            Boolean indicating if the .compile method of the underlying QuantumSession
            should be called before. The default is True.
        compilation_kwargs  : dict, optional
            Keyword arguments for the compile method. For more details check
            :meth:`QuantumSession.compile <qrisp.QuantumSession.compile>`.
            The default is ``{}``.
        subs_dic : dict, optional
            A dictionary of sympy symbols and floats to specify parameters in the case
            of a circuit with unspecified, abstract parameters. The default is {}.
        circuit_preprocessor : Python function, optional
            A function which recieves a QuantumCircuit and returns one, which is applied
            after compilation and parameter substitution. The default is None.

        Raises
        ------
        Exception
            Tried to get measurement within open environment.

        Returns
        -------
        list of tuples
            The measurement results in the form [(outcome_label, probability), ...].

        Examples
        --------

        >>> from qrisp import QuantumFloat, QuantumArray
        >>> qtype = QuantumFloat(3)
        >>> q_array = QuantumArray(qtype)
        >>> q_array[:] = [[1,0],[0,1]]
        >>> res = q_array.get_measurement()
        >>> print(res)
        {OutcomeArray([[1, 0],
                       [0, 1]]): 1.0}
        """

        if check_for_tracing_mode():
            raise Exception("Tried to get_measurement from QuantumArray in tracing mode")

        for qv in self.flatten():
            if qv.is_deleted():
                raise Exception(
                    "Tried to measure QuantumArray containing deleted QuantumVariables"
                )


        if backend is None:
            if self.qs.backend is None:
                from qrisp.default_backend import def_backend

                backend = def_backend
            else:
                backend = self.qs.backend

        if len(self.qs.env_stack) != 0:
            raise Exception("Tried to get measurement within open environment")

        qubits = sum([qv.reg for qv in self.flatten()[::-1]], [])
        # Copy circuit in over to prevent modification
        # from qrisp.quantum_network import QuantumNetworkClient

        if precompiled_qc is None:
            if compile:
                qc = qompiler(
                    self.qs, intended_measurements=qubits, **compilation_kwargs
                )
            else:
                qc = self.qs.copy()

            # Transpile circuit
            qc = transpile(qc)
        else:
            qc = precompiled_qc.copy()

        # Bind parameters
        if subs_dic:
            qc = qc.bind_parameters(subs_dic)
            from qrisp.core.compilation import combine_single_qubit_gates

            qc = combine_single_qubit_gates(qc)

        # Execute user specified circuit_preprocessor
        if circuit_preprocessor is not None:
            qc = circuit_preprocessor(qc)

        from qrisp.misc import get_measurement_from_qc

        counts = get_measurement_from_qc(qc, qubits, backend, shots)

        # Insert outcome labels (if available and hashable)
        new_counts_dic = {}
        for key in counts.keys():
            outcome_label = self.decoder(key)

            new_counts_dic[outcome_label] = counts[key]

        counts = new_counts_dic

        # Sort keys
        sorted_key_list = list(counts.keys())
        sorted_key_list.sort(key=lambda x: -counts[x])
        counts = {key: counts[key] for key in sorted_key_list}

        return counts
    
    def decoder(self, code_int):
        """
        The decoder method specifies how a QuantumArray turns the outcomes of
        measurements into human-readable values. It recieves an integer i and returns an
        OutcomeArray.

        Parameters
        ----------
        i : int
            Integer representing the outcome of a measurement of the qubits of this
            QuantumArray.

        Returns
        -------
        res : np.ndarray
            An array with entries of the type of the results of the .decoder of the
            qtype of this array.

        Examples
        --------

        We create a QuantumFloat and inspect its decoder:

        >>> from qrisp import QuantumArray, QuantumFloat
        >>> qtype = QuantumFloat(3)
        >>> q_array = QuantumArray(qtype, (2,2))
        >>> print(q_array.decoder(1))
        [[0 0]
         [0 1]]

        """

        flattened_array = self.flatten()

        from qrisp.qtypes.quantum_float import QuantumFloat

        if isinstance(self.qtype, QuantumFloat):
            if self.qtype.exponent >= 0:
                res = np.zeros(len(flattened_array), dtype=np.int32)
            else:
                res = np.zeros(len(flattened_array))
        else:
            res = np.zeros(len(flattened_array))

        n = len(self.qtype)

        bin_string = bin_rep(code_int, len(flattened_array) * n)

        for i in range(len(flattened_array)):
            if isinstance(self.qtype, QuantumFloat):
                res[i] = self.qtype.decoder(int(bin_string[i * n : (i + 1) * n], 2))
            else:
                res = res.astype("object")
                res[i] = self.qtype.decoder(int(bin_string[i * n : (i + 1) * n], 2))

        return OutcomeArray(res.reshape(self.shape))
    
    def __len__(self):
        return len(self.ind_array)
    
    def __str__(self):
        if not check_for_tracing_mode():
            return str(self.get_measurement())
        else:
            return "<QuantumArray[" + str(self.shape)[1:-1] + "]>"
        
    def __matmul__(self, other):
        from qrisp import QuantumFloat

        if isinstance(self.qtype, QuantumFloat):
            if isinstance(other, QuantumArray):
                from qrisp.alg_primitives.arithmetic import q_matmul

                return q_matmul(self, other)

            elif isinstance(other, np.ndarray):
                from qrisp.alg_primitives.arithmetic import semi_classic_matmul

                return semi_classic_matmul(self, other)

        raise Exception("Matrix multiplication for non-float types not implemented")
        
    def __rmatmul__(self, other):
        from qrisp import QuantumFloat

        if isinstance(self.qtype, QuantumFloat):
            return (self.transpose() @ other.transpose()).transpose()
        
    def __iter__(self):
        return QuantumArrayIterator(self.flatten())
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc is np.matmul:
            return inputs[1].__rmatmul__(inputs[0])
        return NotImplemented
    
    def concatenate(self, other, axis = 0):
        
        if not self.qtype is other.qtype:
            raise Exception("Tried to concatenate two QuantumArrays with non-identical qtype")
        
        res = copy.copy(self)
        
        ind_array_other_shifted = other.ind_array + self.size
        
        concat_ind_array = jnp.concatenate((self.ind_array, ind_array_other_shifted), axis = axis)
        
        res.ind_array = concat_ind_array
        
        if check_for_tracing_mode():
            res.qb_array = self.qb_array + other.qb_array
        else:
            merge([self.qs, other.qs])
            res.qv_list = self.qv_list + other.qv_list
        
        return res


class QuantumArrayIterator:
    
    def __init__(self, qa):
        self.qa = qa
        self.counter = -1
    
    def __next__(self):
        self.counter += 1
        if self.counter >= self.qa.size:
            raise StopIteration
        return self.qa[self.counter]


def flatten_qa(qa):
    
    children = []
    
    qtype_children, qtype_aux_values = jax.tree.flatten(qa.qtype)
    
    children.append(qa.qtype_size)
    children.append(qa.ind_array)
    children.append(qa.qb_array)
    children.extend(qtype_children)
    
    aux_values = [qtype_aux_values]
    
    return tuple(children), tuple(aux_values)

def unflatten_qa(aux_data, children):
    
    qtype_children = children[3:]
    qtype_aux_values = aux_data[0]
    
    qtype = jax.tree.unflatten(qtype_aux_values, qtype_children)
    
    qa_dummy = object.__new__(QuantumArray)
    
    qtype_size = children[0]
    ind_array = children[1]
    qb_array  = children[2]
    
    qa_dummy.qtype_size = qtype_size
    qa_dummy.qtype = qtype
    qa_dummy.ind_array = ind_array
    qa_dummy.qb_array = qb_array
    qa_dummy.qs = TracingQuantumSession.get_instance()
    
    return qa_dummy
    
jax.tree_util.register_pytree_node(QuantumArray, flatten_qa, unflatten_qa)