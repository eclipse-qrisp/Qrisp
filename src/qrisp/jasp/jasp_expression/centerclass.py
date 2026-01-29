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

from functools import lru_cache

import jax
from jax import make_jaxpr
from jax.extend.core import Jaxpr, Literal, ClosedJaxpr
from jax.tree_util import tree_flatten

from qrisp.jasp.jasp_expression import invert_jaspr, collect_environments
from qrisp.jasp import (
    eval_jaxpr,
    pjit_to_gate,
    flatten_environments,
    cond_to_cl_control,
    extract_invalues,
    insert_outvalues
)
from qrisp.jasp.interpreter_tools.interpreters import ProcessedMeasurement
from qrisp.jasp.primitives import AbstractQuantumCircuit, QuantumPrimitive


class Jaspr(ClosedJaxpr):
    """
    The ``Jaspr`` class enables an efficient representations of a wide variety
    of (hybrid) algorithms. For many applications, the representation is agnostic
    to the scale of the problem, implying function calls with 10 or 10000 qubits
    can be represented by the same object. The actual unfolding to a circuit-level
    description is outsourced to
    `established, classical compilation infrastructure <https://mlir.llvm.org/>`_,
    implying state-of-the-art compilation speed can be reached.

    As a subtype of ``jax.extend.core.ClosedJaxpr``, Jasprs are embedded into the well matured
    `Jax ecosystem <https://github.com/n2cholas/awesome-jax>`_,
    which facilitates the compilation of classical `real-time computation <https://arxiv.org/abs/2206.12950>`_
    using some of the most advanced libraries in the world such as
    `CUDA <https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html>`_.
    Especially `machine learning <https://ai.google.dev/gemma/docs/jax_inference>`_
    and other scientific computations tasks are particularly well supported.

    To get a better understanding of the syntax and semantics of Jaxpr (and with
    that also Jaspr) please check `this link <https://jax.readthedocs.io/en/latest/jaxpr.html>`__.

    Similar to Jaxpr, Jaspr objects represent (hybrid) quantum
    algorithms in the form of a `functional programming language <https://en.wikipedia.org/wiki/Functional_programming>`_
    in `SSA-form <https://en.wikipedia.org/wiki/Static_single-assignment_form>`_.

    It is possible to compile Jaspr objects into QIR, which is facilitated by the
    `Catalyst framework <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__
    (check :meth:`qrisp.jasp.jaspr.to_qir` for more details).

    Qrisp scripts can be turned into Jaspr objects by
    calling the ``make_jaspr`` function, which has similar semantics as
    `jax.make_jaxpr <https://jax.readthedocs.io/en/latest/_autosummary/jax.make_jaxpr.html>`_.

    ::

        from qrisp import *
        from qrisp.jasp import make_jaspr

        def test_fun(i):

            qv = QuantumFloat(i, -1)
            x(qv[0])
            cx(qv[0], qv[i-1])
            meas_res = measure(qv)
            meas_res += 1
            return meas_res


        jaspr = make_jaspr(test_fun)(4)
        print(jaspr)

    This will give you the following output:

    .. code-block::

        { lambda ; a:QuantumCircuit b:i32[]. let
            c:QuantumCircuit d:QubitArray = create_qubits a b
            e:Qubit = get_qubit d 0
            f:QuantumCircuit = x c e
            g:i32[] = sub b 1
            h:Qubit = get_qubit d g
            i:QuantumCircuit = cx f e h
            j:QuantumCircuit k:i32[] = measure i d
            l:f32[] = convert_element_type[new_dtype=float64 weak_type=True] k
            m:f32[] = mul l 0.5
            n:f32[] = add m 1.0
          in (j, n) }


    A defining feature of the Jaspr class is that the first input and the
    first output are always of QuantumCircuit type. Therefore, Jaspr objects always
    represent some (hybrid) quantum operation.

    Qrisp comes with a built-in Jaspr interpreter. For that you simply have to
    call the object like a function:


    >>> print(jaspr(2))
    2.5
    >>> print(jaspr(4))
    5.5
    """

    __slots__ = (
        "permeability",
        "isqfree",
        "hashvalue",
        "ctrl_jaspr",
        "inv_jaspr",
        "envs_flattened",
    )

    def __init__(
        self, *args, permeability=None, isqfree=None, ctrl_jaspr=None, inv_jaspr=None, **kwargs
    ):

        if len(args) == 2:
            if not isinstance(args[0], Jaxpr) or not isinstance(args[1], list):
                raise Exception(f"Tried to call the Jaspr constructor with two arguments and signature {type(args[0]), type(args[1])} (allowed is (Jaxpr, list))")
            kwargs["jaxpr"] = args[0]            
            kwargs["consts"] = args[1]
            
        elif len(args) == 1:
            if not isinstance(args[0], ClosedJaxpr):
                raise Exception(f"Tried to call the Jaspr constructor with one argument and signature {type(args[0])} (allowed is (ClosedJaxpr))")
            kwargs["jaxpr"] = args[0].jaxpr
            kwargs["consts"] = args[0].consts

        if "jaxpr" in kwargs:

            ClosedJaxpr.__init__(self,
                                 kwargs["jaxpr"],
                                 kwargs["consts"])
        else:
            
            if "consts" in kwargs:
                consts = kwargs["consts"]
                del kwargs["consts"]
            else:
                if len(kwargs["constvars"]):
                    raise Exception("Tried to create Jaspr with constvars but no constants")
                consts = []
            
            ClosedJaxpr.__init__(self, 
                                 jaxpr = Jaxpr(**kwargs), 
                                 consts = consts)
            
        self.hashvalue = id(self)
        self.permeability = {}
        if permeability is None:
            permeability = {}
        for var in self.constvars + self.invars + self.outvars:
            if isinstance(var, Literal):
                continue
            self.permeability[var] = permeability.get(var, None)

        self.isqfree = isqfree
        self.ctrl_jaspr = ctrl_jaspr
        self.inv_jaspr = inv_jaspr
        self.envs_flattened = False

        if not isinstance(self.invars[-1].aval, AbstractQuantumCircuit):
            raise Exception(
                f"Tried to create a Jaspr from data that doesn't have a QuantumCircuit the last argument (got {type(self.invars[-1].aval)} instead)"
            )

        if not isinstance(self.outvars[-1].aval, AbstractQuantumCircuit):
            raise Exception(
                f"Tried to create a Jaspr from data that doesn't have a QuantumCircuit the last entry of return type (got {type(self.outvars[-1].aval)} instead)"
            )

    @property
    def constvars(self):
        return self.jaxpr.constvars
    
    @property
    def eqns(self):
        return self.jaxpr.eqns

    @property
    def invars(self):
        return self.jaxpr.invars

    @property
    def outvars(self):
        return self.jaxpr.outvars
    
    @property
    def debug_info(self):
        return self.jaxpr.debug_info

    def __hash__(self):
        return self.hashvalue

    def __eq__(self, other):
        if not isinstance(other, Jaxpr):
            return False

        return id(self) == id(other)

    def copy(self):

        if self.ctrl_jaspr is None:
            ctrl_jaspr = None
        else:
            ctrl_jaspr = self.ctrl_jaspr.copy()

        res = Jaspr(
            permeability=self.permeability,
            isqfree=self.isqfree,
            ctrl_jaspr=ctrl_jaspr,
            constvars=list(self.constvars),
            invars=list(self.invars),
            outvars=list(self.outvars),
            eqns=list(self.eqns),
            effects=self.effects,
            debug_info=self.debug_info,
        )

        res.envs_flattened = self.envs_flattened

        return res

    def inverse(self):
        """
        Returns the inverse Jaspr (if applicable). For Jaspr that contain realtime
        computations or measurements, the inverse does not exist.

        Returns
        -------
        Jaspr
            The daggered Jaspr.

        Examples
        --------

        We create a simple script and inspect the daggered version:

        ::

            from qrisp import *
            from qrisp.jasp import make_jaspr

            def example_function(i):

                qv = QuantumVariable(i)
                cx(qv[0], qv[1])
                t(qv[1])
                return qv

            jaspr = make_jaspr(example_function)(2)

            print(jaspr.inverse())
            # Yields
            # { lambda ; a:QuantumCircuit b:i32[]. let
            #     c:QuantumCircuit d:QubitArray = create_qubits a b
            #     e:Qubit = get_qubit d 0
            #     f:Qubit = get_qubit d 1
            #     g:QuantumCircuit = t_dg c f
            #     h:QuantumCircuit = cx g e f
            #   in (h, d) }
        """
        return invert_jaspr(self)

    def control(self, num_ctrl, ctrl_state=-1):
        """
        Returns the controlled version of the Jaspr. The control qubits are added
        to the signature of the Jaspr as the arguments after the QuantumCircuit.

        Parameters
        ----------
        num_ctrl : int
            The amount of controls to be added.
        ctrl_state : int of str, optional
            The control state on which to activate. The default is -1.

        Returns
        -------
        Jaspr
            The controlled Jaspr.

        Examples
        --------

        We create a simple script and inspect the controlled version:

        ::

            from qrisp import *
            from qrisp.jasp import make_jaspr

            def example_function(i):

                qv = QuantumVariable(i)
                cx(qv[0], qv[1])
                t(qv[1])
                return qv

            jaspr = make_jaspr(example_function)(2)

            print(jaspr.control(2))
            # Yields
            # { lambda ; a:QuantumCircuit b:Qubit c:Qubit d:i32[]. let
            #     e:QuantumCircuit f:QubitArray = create_qubits a 1
            #     g:Qubit = get_qubit f 0
            #     h:QuantumCircuit = ccx e b c g
            #     i:QuantumCircuit j:QubitArray = create_qubits h d
            #     k:Qubit = get_qubit j 0
            #     l:Qubit = get_qubit j 1
            #     m:QuantumCircuit = ccx i g k l
            #     n:QuantumCircuit = ct m g l
            #     o:QuantumCircuit = ccx n b c g
            #   in (o, j) }

        We see that the control qubits are part of the function signature
        (``a`` and ``b``)

        """
        if self.ctrl_jaspr is not None and num_ctrl == 1 and ctrl_state == -1:
            return self.ctrl_jaspr

        from qrisp.jasp import ControlledJaspr

        if isinstance(ctrl_state, int):
            if ctrl_state < 0:
                ctrl_state += 2**num_ctrl

            ctrl_state = bin(ctrl_state)[2:].zfill(num_ctrl)
        else:
            ctrl_state = str(ctrl_state)

        return ControlledJaspr.from_cache(self, ctrl_state)

    def to_qc(self, *args):
        """
        Converts the Jaspr into a :ref:`QuantumCircuit` if applicable. Circuit
        conversion of algorithms involving realtime computations is not possible.
        
        Any computations that perform classical postprocessing of measurements
        can not be reflected within the QuantumCircuit object itself and will
        generate an object of type ``ProcessedMeasurement``. These objects hold
        no further information and are simply used as placeholders to emulate
        the computation.

        Parameters
        ----------
        *args : tuple
            The arguments to call the Jaspr with.

        Returns
        -------
        return_values : tuple
            The return values of the Jaspr. QuantumVariable return types are
            returned as lists of Qubits.
        :ref:`QuantumCircuit`
            The resulting QuantumCircuit.

        Examples
        --------

        We create a simple script and inspect the QuantumCircuit:

        ::

            from qrisp import *
            from qrisp.jasp import make_jaspr

            def example_function(i):

                qv = QuantumVariable(i)
                cx(qv[0], qv[1])
                t(qv[1])
                return qv

            jaspr = make_jaspr(example_function)(2)

            qb_list, qc = jaspr.to_qc(2)
            print(qc)
            # Yields
            # qb_0: ──■───────
            #       ┌─┴─┐┌───┐
            # qb_1: ┤ X ├┤ T ├
            #       └───┘└───┘
            
        To demonstrate the behavior under measurement post-processing, we build
        a similar script:
            
        ::
            
            from qrisp import ProcessedMeasurement
            
            def example_function(i):
            
                qf = QuantumFloat(i)
                cx(qf[0], qf[1])
                t(qf[1])
                
                meas_res = measure(qf)
                # Perform classical post processing
                meas_res *= 2
                return meas_res
            
            jaspr = make_jaspr(example_function)(2)
            
            meas_res, qc = jaspr.to_qc(2)
            print(isinstance(meas_res, ProcessedMeasurement))
            # True
            
        

        """
        from qrisp.jasp.interpreter_tools.interpreters import jaspr_to_qc
        return jaspr_to_qc(self, *args)

    def extract_post_processing(self, *args):
        """
        Extracts the post-processing logic from this Jaspr and returns a function
        that performs the post-processing on measurement results.
        
        This method is useful for separating the quantum circuit from the classical
        post-processing of measurement results. The quantum circuit can be executed
        on a NISQ-style backend to obtain measurement results, and then the post-processing
        function can be applied to those results to obtain the final output.
        
        .. note::
            
            It is not possible to extract QuantumCircuits from Jaspr objects
            involving real-time computation, but it is possible to extract a post
            processing function.
        
        Parameters
        ----------
        *args : tuple
            The static argument values that were used for circuit extraction. 
            These will be bound into the post-processing function as Literals.
        
        Returns
        -------
        callable
            A function that takes measurement results and returns the post-processed results.
            Accepts either a string of '0' and '1' characters or a JAX array of booleans
            with shape (n,). String inputs are automatically converted to boolean arrays.
        
        Examples
        --------
        
        We create a Jaspr that performs post-processing on measurement results:
        
        ::
        
            from qrisp import *
            import jax.numpy as jnp
            
            @make_jaspr
            def example_function(i):
                qv = QuantumFloat(5)
                # First measurement
                meas_1 = measure(qv[i])
                h(qv[1])
                # Second measurement
                meas_2 = measure(qv[1])
                # Classical post-processing
                return meas_1 + 2, meas_2
            
            jaspr = example_function(1)
            
            # Extract the quantum circuit
            a, b, qc = jaspr.to_qc(1)
            
            # Extract the post-processing function with the SAME arguments
            post_proc = jaspr.extract_post_processing(1)
            
            # Execute qc on a backend to get measurement results
            results = qc.run()
            
            # Apply post-processing to each result
            for bitstring, count in results.items():
                processed = post_proc(bitstring)
                print(f"{bitstring} -> {processed}")
              
            # Yields:
            # 00 -> (Array(2, dtype=int64), Array(False, dtype=bool))
            # 01 -> (Array(2, dtype=int64), Array(True, dtype=bool))
            
            # Can also use with array input (useful for JAX jitting):
            import jax.numpy as jnp
            meas_array = jnp.array([False, True])
            processed = post_proc(meas_array)
        
        Note that the static arguments (in this case `1`) must be the same as those
        used for circuit extraction, since they affect the structure of both the
        quantum circuit and the post-processing logic.
        """
        from qrisp.jasp.interpreter_tools.interpreters import extract_post_processing
        return extract_post_processing(self, *args)

    def eval(self, *args, eqn_evaluator=lambda x, y: True):
        return eval_jaxpr(self, eqn_evaluator=eqn_evaluator)(*args)

    def flatten_environments(self):
        """
        Flattens all environments by applying the corresponding compilation
        routines such that no more ``q_env`` primitives are left.

        Returns
        -------
        Jaspr
            The Jaspr with flattened environments.

        Examples
        --------

        We create a Jaspr containing an :ref:`InversionEnvironment` and flatten:

        ::

                def test_function(i):
                        qv = QuantumVariable(i)

                        with invert():
                                t(qv[0])
                                cx(qv[0], qv[1])

                        return qv

                jaspr = make_jaspr(test_function)(2)
                print(jaspr)

        ::

                { lambda ; a:QuantumCircuit b:i32[]. let
                        c:QuantumCircuit d:QubitArray = create_qubits a b
                        e:QuantumCircuit = q_env[
                        jaspr={ lambda ; f:QuantumCircuit d:QubitArray. let
                  g:Qubit = get_qubit d 0
                  h:QuantumCircuit = t f g
                  i:Qubit = get_qubit d 1
                  j:QuantumCircuit = cx h g i
                in (j,) }
                        type=InversionEnvironment
                        ] c d
                  in (e, d) }

        You can see how the body of the :ref:`InversionEnvironment` is __collected__
        into another Jaspr. This reflects the fact that at their core,
        :ref:`QuantumEnvironment <QuantumEnvironment>` describe `higher-order
        quantum functions <https://en.wikipedia.org/wiki/Higher-order_function>`_
        (ie. functions that operate on functions). In order to apply the
        transformations induced by the QuantumEnvironment, we can call
        ``jaspr.flatten_environments``:

        >>> print(jaspr.flatten_environments)
        { lambda ; a:QuantumCircuit b:i32[]. let
            c:QuantumCircuit d:QubitArray = create_qubits a b
            e:Qubit = get_qubit d 0
            f:Qubit = get_qubit d 1
            g:QuantumCircuit = cx c e f
            h:QuantumCircuit = t_dg g e
          in (h, d) }

        We see that as expected, the order of the ``cx`` and the ``t`` gate has been switched and the ``t`` gate has been turned into a ``t_dg``.

        """
        res = flatten_environments(self)
        if self.ctrl_jaspr is not None:
            res.ctrl_jaspr = self.ctrl_jaspr.flatten_environments()
        if self.inv_jaspr is not None:
            if not self.inv_jaspr.envs_flattened:
                res.inv_jaspr = self.inv_jaspr.flatten_environments()
            else:
                res.inv_jaspr = self.inv_jaspr
            res.inv_jaspr.inv_jaspr = res
        return res

    def __call__(self, *args):
        from qrisp.jasp.evaluation_tools.jaspification import simulate_jaspr

        return simulate_jaspr(self, *args)

        if len(self.outvars) == 1:
            return None

        from qrisp.simulator import BufferedQuantumState

        args = [BufferedQuantumState()] + list(tree_flatten(args)[0])

        from qrisp.jasp import extract_invalues, insert_outvalues, eval_jaxpr

        flattened_jaspr = self

        def eqn_evaluator(eqn, context_dic):
            if eqn.primitive.name == "jit":

                if eqn.params["name"] == "expectation_value_eval_function":
                    from qrisp.jasp.program_control import sampling_evaluator

                    sampling_evaluator("ev")(
                        eqn, context_dic, eqn_evaluator=eqn_evaluator
                    )
                    return

                if eqn.params["name"] == "sampling_eval_function":
                    from qrisp.jasp.program_control import sampling_evaluator

                    sampling_evaluator("array")(
                        eqn, context_dic, eqn_evaluator=eqn_evaluator
                    )
                    return

                invalues = extract_invalues(eqn, context_dic)
                outvalues = eval_jaxpr(
                    eqn.params["jaxpr"], eqn_evaluator=eqn_evaluator
                )(*invalues)
                if not isinstance(outvalues, (list, tuple)):
                    outvalues = [outvalues]
                insert_outvalues(eqn, context_dic, outvalues)
            elif eqn.primitive.name == "jasp.create_quantum_kernel":
                insert_outvalues(eqn, context_dic, BufferedQuantumState())
            else:
                return True

        res = eval_jaxpr(flattened_jaspr, eqn_evaluator=eqn_evaluator)(
            *(args + self.consts)
        )

        if len(self.outvars) == 2:
            return res[1]
        else:
            return res[1:]

    def inline(self, *args):

        from qrisp.jasp import TracingQuantumSession

        qs = TracingQuantumSession.get_instance()
        abs_qc = qs.abs_qc

        ammended_args = list(args) + [abs_qc]
        res = eval_jaxpr(self)(*ammended_args)

        if isinstance(res, tuple):
            new_abs_qc = res[-1]
            res = res[:-1]
        else:
            new_abs_qc = res
            res = None
        qs.abs_qc = new_abs_qc
        return res

    def count_ops(self, *args, meas_behavior="1"):
        from qrisp.jasp.evaluation_tools import profile_jaspr

        return profile_jaspr(self, meas_behavior)(*args)

    def embedd(self, *args, name=None, inline=False):
        from qrisp.jasp import TracingQuantumSession, get_last_equation

        qs = TracingQuantumSession.get_instance()
        abs_qc = qs.abs_qc

        ammended_args = list(args) + [abs_qc]
        if not inline:
            res = jax.jit(eval_jaxpr(self))(*ammended_args)

            eqn = get_last_equation()
            
            eqn.params["jaxpr"] = self
            if name is not None:
                eqn.params["name"] = name
        else:
            res = eval_jaxpr(self)(*ammended_args)

        if isinstance(res, tuple):
            new_abs_qc = res[-1]
            res = res[:-1]
        else:
            new_abs_qc = res
            res = None
        qs.abs_qc = new_abs_qc
        return res

    def qjit(self, *args, function_name="jaspr_function", device=None):
        """
        Leverages the Catalyst pipeline to compile a QIR representation of
        this function and executes that function using the Catalyst QIR runtime.

        Parameters
        ----------
        *args : iterable
            The arguments to call the function with.
        device : object
            The `PennyLane device <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/devices.html>`_ to execute the function. 
            The default device is `"lightning.qubit" <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_qubit/device.html>`_, 
            a fast state-vector qubit simulator.

        Returns
        -------
            The values returned by the compiled, executed function.

        """
        flattened_jaspr = self

        from qrisp.jasp.evaluation_tools.catalyst_interface import (
            jaspr_to_catalyst_qjit,
        )

        qjit_obj = jaspr_to_catalyst_qjit(flattened_jaspr, function_name=function_name, device=device)
        res = qjit_obj.compiled_function(*args)
        if not isinstance(res, (tuple, list)):
            return res
        elif len(res) == 1:
            return res[0]
        else:
            return res

    @classmethod
    @lru_cache(maxsize=int(1e5))
    def from_cache(cls, closed_jaxpr):
        res = Jaspr(jaxpr=closed_jaxpr.jaxpr, consts=closed_jaxpr.consts)
        remove_redundant_allocations(res)
        return res

    def update_eqns(self, eqns):
        return Jaspr(
            constvars=list(self.constvars),
            invars=list(self.invars),
            outvars=list(self.outvars),
            eqns=list(eqns),
            consts=list(self.consts),
            debug_info=self.debug_info
        )

    def to_qir(self):
        """
        Compiles the Jaspr to QIR using the `Catalyst framework <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The QIR string.

        Examples
        --------

        We create a simple script and inspect the QIR string:

        ::

            from qrisp import *
            from qrisp.jasp import make_jaspr

            def example_function(i):

                qv = QuantumFloat(i)
                cx(qv[0], qv[1])
                t(qv[1])
                meas_res = measure(qv)
                meas_res += 1
                return meas_res

            jaspr = make_jaspr(example_function)(2)
            print(jaspr.to_qir())

        Yields:

        .. code-block:: none

            ; ModuleID = 'LLVMDialectModule'
            source_filename = "LLVMDialectModule"

            @"{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" = internal constant [66 x i8] c"{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}\00"
            @lightning.qubit = internal constant [16 x i8] c"lightning.qubit\00"
            @"/home/positr0nium/miniconda3/envs/qrisp/lib/python3.10/site-packages/catalyst/utils/../lib/librtd_lightning.so" = internal constant [111 x i8] c"/home/positr0nium/miniconda3/envs/qrisp/lib/python3.10/site-packages/catalyst/utils/../lib/librtd_lightning.so\00"

            declare void @__catalyst__rt__finalize() local_unnamed_addr

            declare void @__catalyst__rt__initialize() local_unnamed_addr

            declare ptr @__catalyst__qis__Measure(ptr, i32) local_unnamed_addr

            declare void @__catalyst__qis__T(ptr, ptr) local_unnamed_addr

            declare void @__catalyst__qis__CNOT(ptr, ptr, ptr) local_unnamed_addr

            declare ptr @__catalyst__rt__array_get_element_ptr_1d(ptr, i64) local_unnamed_addr

            declare ptr @__catalyst__rt__qubit_allocate_array(i64) local_unnamed_addr

            declare void @__catalyst__rt__device_init(ptr, ptr, ptr) local_unnamed_addr

            declare void @_mlir_memref_to_llvm_free(ptr) local_unnamed_addr

            declare ptr @_mlir_memref_to_llvm_alloc(i64) local_unnamed_addr

            define { ptr, ptr, i64 } @jit_jaspr_function(ptr nocapture readnone %0, ptr nocapture readonly %1, i64 %2) local_unnamed_addr {
              tail call void @__catalyst__rt__device_init(ptr nonnull @"/home/positr0nium/miniconda3/envs/qrisp/lib/python3.10/site-packages/catalyst/utils/../lib/librtd_lightning.so", ptr nonnull @lightning.qubit, ptr nonnull @"{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}")
              %4 = tail call ptr @__catalyst__rt__qubit_allocate_array(i64 20)
              %5 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 0)
              %6 = load ptr, ptr %5, align 8
              %7 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 1)
              %8 = load ptr, ptr %7, align 8
              tail call void @__catalyst__qis__CNOT(ptr %6, ptr %8, ptr null)
              %9 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 1)
              %10 = load ptr, ptr %9, align 8
              tail call void @__catalyst__qis__T(ptr %10, ptr null)
              %11 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 65)
              %12 = ptrtoint ptr %11 to i64
              %13 = add i64 %12, 63
              %14 = and i64 %13, -64
              %15 = inttoptr i64 %14 to ptr
              %16 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 65)
              %17 = ptrtoint ptr %16 to i64
              %18 = add i64 %17, 63
              %19 = and i64 %18, -64
              %20 = inttoptr i64 %19 to ptr
              %21 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
              %22 = ptrtoint ptr %21 to i64
              %23 = add i64 %22, 63
              %24 = and i64 %23, -64
              %25 = inttoptr i64 %24 to ptr
              %26 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
              %27 = ptrtoint ptr %26 to i64
              %28 = add i64 %27, 63
              %29 = and i64 %28, -64
              %30 = inttoptr i64 %29 to ptr
              %31 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
              %32 = ptrtoint ptr %31 to i64
              %33 = add i64 %32, 63
              %34 = and i64 %33, -64
              %35 = inttoptr i64 %34 to ptr
              %36 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
              %37 = ptrtoint ptr %36 to i64
              %38 = add i64 %37, 63
              %39 = and i64 %38, -64
              %40 = inttoptr i64 %39 to ptr
              %41 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
              store i64 0, ptr %41, align 1
              %42 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
              store i64 0, ptr %42, align 1
              %43 = load i64, ptr %1, align 4
              %44 = icmp slt i64 %43, 1
              store i1 %44, ptr %15, align 64
              %45 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
              %46 = load i64, ptr %42, align 1
              store i64 %46, ptr %45, align 1
              tail call void @_mlir_memref_to_llvm_free(ptr nonnull %42)
              %47 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
              %48 = load i64, ptr %41, align 1
              store i64 %48, ptr %47, align 1
              tail call void @_mlir_memref_to_llvm_free(ptr nonnull %41)
              br i1 %44, label %.lr.ph, label %._crit_edge

            .lr.ph:                                           ; preds = %3, %.lr.ph
              %49 = phi ptr [ %87, %.lr.ph ], [ %47, %3 ]
              %50 = phi ptr [ %85, %.lr.ph ], [ %45, %3 ]
              %51 = load i64, ptr %50, align 4
              %52 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 %51)
              %53 = load ptr, ptr %52, align 8
              %54 = tail call ptr @__catalyst__qis__Measure(ptr %53, i32 -1)
              %55 = load i1, ptr %54, align 1
              store i1 %55, ptr %20, align 64
              %56 = load i64, ptr %50, align 4
              store i64 %56, ptr %25, align 64
              %57 = shl i64 2, %56
              %58 = icmp ult i64 %56, 64
              %59 = select i1 %58, i64 %57, i64 0
              store i64 %59, ptr %30, align 64
              %60 = load i1, ptr %20, align 64
              %61 = zext i1 %60 to i64
              store i64 %61, ptr %35, align 64
              %62 = load i64, ptr %30, align 64
              %63 = select i1 %60, i64 %62, i64 0
              store i64 %63, ptr %40, align 64
              %64 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
              %65 = ptrtoint ptr %64 to i64
              %66 = add i64 %65, 63
              %67 = and i64 %66, -64
              %68 = inttoptr i64 %67 to ptr
              %69 = load i64, ptr %49, align 4
              %70 = load i64, ptr %40, align 64
              %71 = add i64 %70, %69
              store i64 %71, ptr %68, align 64
              tail call void @_mlir_memref_to_llvm_free(ptr nonnull %49)
              %72 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
              %73 = ptrtoint ptr %72 to i64
              %74 = add i64 %73, 63
              %75 = and i64 %74, -64
              %76 = inttoptr i64 %75 to ptr
              %77 = load i64, ptr %50, align 4
              %78 = add i64 %77, 1
              store i64 %78, ptr %76, align 64
              tail call void @_mlir_memref_to_llvm_free(ptr nonnull %50)
              %79 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
              %80 = load i64, ptr %68, align 64
              store i64 %80, ptr %79, align 1
              tail call void @_mlir_memref_to_llvm_free(ptr %64)
              %81 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
              %82 = load i64, ptr %76, align 64
              store i64 %82, ptr %81, align 1
              tail call void @_mlir_memref_to_llvm_free(ptr %72)
              %.pre = load i64, ptr %81, align 4
              %83 = load i64, ptr %1, align 4
              %84 = icmp sge i64 %.pre, %83
              store i1 %84, ptr %15, align 64
              %85 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
              %86 = load i64, ptr %81, align 1
              store i64 %86, ptr %85, align 1
              tail call void @_mlir_memref_to_llvm_free(ptr nonnull %81)
              %87 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
              %88 = load i64, ptr %79, align 1
              store i64 %88, ptr %87, align 1
              tail call void @_mlir_memref_to_llvm_free(ptr nonnull %79)
              br i1 %84, label %.lr.ph, label %._crit_edge

            ._crit_edge:                                      ; preds = %.lr.ph, %3
              %.lcssa20 = phi ptr [ %45, %3 ], [ %85, %.lr.ph ]
              %.lcssa = phi ptr [ %47, %3 ], [ %87, %.lr.ph ]
              tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.lcssa20)
              tail call void @_mlir_memref_to_llvm_free(ptr %36)
              tail call void @_mlir_memref_to_llvm_free(ptr %31)
              tail call void @_mlir_memref_to_llvm_free(ptr %26)
              tail call void @_mlir_memref_to_llvm_free(ptr %21)
              tail call void @_mlir_memref_to_llvm_free(ptr %16)
              tail call void @_mlir_memref_to_llvm_free(ptr %11)
              %89 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
              %90 = ptrtoint ptr %89 to i64
              %91 = add i64 %90, 63
              %92 = and i64 %91, -64
              %93 = inttoptr i64 %92 to ptr
              %94 = load i64, ptr %.lcssa, align 4
              %95 = trunc i64 %94 to i32
              store i32 %95, ptr %93, align 64
              tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.lcssa)
              %96 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
              %97 = ptrtoint ptr %96 to i64
              %98 = add i64 %97, 63
              %99 = and i64 %98, -64
              %100 = inttoptr i64 %99 to ptr
              %101 = load i32, ptr %93, align 64
              %102 = add i32 %101, 1
              store i32 %102, ptr %100, align 64
              tail call void @_mlir_memref_to_llvm_free(ptr %89)
              %103 = icmp eq ptr %96, inttoptr (i64 3735928559 to ptr)
              br i1 %103, label %104, label %107

            104:                                              ; preds = %._crit_edge
              %105 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 4)
              %106 = load i32, ptr %100, align 64
              store i32 %106, ptr %105, align 1
              br label %107

            107:                                              ; preds = %104, %._crit_edge
              %.pn16 = phi ptr [ %105, %104 ], [ %96, %._crit_edge ]
              %.pn14 = phi ptr [ %105, %104 ], [ %100, %._crit_edge ]
              %.pn13 = insertvalue { ptr, ptr, i64 } undef, ptr %.pn16, 0
              %.pn = insertvalue { ptr, ptr, i64 } %.pn13, ptr %.pn14, 1
              %108 = insertvalue { ptr, ptr, i64 } %.pn, i64 0, 2
              ret { ptr, ptr, i64 } %108
            }

            define void @_catalyst_pyface_jit_jaspr_function(ptr nocapture writeonly %0, ptr nocapture readonly %1) local_unnamed_addr {
              %.unpack = load ptr, ptr %1, align 8
              %.elt1.i = getelementptr inbounds { ptr, ptr, i64 }, ptr %.unpack, i64 0, i32 1
              %.unpack2.i = load ptr, ptr %.elt1.i, align 8
              %3 = tail call { ptr, ptr, i64 } @jit_jaspr_function(ptr poison, ptr %.unpack2.i, i64 poison)
              %.elt.i = extractvalue { ptr, ptr, i64 } %3, 0
              store ptr %.elt.i, ptr %0, align 8
              %.repack5.i = getelementptr inbounds { ptr, ptr, i64 }, ptr %0, i64 0, i32 1
              %.elt6.i = extractvalue { ptr, ptr, i64 } %3, 1
              store ptr %.elt6.i, ptr %.repack5.i, align 8
              %.repack7.i = getelementptr inbounds { ptr, ptr, i64 }, ptr %0, i64 0, i32 2
              %.elt8.i = extractvalue { ptr, ptr, i64 } %3, 2
              store i64 %.elt8.i, ptr %.repack7.i, align 8
              ret void
            }

            define void @_catalyst_ciface_jit_jaspr_function(ptr nocapture writeonly %0, ptr nocapture readonly %1) local_unnamed_addr {
              %.elt1 = getelementptr inbounds { ptr, ptr, i64 }, ptr %1, i64 0, i32 1
              %.unpack2 = load ptr, ptr %.elt1, align 8
              %3 = tail call { ptr, ptr, i64 } @jit_jaspr_function(ptr poison, ptr %.unpack2, i64 poison)
              %.elt = extractvalue { ptr, ptr, i64 } %3, 0
              store ptr %.elt, ptr %0, align 8
              %.repack5 = getelementptr inbounds { ptr, ptr, i64 }, ptr %0, i64 0, i32 1
              %.elt6 = extractvalue { ptr, ptr, i64 } %3, 1
              store ptr %.elt6, ptr %.repack5, align 8
              %.repack7 = getelementptr inbounds { ptr, ptr, i64 }, ptr %0, i64 0, i32 2
              %.elt8 = extractvalue { ptr, ptr, i64 } %3, 2
              store i64 %.elt8, ptr %.repack7, align 8
              ret void
            }

            define void @setup() local_unnamed_addr {
              tail call void @__catalyst__rt__initialize()
              ret void
            }

            define void @teardown() local_unnamed_addr {
              tail call void @__catalyst__rt__finalize()
              ret void
            }

            !llvm.module.flags = !{!0}

            !0 = !{i32 2, !"Debug Info Version", i32 3}

        """
        from qrisp.jasp.evaluation_tools.catalyst_interface import jaspr_to_qir

        return jaspr_to_qir(self.flatten_environments())
    
    def to_mlir(self):
        """
        Compiles the Jaspr to an xDSL module using the Jasp Dialect.
        Requires the xDSL package to be installed (``pip install xdsl``).
        
        
        .. note::
        
            An xDSL module can be visualized via:
                
            ::
                
                print(xdsl_module)
                
            and serialized to a string using:
                
            ::
                
                from xdsl.printer import Printer
                Printer().print_op(xdsl_module)
        

        Returns
        -------
        xdsl.dialects.builtin.ModuleOp
            An xDSL module representing the quantum computation.
            
        Examples
        --------

        We create a simple script and inspect the MLIR string:

        ::

            from qrisp import *
            from qrisp.jasp import make_jaspr

            def example_function(i):

                qv = QuantumVariable(i)
                cx(qv[0], qv[1])
                t(qv[1])
                meas_res = measure(qv)
                meas_res += 1
                return meas_res

            jaspr = make_jaspr(example_function)(2)
            print(jaspr.to_mlir())
            
        .. code-block:: none
        
            builtin.module @jasp_module {
              func.func public @main(%arg0 : tensor<i64>, %arg1 : !jasp.QuantumState) -> (tensor<i64>, !jasp.QuantumState) {
                %0, %1 = "jasp.create_qubits"(%arg0, %arg1) : (tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState)
                %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %3 = "jasp.get_qubit"(%0, %2) : (!jasp.QubitArray, tensor<i64>) -> !jasp.Qubit
                %4 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %5 = "jasp.get_qubit"(%0, %4) : (!jasp.QubitArray, tensor<i64>) -> !jasp.Qubit
                %6 = "jasp.quantum_gate"(%3, %5, %1) {gate_type = "cx"} : (!jasp.Qubit, !jasp.Qubit, !jasp.QuantumState) -> !jasp.QuantumState
                %7 = "jasp.quantum_gate"(%5, %6) {gate_type = "t"} : (!jasp.Qubit, !jasp.QuantumState) -> !jasp.QuantumState
                %8, %9 = "jasp.measure"(%0, %7) : (!jasp.QubitArray, !jasp.QuantumState) -> (tensor<i64>, !jasp.QuantumState)
                %10 = "stablehlo.add"(%8, %4) : (tensor<i64>, tensor<i64>) -> tensor<i64>
                %11 = "jasp.reset"(%0, %9) : (!jasp.QubitArray, !jasp.QuantumState) -> !jasp.QuantumState
                %12 = "jasp.delete_qubits"(%0, %11) : (!jasp.QubitArray, !jasp.QuantumState) -> !jasp.QuantumState
                func.return %10, %12 : tensor<i64>, !jasp.QuantumState
              }
            }
        
        """
        from qrisp.jasp.mlir import jaspr_to_mlir
        return jaspr_to_mlir(self)

    def to_catalyst_mlir(self):
        """
        Compiles the Jaspr to MLIR using the `Catalyst dialect <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The MLIR string.

        Examples
        --------

        We create a simple script and inspect the MLIR string:

        ::

            from qrisp import *
            from qrisp.jasp import make_jaspr

            def example_function(i):

                qv = QuantumFloat(i)
                cx(qv[0], qv[1])
                t(qv[1])
                meas_res = measure(qv)
                meas_res += 1
                return meas_res

            jaspr = make_jaspr(example_function)(2)
            print(jaspr.to_catalyst_mlir())

        .. code-block:: none

                        module @jaspr_function {
              func.func public @jit_jaspr_function(%arg0: tensor<i64>) -> tensor<i32> attributes {llvm.emit_c_interface} {
                %0 = stablehlo.constant dense<1> : tensor<i32>
                %1 = stablehlo.constant dense<2> : tensor<i64>
                %2 = stablehlo.constant dense<1> : tensor<i64>
                %3 = stablehlo.constant dense<0> : tensor<i64>
                quantum.device["/home/positr0nium/miniconda3/envs/qrisp/lib/python3.10/site-packages/catalyst/utils/../lib/librtd_lightning.so", "lightning.qubit", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
                %4 = quantum.alloc( 20) : !quantum.reg
                %5 = quantum.extract %4[ 0] : !quantum.reg -> !quantum.bit
                %6 = quantum.extract %4[ 1] : !quantum.reg -> !quantum.bit
                %out_qubits:2 = quantum.custom "CNOT"() %5, %6 : !quantum.bit, !quantum.bit
                %7 = quantum.insert %4[ 0], %out_qubits#0 : !quantum.reg, !quantum.bit
                %8 = quantum.insert %7[ 1], %out_qubits#1 : !quantum.reg, !quantum.bit
                %9 = quantum.extract %8[ 1] : !quantum.reg -> !quantum.bit
                %out_qubits_0 = quantum.custom "T"() %9 : !quantum.bit
                %10 = quantum.insert %8[ 1], %out_qubits_0 : !quantum.reg, !quantum.bit
                %11 = stablehlo.add %3, %arg0 : tensor<i64>
                %12:3 = scf.while (%arg1 = %3, %arg2 = %3, %arg3 = %10) : (tensor<i64>, tensor<i64>, !quantum.reg) -> (tensor<i64>, tensor<i64>, !quantum.reg) {
                  %16 = stablehlo.compare  GE, %arg1, %11,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                  %extracted = tensor.extract %16[] : tensor<i1>
                  scf.condition(%extracted) %arg1, %arg2, %arg3 : tensor<i64>, tensor<i64>, !quantum.reg
                } do {
                ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: !quantum.reg):
                  %extracted = tensor.extract %arg1[] : tensor<i64>
                  %16 = quantum.extract %arg3[%extracted] : !quantum.reg -> !quantum.bit
                  %mres, %out_qubit = quantum.measure %16 : i1, !quantum.bit
                  %from_elements = tensor.from_elements %mres : tensor<i1>
                  %extracted_1 = tensor.extract %arg1[] : tensor<i64>
                  %17 = quantum.insert %arg3[%extracted_1], %out_qubit : !quantum.reg, !quantum.bit
                  %18 = stablehlo.subtract %arg1, %3 : tensor<i64>
                  %19 = stablehlo.shift_left %1, %18 : tensor<i64>
                  %20 = stablehlo.convert %from_elements : (tensor<i1>) -> tensor<i64>
                  %21 = stablehlo.multiply %19, %20 : tensor<i64>
                  %22 = stablehlo.add %arg2, %21 : tensor<i64>
                  %23 = stablehlo.add %arg1, %2 : tensor<i64>
                  scf.yield %23, %22, %17 : tensor<i64>, tensor<i64>, !quantum.reg
                }
                %13 = stablehlo.convert %12#1 : (tensor<i64>) -> tensor<i32>
                %14 = stablehlo.multiply %13, %0 : tensor<i32>
                %15 = stablehlo.add %14, %0 : tensor<i32>
                return %15 : tensor<i32>
              }
              func.func @setup() {
                quantum.init
                return
              }
              func.func @teardown() {
                quantum.finalize
                return
              }
            }

        """
        from qrisp.jasp.evaluation_tools.catalyst_interface import jaspr_to_mlir

        return jaspr_to_mlir(self.flatten_environments())

    def to_qasm(self, *args):
        """
        Compiles the Jaspr into an OpenQASM 2 string. Real-time control is possible
        as long as no computations on the measurement results are performed.

        Parameters
        ----------
        *args : list
            The arguments to call the :ref:`QuantumCircuit` evaluation with.

        Returns
        -------
        str
            The OpenQASM 2 string.

        Examples
        --------

        We create a simple script and inspect the QASM 2 string:

        ::

            from qrisp import *
            from qrisp.jasp import make_jaspr

            def main(i):

                qv = QuantumVariable(i)
                cx(qv[0], qv[1])
                t(qv[1])
                return qv

            jaspr = make_jaspr(main)(2)

            qasm_str = jaspr.to_qasm(2)
            print(qasm_str)
            # Yields

            # OPENQASM 2.0;
            # include "qelib1.inc";
            # qreg qb_59[1];
            # qreg qb_60[1];
            # cx qb_59[0],qb_60[0];
            # t qb_60[0];


        It is also possible to compile simple real-time control features:

        ::

            def main(phi):

                qf = QuantumFloat(5)
                h(qf)
                bl = measure(qf[0])

                with control(bl):
                    rz(phi, qf[1])
                    x(qf[1])

                return

            jaspr = make_jaspr(main)(0.5)
            print(jaspr.to_qasm(0.5))

        This gives:

        ::

            OPENQASM 2.0;
            include "qelib1.inc";
            qreg qb_59[1];
            qreg qb_60[1];
            qreg qb_61[1];
            qreg qb_62[1];
            qreg qb_63[1];
            creg cb_0[1];
            h qb_59[0];
            h qb_60[0];
            h qb_61[0];
            reset qb_61[0];
            h qb_62[0];
            reset qb_62[0];
            h qb_63[0];
            reset qb_63[0];
            measure qb_59[0] -> cb_0[0];
            reset qb_59[0];
            if(cb_0==1) rz(0.5) qb_60[0];
            if(cb_0==1) x qb_60[0];
            reset qb_60[0];

        """
        res = self.to_qc(*args)
        if len(self.outvars) == 1:
            res = [res]
        qrisp_qc = res[-1]
        return qrisp_qc.qasm()

    def to_catalyst_jaxpr(self):
        """
        Compiles the jaspr to the corresponding `Catalyst jaxpr <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__.

        Parameters
        ----------
        *args : tuple
            The arguments to call the jaspr with.

        Returns
        -------
        jax.core.Jaxpr
            The Jaxpr using Catalyst primitives.

        Examples
        --------

        We create a simple script and inspect the Catalyst Jaxpr:

        ::

            from qrisp import *
            from qrisp.jasp import make_jaspr

            def example_function(i):

                qv = QuantumFloat(i)
                cx(qv[0], qv[1])
                t(qv[1])
                meas_res = measure(qv)
                meas_res += 1
                return meas_res

            jaspr = make_jaspr(example_function)(2)

            print(jaspr.to_catalyst_jaxpr())
            # Yields
            # { lambda ; a:AbstractQreg() b:i64[] c:i32[]. let
            #   d:i64[] = convert_element_type[new_dtype=int64 weak_type=True] c
            #   e:i64[] = add b d
            #   f:i64[] = add b 0
            #   g:i64[] = add b 1
            #   h:AbstractQbit() = qextract a f
            #   i:AbstractQbit() = qextract a g
            #   j:AbstractQbit() k:AbstractQbit() = qinst[op=CNOT qubits_len=2] h i
            #   l:AbstractQreg() = qinsert a f j
            #   m:AbstractQreg() = qinsert l g k
            #   n:AbstractQbit() = qextract m g
            #   o:AbstractQbit() = qinst[op=T qubits_len=1] n
            #   p:AbstractQreg() = qinsert m g o
            #   q:i64[] = convert_element_type[new_dtype=int64 weak_type=True] c
            #   r:i64[] = add b q
            #   _:i64[] s:i64[] t:AbstractQreg() _:i64[] _:i64[] = while_loop[
            #     body_jaxpr={ lambda ; u:i64[] v:i64[] w:AbstractQreg() x:i64[] y:i64[]. let
            #         z:AbstractQbit() = qextract w u
            #         ba:bool[] bb:AbstractQbit() = qmeasure z
            #         bc:AbstractQreg() = qinsert w u bb
            #         bd:i64[] = sub u x
            #         be:i64[] = shift_left 2 bd
            #         bf:i64[] = convert_element_type[new_dtype=int64 weak_type=True] ba
            #         bg:i64[] = mul be bf
            #         bh:i64[] = add v bg
            #         bi:i64[] = add u 1
            #       in (bi, bh, bc, x, y) }
            #     body_nconsts=0
            #     cond_jaxpr={ lambda ; bj:i64[] bk:i64[] bl:AbstractQreg() bm:i64[] bn:i64[]. let
            #         bo:bool[] = ge bj bn
            #       in (bo,) }
            #     cond_nconsts=0
            #     nimplicit=0
            #     preserve_dimensions=True
            #   ] b 0 p b r
            #   bp:i32[] = convert_element_type[new_dtype=int64 weak_type=False] s
            #   bq:i32[] = mul bp 1
            #   br:i32[] = add bq 1
            # in (t, e, br) }

        """
        from qrisp.jasp.evaluation_tools.catalyst_interface import (
            jaspr_to_catalyst_jaxpr,
        )

        return jaspr_to_catalyst_jaxpr(self.flatten_environments())


def make_jaspr(fun, flatten_envs=True, **jax_kwargs):
    from qrisp.jasp import (
        AbstractQuantumCircuit,
        TracingQuantumSession,
        check_for_tracing_mode,
    )
    from qrisp.core import recursive_qv_search

    def jaspr_creator(*args, **kwargs):

        qs = TracingQuantumSession.get_instance()

        # Close any tracing quantum sessions that might have not been
        # properly closed due to whatever reason.
        if not check_for_tracing_mode():
            while qs.abs_qc is not None:
                qs.conclude_tracing()

        # This function will be traced by Jax.
        # Note that we add the abs_qc keyword as the tracing quantum circuit
        def ammended_function(*args, **kwargs):

            abs_qc = kwargs[10*"~"]
            del kwargs[10*"~"]

            qs.start_tracing(abs_qc)

            # If the signature contains QuantumVariables, these QuantumVariables went
            # through a flattening/unflattening procedure. The unflattening creates
            # a copy of the QuantumVariable object, which is however not yet registered in any
            # QuantumSession. We register these QuantumVariables in the current QuantumSession.
            arg_qvs = recursive_qv_search(args)
            for qv in arg_qvs:
                qs.register_qv(qv, None)

            try:
                res = fun(*args, **kwargs)
            except Exception as e:
                qs.conclude_tracing()
                raise e

            res_qc = qs.conclude_tracing()

            return res, res_qc

        ammended_kwargs = dict(kwargs)
        ammended_kwargs[10*"~"] = AbstractQuantumCircuit()
        closed_jaxpr = make_jaxpr(ammended_function, **jax_kwargs)(
            *args, **ammended_kwargs
        )
            
        # Collect the environments
        # This means that the quantum environments no longer appear as
        # enter/exit primitives but as primitive that "call" a certain Jaspr.
        res = Jaspr.from_cache(collect_environments(closed_jaxpr))

        if flatten_envs:
            res = res.flatten_environments()

        return res

    # Since we are calling the "ammended function", where the first parameter
    # is the AbstractQuantumCircuit, we need to move the static_argnums indicator.
    if "static_argnums" in jax_kwargs:
        jax_kwargs = dict(jax_kwargs)
        if isinstance(jax_kwargs["static_argnums"], list):
            jax_kwargs["static_argnums"] = list(jax_kwargs["static_argnums"])
            for i in range(len(jax_kwargs["static_argnums"])):
                jax_kwargs["static_argnums"][i] += 1
        else:
            jax_kwargs["static_argnums"] += 1

    return jaspr_creator


def check_aval_equivalence(invars_1, invars_2):
    avals_1 = [invar.aval for invar in invars_1]
    avals_2 = [invar.aval for invar in invars_2]
    return all([type(avals_1[i]) == type(avals_2[i]) for i in range(len(avals_1))])

def remove_redundant_allocations(closed_jaxpr):
    """
    Optimizes the Jaspr by removing redundant qubit allocations.
    
    Strategy:
    1.  Map usages of all variables to identify how QubitArrays are consumed.
    2.  Identify `jasp.create_qubits` operations that are redundant. An allocation is redundant if:
        -   The resulting QubitArray is not returned by the function.
        -   The QubitArray is either unused (DropVar) or ONLY used by `free` primitives 
            (like `get_size`, `delete_qubits`) that don't actually require the physical qubits 
            if we know the allocation parameters.
    3.  Plan removals and replacements:
        -   Mark redundant `create_qubits` equations for removal.
        -   Mark dependent `get_size`, `reset`, `delete_qubits` equations for removal.
        -   Map the output circuit of removed equations to their input circuit (bypassing the operation).
        -   Map the output of `get_size` to the size input of the `create_qubits` equation.
    4.  Rewrite the Jaxpr:
        -   Filter out removed equations.
        -   Update variable references in remaining equations and output variables using the replacement map.
    """
    from jax.core import DropVar
    try:
        from jax.extend.core import Literal
    except ImportError:
        from jax.core import Literal
    
    jaxpr = closed_jaxpr.jaxpr
    eqns = jaxpr.eqns
    
    usages = {}
    
    # 1. Build usage map
    # We iterate over all equations and record which equation uses which input variable.
    for eqn in eqns:
        for var in eqn.invars:
            if isinstance(var, DropVar):
                continue
            if isinstance(var, Literal):
                continue
            if var not in usages:
                usages[var] = []
            usages[var].append(eqn)
    
    # Identify variables that are returned by the function, as these cannot be optimized away.
    returned_vars = set()
    for var in jaxpr.outvars:
        if not isinstance(var, DropVar) and not isinstance(var, Literal):
            returned_vars.add(var)
    
    replacements = {}
    eqns_to_remove = set()
    
    # Primitives that are "safe" to exist on a redundant qubit array.
    # These operations can be resolved without the actual qubits if the allocation is known.
    allowed_primitives = {"jasp.get_size", "jasp.delete_qubits"}
    
    # 2. Identify and plan removals
    for eqn in eqns:
        if eqn.primitive.name == "jasp.create_qubits":
            out_qa = eqn.outvars[0]
            out_qc = eqn.outvars[1]
            
            # If the QubitArray is part of the output, we must keep it.
            if out_qa in returned_vars:
                continue
                
            # Check if out_qa is only used by allowed primitives or not at all
            is_redundant = False
            
            if isinstance(out_qa, DropVar) or out_qa not in usages:
                is_redundant = True
            else:
                only_allowed_usages = True
                for u_eqn in usages[out_qa]:
                    if u_eqn.primitive.name not in allowed_primitives:
                        # Found a usage (e.g., gate) that requires real qubits.
                        only_allowed_usages = False
                        break
                if only_allowed_usages:
                    is_redundant = True
            
            if is_redundant:
                eqns_to_remove.add(id(eqn))
                
                # Rewire circuit: Bypass the allocation by mapping output circuit to input circuit.
                replacements[out_qc] = eqn.invars[1]
                
                # Handle dependent equations (safe primitives)
                if not isinstance(out_qa, DropVar) and out_qa in usages:
                    for u_eqn in usages[out_qa]:
                        eqns_to_remove.add(id(u_eqn))
                        
                        if u_eqn.primitive.name == "jasp.get_size":
                             # Optimize get_size: Use the input size from create_qubits directly.
                             replacements[u_eqn.outvars[0]] = eqn.invars[0]
                        elif u_eqn.primitive.name == "jasp.delete_qubits":
                             # Bypass cleanup: Map the cleanup's output circuit to its input circuit.
                             replacements[u_eqn.outvars[0]] = u_eqn.invars[1]

    if not eqns_to_remove:
        return

    # 3. Apply replacements
    def resolve(var):
        # Literals are not hashable and cannot be in replacements
        if isinstance(var, Literal):
            return var
        try:
            # Resolving chains of replacements
            while var in replacements:
                var = replacements[var]
        except TypeError:
            # Handle other potentially unhashable types
            pass
        return var

    new_eqns = []
    for eqn in eqns:
        if id(eqn) in eqns_to_remove:
            continue
        
        # Update inputs of remaining equations using the replacement map
        eqn.invars[:] = [resolve(var) for var in eqn.invars]
        new_eqns.append(eqn)
        
    # Update the equations list in-place
    jaxpr.eqns[:] = new_eqns
    # Update the output variables of the jaxpr
    jaxpr.outvars[:] = [resolve(var) for var in jaxpr.outvars]

