"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, cast

import jax
from jax import make_jaxpr
from jax._src.interpreters import partial_eval as part_eval
from jax._src.util import split_list
from jax.core import DebugInfo, DropVar
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal, Var

from qrisp._cache_config import qrisp_lru_compilation_cache
from qrisp.jasp import (
    eval_jaxpr,
    flatten_environments,
)
from qrisp.jasp.jasp_expression import collect_environments, invert_jaspr
from qrisp.jasp.primitives import AbstractQuantumState


class Jaspr(ClosedJaxpr):
    """
    The ``Jaspr`` class enables an efficient representation of a wide variety
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
    and other scientific computation tasks are particularly well supported.

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

        { lambda ; a:i64[] b:QuantumState. let
            c:QubitArray d:QuantumState = jasp.create_qubits a b
            e:Qubit = jasp.get_qubit c 0:i64[]
            f:QuantumState = jasp.quantum_gate[gate=x] e d
            g:i64[] = sub a 1:i64[]
            h:Qubit = jasp.get_qubit c g
            i:QuantumState = jasp.quantum_gate[gate=cx] e h f
            j:i64[] k:QuantumState = jasp.measure c i
            l:f64[] = integer_pow[y=-1] 2.0:f64[]
            m:f64[] = convert_element_type[new_dtype=float64 weak_type=False] j
            n:f64[] = mul m l
            o:f64[] = add n 1.0:f64[]
          in (o, k) }


    A defining feature of the Jaspr class is that the first input and the
    first output are always of QuantumState type. Therefore, Jaspr objects always
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
        self,
        *args,
        permeability: dict | None = None,
        isqfree: bool | None = None,
        ctrl_jaspr: "Jaspr | None" = None,
        inv_jaspr: "Jaspr | None" = None,
        **kwargs,
    ) -> None:
        if len(args) == 2:
            if not isinstance(args[0], Jaxpr) or not isinstance(args[1], list):
                raise TypeError(
                    f"Two-argument Jaspr constructor expects (Jaxpr, list), "
                    f"got ({type(args[0]).__name__}, {type(args[1]).__name__})"
                )
            kwargs["jaxpr"] = args[0]
            kwargs["consts"] = args[1]

        elif len(args) == 1:
            if not isinstance(args[0], ClosedJaxpr):
                raise TypeError(f"One-argument Jaspr constructor expects ClosedJaxpr, got {type(args[0]).__name__}")
            kwargs["jaxpr"] = args[0].jaxpr
            kwargs["consts"] = args[0].consts

        if "jaxpr" in kwargs:
            ClosedJaxpr.__init__(self, kwargs["jaxpr"], kwargs["consts"])
        else:
            if "consts" in kwargs:
                consts = kwargs.pop("consts")
            else:
                if kwargs["constvars"]:
                    raise ValueError("Tried to create Jaspr with constvars but no constants")
                consts = []

            ClosedJaxpr.__init__(self, jaxpr=Jaxpr(**kwargs), consts=consts)

        self.hashvalue = id(self)
        self.permeability: dict = {}
        if permeability is None:
            permeability = {}
        for var in self.constvars + self.invars + self.outvars:
            if isinstance(var, Literal):
                continue
            self.permeability[var] = permeability.get(var)

        self.isqfree = isqfree
        self.ctrl_jaspr = ctrl_jaspr
        self.inv_jaspr = inv_jaspr
        self.envs_flattened = False

        if not isinstance(self.invars[-1].aval, AbstractQuantumState):
            raise ValueError(f"Last invar must be QuantumState, got {type(self.invars[-1].aval).__name__}")

        if not isinstance(self.outvars[-1].aval, AbstractQuantumState):
            raise ValueError(f"Last outvar must be QuantumState, got {type(self.outvars[-1].aval).__name__}")

    @property
    def constvars(self) -> list[Var]:
        """Constant variables of the underlying Jaxpr."""
        return self.jaxpr.constvars

    @property
    def eqns(self):
        """Equations of the underlying Jaxpr."""
        return self.jaxpr.eqns

    @property
    def invars(self) -> list[Var]:
        """Input variables of the underlying Jaxpr."""
        return self.jaxpr.invars

    @property
    def outvars(self) -> list[Var | Literal]:
        """Output variables of the underlying Jaxpr."""
        return self.jaxpr.outvars

    @property
    def debug_info(self) -> DebugInfo | None:
        """Debug info attached to the underlying Jaxpr, or None."""
        return self.jaxpr.debug_info

    def __hash__(self) -> int:
        return self.hashvalue

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Jaxpr):
            return False
        return id(self) == id(other)

    def copy(self) -> "Jaspr":
        """Return a shallow copy of this Jaspr with copied list attributes."""
        if self.ctrl_jaspr is None:
            ctrl_jaspr = None
        else:
            ctrl_jaspr = self.ctrl_jaspr.copy()

        kwargs = {
            "permeability": self.permeability,
            "isqfree": self.isqfree,
            "ctrl_jaspr": ctrl_jaspr,
            "constvars": list(self.constvars),
            "invars": list(self.invars),
            "outvars": list(self.outvars),
            "eqns": list(self.eqns),
            "effects": self.effects,
            "debug_info": self.debug_info,
        }
        if self.consts:
            kwargs["consts"] = list(self.consts)

        res = Jaspr(**kwargs)

        res.envs_flattened = self.envs_flattened

        return res

    def inverse(self) -> "Jaspr":
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
            # { lambda ; a:i64[] b:QuantumState. let
            #     c:QubitArray d:QuantumState = jasp.create_qubits a b
            #     e:Qubit = jasp.get_qubit c 0:i64[]
            #     f:Qubit = jasp.get_qubit c 1:i64[]
            #     g:QuantumState = jasp.quantum_gate[gate=t_dg] f d
            #     h:QuantumState = jasp.quantum_gate[gate=cx] e f g
            #   in (c, h) }
        """
        return invert_jaspr(self)

    def control(self, num_ctrl: int, ctrl_state: int | str = -1) -> "Jaspr":
        """
        Returns the controlled version of the Jaspr. The control qubits are added
        to the signature of the Jaspr as the arguments after the QuantumState.

        Parameters
        ----------
        num_ctrl : int
            The amount of controls to be added.
        ctrl_state : int or str, optional
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
            # { lambda ; a:Qubit b:Qubit c:i64[] d:QuantumState. let
            #     e:QubitArray f:QuantumState = jasp.create_qubits 1:i64[] d
            #     g:Qubit = jasp.get_qubit e 0:i64[]
            #     h:QuantumState = jasp.quantum_gate[gate=2cx] a b g f
            #     i:QubitArray j:QuantumState = jasp.create_qubits c h
            #     k:Qubit = jasp.get_qubit i 0:i64[]
            #     l:Qubit = jasp.get_qubit i 1:i64[]
            #     m:QuantumState = jasp.quantum_gate[gate=ccx] g k l j
            #     n:QuantumState = jasp.quantum_gate[gate=ct] g l m
            #     o:QuantumState = jasp.quantum_gate[gate=2cx] a b g n
            #     p:QuantumState = jasp.delete_qubits e o
            #   in (i, p) }

        We see that the control qubits are part of the function signature
        (``a`` and ``b``).

        """
        if self.ctrl_jaspr is not None and num_ctrl == 1 and ctrl_state == -1:
            return self.ctrl_jaspr

        from qrisp.jasp import ControlledJaspr

        if isinstance(ctrl_state, int):
            ctrl_int: int = ctrl_state
            if ctrl_int < 0:
                ctrl_int += 2**num_ctrl
            ctrl_state = bin(ctrl_int)[2:].zfill(num_ctrl)
        else:
            ctrl_state = str(ctrl_state)

        return ControlledJaspr.from_cache(self, ctrl_state)

    def to_qc(self, *args: Any) -> Any:
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

    def extract_post_processing(self, *args: Any) -> Callable:
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
            meas_array = jnp.array([False, True])
            processed = post_proc(meas_array)

        Note that the static arguments (in this case `1`) must be the same as those
        used for circuit extraction, since they affect the structure of both the
        quantum circuit and the post-processing logic.
        """
        from qrisp.jasp.interpreter_tools.interpreters import extract_post_processing

        return extract_post_processing(self, *args)

    def eval(self, *args: Any, eqn_evaluator: Callable = lambda x, y: True) -> Any:
        """Evaluate this Jaspr with a custom per-equation evaluator hook."""
        return eval_jaxpr(self, eqn_evaluator=eqn_evaluator)(*args)

    def flatten_environments(self) -> "Jaspr":
        """
        Flattens all environments by applying the corresponding compilation
        routines such that no more ``q_env`` primitives are left.

        Returns
        -------
        Jaspr
            The Jaspr with flattened environments.

        Examples
        --------

        Create a Jaspr with ``flatten_envs=False`` so that the
        :ref:`InversionEnvironment` is still visible as a ``jasp.q_env`` primitive:

        ::

            from qrisp import QuantumVariable, cx, t, invert
            from qrisp.jasp import make_jaspr

            def bell_state_inverted(i):
                qv = QuantumVariable(i)
                with invert():
                    t(qv[0])
                    cx(qv[0], qv[1])
                return qv

            jaspr = make_jaspr(bell_state_inverted, flatten_envs=False)(2)
            print(jaspr)

        ::

            { lambda ; a:i64[] b:QuantumState. let
                c:QubitArray d:QuantumState = jasp.create_qubits a b
                e:QuantumState = jasp.q_env[
                  jaspr={ lambda ; c:QubitArray f:QuantumState. let
                      g:Qubit = jasp.get_qubit c 0:i64[]
                      h:QuantumState = jasp.quantum_gate[gate=t] g f
                      i:Qubit = jasp.get_qubit c 1:i64[]
                      j:QuantumState = jasp.quantum_gate[gate=cx] g i h
                    in (j,) }
                  type=InversionEnvironment
                ] c d
              in (c, e) }

        The body of the :ref:`InversionEnvironment` is *collected* into a nested
        Jaspr. Calling :meth:`flatten_environments` applies the inversion
        transformation, reversing gate order and replacing each gate with its
        inverse:

        ::

            print(jaspr.flatten_environments())

        ::

            { lambda ; a:i64[] b:QuantumState. let
                c:QubitArray d:QuantumState = jasp.create_qubits a b
                e:Qubit = jasp.get_qubit c 0:i64[]
                f:Qubit = jasp.get_qubit c 1:i64[]
                g:QuantumState = jasp.quantum_gate[gate=cx] e f d
                h:QuantumState = jasp.quantum_gate[gate=t_dg] e g
              in (c, h) }

        As expected, ``cx`` and ``t`` have been swapped and ``t`` replaced by
        ``t_dg`` (the dagger/inverse of ``t``).

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

    def __call__(self, *args: Any) -> Any:
        from qrisp.jasp.evaluation_tools.jaspification import simulate_jaspr

        return simulate_jaspr(self, *args)

    def inline(self, *args: Any) -> Any:
        """Inline this Jaspr into the current tracing context without JIT-wrapping."""
        from qrisp.jasp import TracingQuantumSession

        qs = TracingQuantumSession.get_instance()
        abs_qst = qs.abs_qst

        amended_args = list(args) + [abs_qst]
        res = eval_jaxpr(self)(*amended_args)

        if isinstance(res, tuple):
            new_abs_qst = res[-1]
            res = res[:-1]
        else:
            new_abs_qst = res
            res = None
        qs.abs_qst = new_abs_qst
        return res

    def count_ops(
        self,
        *args: Any,
        meas_behavior: str,
        callback_threshold: int | None = None,
    ) -> Any:
        """Return an operation count dict for this Jaspr evaluated on *args*."""
        from qrisp.jasp.evaluation_tools import profile_jaspr

        return profile_jaspr(self, "count_ops", meas_behavior, callback_threshold=callback_threshold)(*args)

    def depth(
        self,
        *args: Any,
        meas_behavior: str,
        max_qubits: int = 1024,
        callback_threshold: int | None = None,
    ) -> Any:
        """Return the circuit depth of this Jaspr evaluated on *args*."""
        from qrisp.jasp.evaluation_tools import profile_jaspr

        return profile_jaspr(
            self,
            "depth",
            meas_behavior,
            max_qubits=max_qubits,
            callback_threshold=callback_threshold,
        )(*args)

    def num_qubits(
        self,
        *args: Any,
        meas_behavior: str,
        max_allocations: int = 1000,
        callback_threshold: int | None = None,
    ) -> Any:
        """Return the peak qubit count of this Jaspr evaluated on *args*."""
        from qrisp.jasp.evaluation_tools import profile_jaspr

        return profile_jaspr(
            self,
            "num_qubits",
            meas_behavior,
            max_allocations=max_allocations,
            callback_threshold=callback_threshold,
        )(*args)

    def embedd(self, *args: Any, name: str | None = None, inline: bool = False) -> Any:
        """Embed this Jaspr into the current tracing context, optionally JIT-wrapping it."""
        from qrisp.jasp import TracingQuantumSession, get_last_equation

        qs = TracingQuantumSession.get_instance()
        abs_qst = qs.abs_qst

        amended_args = list(args) + [abs_qst]
        if not inline:
            res = jax.jit(eval_jaxpr(self))(*amended_args)

            eqn = get_last_equation()

            eqn.params["jaxpr"] = self
            if name is not None:
                eqn.params["name"] = name
        else:
            res = eval_jaxpr(self)(*amended_args)

        if isinstance(res, tuple):
            new_abs_qst = res[-1]
            res = res[:-1]
        else:
            new_abs_qst = res
            res = None
        qs.abs_qst = new_abs_qst
        return res

    def qjit(self, *args: Any, function_name: str = "jaspr_function", device: Any = None) -> Any:
        """
        Leverages the Catalyst pipeline to compile a QIR representation of
        this function and executes that function using the Catalyst QIR runtime.
        Requires the Catalyst package to be installed (``pip install qrisp[catalyst]``).

        Parameters
        ----------
        *args : iterable
            The arguments to call the function with.
        function_name : str, optional
            The name given to the compiled function in the QIR module.
            The default is ``"jaspr_function"``.
        device : object
            The `PennyLane device <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/devices.html>`_ to execute the function.
            The default device is `"lightning.qubit" <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_qubit/device.html>`_,
            a fast state-vector qubit simulator.

        Returns
        -------
            The values returned by the compiled, executed function.

        """
        from qrisp.jasp.evaluation_tools.catalyst_interface import (
            jaspr_to_catalyst_qjit,
        )

        qjit_obj = jaspr_to_catalyst_qjit(self, function_name=function_name, device=device)
        if qjit_obj.compiled_function is None:
            raise RuntimeError("Catalyst compilation produced no compiled function")
        res = qjit_obj.compiled_function(*args)
        if not isinstance(res, (tuple, list)):
            return res
        if len(res) == 1:
            return res[0]
        return res

    # LRU cache controlled by QRISP_COMPILATION_CACHE_SIZE env var
    @classmethod
    @qrisp_lru_compilation_cache
    def from_cache(cls, closed_jaxpr: ClosedJaxpr) -> "Jaspr":
        """
        Construct a :class:`Jaspr` from a :class:`~jax.extend.core.ClosedJaxpr`,
        caching the result so that repeated calls with the same argument are free.

        :func:`remove_redundant_allocations` is run on the newly created instance to
        clean up any trivially unused qubit allocations produced during tracing.

        Parameters
        ----------
        closed_jaxpr : ClosedJaxpr
            The closed jaxpr to convert.

        Returns
        -------
        Jaspr
            The corresponding :class:`Jaspr` instance.
        """
        res = Jaspr(jaxpr=closed_jaxpr.jaxpr, consts=closed_jaxpr.consts)
        remove_redundant_allocations(res)
        return res

    def update_eqns(self, eqns: list) -> "Jaspr":
        """Return a copy of this Jaspr with its equation list replaced by *eqns*."""
        return Jaspr(
            constvars=list(self.constvars),
            invars=list(self.invars),
            outvars=list(self.outvars),
            eqns=list(eqns),
            consts=list(self.consts),
            debug_info=self.debug_info,
        )

    def to_qir(self) -> str:
        """
        Compiles the Jaspr to QIR using the `Catalyst framework <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__.
        Requires the Catalyst package to be installed (``pip install qrisp[catalyst]``).

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

    def to_mlir(self, lower_stablehlo: bool = False) -> Any:
        """
        Compiles the Jaspr to an xDSL module using the Jasp Dialect.
        Requires the xDSL package to be installed (``pip install qrisp[xdsl]``).

        .. note::

            An xDSL module can be visualized via:

            ::

                print(xdsl_module)

            and serialized to a string using:

            ::

                from xdsl.printer import Printer
                Printer().print_op(xdsl_module)

        Parameters
        ----------
        lower_stablehlo : bool, optional
            If True, runs additional MLIR passes to lower StableHLO operations
            (like arithmetic and data operations) to lower-level dialects such
            as linalg, arith, and tensor. StableHLO control flow involving
            quantum types is preserved and rewritten to SCF by xDSL.
            The default is False.

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

        return jaspr_to_mlir(self, lower_stablehlo)

    def to_catalyst_mlir(self) -> str | None:
        """
        Compiles the Jaspr to MLIR using the `Catalyst dialect <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__.
        Requires the Catalyst package to be installed (``pip install qrisp[catalyst]``).

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

    def to_qasm(self, *args: Any) -> str:
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

    def to_catalyst_jaxpr(self) -> Any:
        """
        Compiles the jaspr to the corresponding `Catalyst jaxpr <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__.
        Requires the Catalyst package to be installed (``pip install qrisp[catalyst]``).

        Returns
        -------
        object
            A ClosedJaxpr-like object using Catalyst primitives.

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


def make_jaxpr_mod(
    fun: Callable,
    static_argnums: int | Sequence[int] = (),
    return_shape: bool = False,
    abstracted_axes: Any = None,
) -> Callable:
    """
    Creates a function that produces the jaxpr of a traced function.

    This is a modified version of JAX's ``make_jaxpr`` that supports
    ``return_shape=True`` even when the function returns custom abstract
    types that don't have ``shape``/``dtype`` attributes (such as
    ``AbstractQuantumState``).

    The interface is identical to ``jax.make_jaxpr``.

    Parameters
    ----------
    fun : Callable
        The function whose jaxpr is to be computed.
    static_argnums : int or Sequence[int], optional
        Indices of arguments that should be treated as static (not traced).
        Default is ``()``.
    return_shape : bool, optional
        If True, the returned function produces a tuple ``(jaxpr, out_tree)``
        where ``out_tree`` is a PyTreeDef representing the structure of the
        output. This can be used to reconstruct PyTree objects from flat
        output lists using ``jax.tree_util.tree_unflatten``.
        Default is False.
    abstracted_axes : optional
        Specification for which axes to abstract over. Default is None.

    Returns
    -------
    Callable
        A function that, when called with example arguments, returns either:
        - A ClosedJaxpr representation of ``fun`` (if ``return_shape=False``)
        - A tuple ``(ClosedJaxpr, out_tree)`` (if ``return_shape=True``)

    Notes
    -----
    JAX's native ``make_jaxpr(return_shape=True)`` fails on custom abstract
    types because it tries to create ``ShapeDtypeStruct`` objects from the
    outputs, which requires ``shape`` and ``dtype`` attributes. This function
    avoids that by using ``jit(...).trace()`` to directly access the output
    tree structure.

    Examples
    --------
    >>> def f(x):
    ...     return {"a": x + 1, "b": x * 2}
    >>> jaxpr, out_tree = make_jaxpr_mod(f, return_shape=True)(1.0)
    >>> # out_tree can be used with tree_unflatten to reconstruct the dict
    """

    def jaxpr_creator(*args, **kwargs):
        if not return_shape:
            return make_jaxpr(fun, static_argnums=static_argnums, abstracted_axes=abstracted_axes)(*args, **kwargs)

        # Use jit(...).trace() directly to get access to _out_tree.
        # This avoids JAX's make_jaxpr return_shape logic which fails on
        # custom abstract types that don't have shape/dtype attributes.
        traced = jax.jit(fun, static_argnums=static_argnums, abstracted_axes=abstracted_axes).trace(*args, **kwargs)

        # Extract the jaxpr, handling constants if needed (same logic as JAX's make_jaxpr).
        if traced._num_consts:
            consts, _ = split_list(cast(list, traced._args_flat), [traced._num_consts])
            jaxpr_ = part_eval.convert_invars_to_constvars(traced.jaxpr.jaxpr, traced._num_consts)
            closed_jaxpr = ClosedJaxpr(jaxpr_, consts)
        else:
            closed_jaxpr = traced.jaxpr

        return closed_jaxpr, traced._out_tree

    return jaxpr_creator


def make_jaspr(
    fun: Callable,
    flatten_envs: bool = True,
    return_shape: bool = False,
    **jax_kwargs: Any,
) -> Callable:
    """
    Creates a function that returns the Jaspr representation of a quantum function.

    This function is analogous to JAX's ``make_jaxpr``, but produces a Jaspr
    (a Jaxpr enhanced with quantum primitives) from a Qrisp quantum function.

    Parameters
    ----------
    fun : Callable
        The quantum function whose Jaspr is to be computed.
    flatten_envs : bool, optional
        If True (default), flatten quantum environments in the resulting Jaspr.
    return_shape : bool, optional
        If True, the returned function produces a tuple ``(jaspr, out_tree)``
        where ``out_tree`` is a PyTreeDef representing the structure of the
        output of ``fun``. This can be used to reconstruct PyTree objects
        from flat output lists using ``jax.tree_util.tree_unflatten``.
        Default is False.
    **jax_kwargs
        Additional keyword arguments passed to ``jax.make_jaxpr``, such as
        ``static_argnums``.

    Returns
    -------
    Callable
        A function that, when called with example arguments, returns either:

        - A :class:`Jaspr` representation of ``fun`` (if ``return_shape=False``)
        - A tuple ``(Jaspr, out_tree)`` (if ``return_shape=True``), where
          ``out_tree`` is a PyTreeDef that can be used with ``tree_unflatten``

    Examples
    --------

    **Basic quantum circuit with measurement**

    Create a Jaspr for a simple Bell state circuit:

    ::

        from qrisp import QuantumVariable, h, cx, measure
        from qrisp.jasp import make_jaspr

        def bell_state():
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])
            return measure(qv)

        jaspr = make_jaspr(bell_state)()
        result = jaspr()  # Returns 0 or 3 with equal probability

    **Parameterized quantum circuit**

    Create a Jaspr with parameterized gates that can be executed with different
    parameters:

    ::

        from qrisp import QuantumVariable, h, p, measure
        from qrisp.jasp import make_jaspr

        def rotation_circuit(angle):
            qv = QuantumVariable(1)
            h(qv)
            p(angle, qv)
            return measure(qv)

        jaspr = make_jaspr(rotation_circuit)(0.5)
        result1 = jaspr(0.5)  # Execute with angle=0.5
        result2 = jaspr(1.0)  # Execute with angle=1.0

    **Using return_shape for PyTree reconstruction**

    Retrieve the output tree structure alongside the Jaspr for reconstructing
    complex return values:

    ::

        from qrisp import QuantumVariable, h, cx, x, measure
        from qrisp.jasp import make_jaspr
        from jax.tree_util import tree_unflatten, tree_flatten

        def multi_output_circuit():
            qa = QuantumVariable(2)
            qb = QuantumVariable(2)
            h(qa[0])
            cx(qa[0], qa[1])
            x(qb)
            return measure(qa), measure(qb)

        jaspr, out_tree = make_jaspr(multi_output_circuit, return_shape=True)()
        result_a, result_b = jaspr()

        # Use out_tree to reconstruct the original tuple structure
        flat_results, _ = tree_flatten((result_a, result_b))
        reconstructed = tree_unflatten(out_tree, flat_results)

    """
    # NOTE: Imported locally to avoid circular imports.
    from qrisp import recursive_qv_search
    from qrisp.jasp import TracingQuantumSession, check_for_tracing_mode

    # The amended function receives an extra leading keyword argument (the abstract
    # quantum state), so any caller-supplied static_argnums must be shifted by one.
    adjusted_jax_kwargs = dict(jax_kwargs)
    if "static_argnums" in adjusted_jax_kwargs:
        sa = adjusted_jax_kwargs["static_argnums"]
        if isinstance(sa, int):
            adjusted_jax_kwargs["static_argnums"] = sa + 1
        else:
            adjusted_jax_kwargs["static_argnums"] = type(sa)(x + 1 for x in sa)

    def jaspr_creator(*args, **kwargs):
        qs = TracingQuantumSession.get_instance()

        # Close any tracing quantum sessions not properly closed due to prior errors.
        if not check_for_tracing_mode():
            while qs.abs_qst is not None:
                qs.conclude_tracing()

        # This function will be traced by JAX. The abstract quantum state is passed
        # as an extra keyword argument so JAX can track it through the trace.
        def amended_function(*args, **kwargs):
            abs_qst = kwargs[10 * "~"]
            del kwargs[10 * "~"]

            qs.start_tracing(abs_qst)

            # QuantumVariables in the signature went through JAX's
            # flatten/unflatten procedure, so their copies are not registered in any
            # QuantumSession yet — register them now.
            arg_qvs = recursive_qv_search(args)
            for qv in arg_qvs:
                qs.register_qv(qv, None)

            try:
                res = fun(*args, **kwargs)
            except Exception:
                qs.conclude_tracing()
                raise

            res_qc = qs.conclude_tracing()
            return res, res_qc

        amended_kwargs = dict(kwargs)
        amended_kwargs[10 * "~"] = AbstractQuantumState()

        static_argnums = adjusted_jax_kwargs.get("static_argnums", ())
        abstracted_axes = adjusted_jax_kwargs.get("abstracted_axes", None)

        result = make_jaxpr_mod(
            amended_function,
            static_argnums=static_argnums,
            return_shape=return_shape,
            abstracted_axes=abstracted_axes,
        )(*args, **amended_kwargs)

        user_out_tree: Any = None
        if return_shape:
            closed_jaxpr, full_out_tree = result
            # full_out_tree is a PyTreeDef for (res, res_qc); take the first child (res).
            user_out_tree = full_out_tree.children()[0]
        else:
            closed_jaxpr = result

        # Collect environments: quantum environments become primitives that "call"
        # a sub-Jaspr rather than enter/exit pairs.
        jaspr = Jaspr.from_cache(collect_environments(closed_jaxpr))

        if flatten_envs:
            jaspr = jaspr.flatten_environments()

        if return_shape:
            return jaspr, user_out_tree
        return jaspr

    return jaspr_creator


def check_aval_equivalence(invars_1, invars_2) -> bool:
    """Return True if every paired invar has the same abstract-value type."""
    return all(type(v1.aval) is type(v2.aval) for v1, v2 in zip(invars_1, invars_2))


def remove_redundant_allocations(closed_jaxpr: ClosedJaxpr) -> None:
    """
    Optimise a Jaspr in-place by removing redundant qubit allocations.

    A ``jasp.create_qubits`` equation is considered redundant when its output
    ``QubitArray`` is not returned by the function *and* every use of that array
    is limited to "free" primitives (``jasp.get_size``, ``jasp.delete_qubits``)
    that can be resolved from the allocation parameters alone, without requiring
    real qubits.

    The transformation proceeds in three phases:

    1. **Build a usage map** — record which equation consumes each variable.
    2. **Plan removals** — mark redundant ``create_qubits`` equations and their
       dependent ``get_size`` / ``delete_qubits`` equations for removal, and
       record variable replacements that re-wire the quantum-state flow.
    3. **Apply** — rewrite the equation list and output-variable list using the
       replacement map.

    Parameters
    ----------
    closed_jaxpr : ClosedJaxpr
        The Jaspr (or ClosedJaxpr) to optimise.  Modified in-place; nothing is
        returned.
    """

    jaxpr = closed_jaxpr.jaxpr
    eqns = jaxpr.eqns

    # 1. Build usage map: var → list of equations that consume it.
    usages: dict = defaultdict(list)
    for eqn in eqns:
        for var in eqn.invars:
            if not isinstance(var, (DropVar, Literal)):
                usages[var].append(eqn)

    # Variables appearing in the function's outputs cannot be optimised away.
    returned_vars = {var for var in jaxpr.outvars if not isinstance(var, (DropVar, Literal))}

    replacements: dict = {}
    eqns_to_remove: set = set()

    # Primitives whose results can be computed without the physical qubits.
    allowed_primitives = {"jasp.get_size", "jasp.delete_qubits"}

    def _plan_dependent_removals(out_qa, create_eqn) -> None:
        """Schedule get_size / delete_qubits equations that depend on out_qa."""
        if isinstance(out_qa, DropVar) or out_qa not in usages:
            return
        for u_eqn in usages[out_qa]:
            eqns_to_remove.add(id(u_eqn))
            if u_eqn.primitive.name == "jasp.get_size":
                # Forward the original size argument instead of querying qubits.
                replacements[u_eqn.outvars[0]] = create_eqn.invars[0]
            elif u_eqn.primitive.name == "jasp.delete_qubits":
                # Bypass the cleanup by passing its input circuit straight through.
                replacements[u_eqn.outvars[0]] = u_eqn.invars[1]

    # 2. Identify and plan removals.
    for eqn in eqns:
        if eqn.primitive.name != "jasp.create_qubits":
            continue

        out_qa = eqn.outvars[0]
        out_qc = eqn.outvars[1]

        if out_qa in returned_vars:
            continue

        is_redundant = (
            isinstance(out_qa, DropVar)
            or out_qa not in usages
            or all(u.primitive.name in allowed_primitives for u in usages[out_qa])
        )

        if not is_redundant:
            continue

        eqns_to_remove.add(id(eqn))
        # Bypass the allocation: re-wire output circuit → input circuit.
        replacements[out_qc] = eqn.invars[1]
        _plan_dependent_removals(out_qa, eqn)

    if not eqns_to_remove:
        return

    # 3. Apply replacements.
    def resolve(var):
        if isinstance(var, Literal):
            return var
        try:
            while var in replacements:
                var = replacements[var]
        except TypeError:
            # Guard against any unhashable variable types.
            pass
        return var

    new_eqns = []
    for eqn in eqns:
        if id(eqn) not in eqns_to_remove:
            eqn.invars[:] = [resolve(v) for v in eqn.invars]
            new_eqns.append(eqn)

    jaxpr.eqns[:] = new_eqns
    jaxpr.outvars[:] = [resolve(var) for var in jaxpr.outvars]
