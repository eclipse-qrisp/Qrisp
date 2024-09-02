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
from functools import lru_cache

from jax import make_jaxpr
from jax.core import Jaxpr, ClosedJaxpr, Literal

from qrisp.jisp.jisp_expression import invert_jispr, multi_control_jispr, collect_environments
from qrisp.jisp import AbstractQuantumCircuit, eval_jaxpr, pjit_to_gate, flatten_environments

class Jispr(Jaxpr):
    """
    The ``Jispr`` class enables an efficient representations of a wide variety 
    of (hybrid) algorithms. For many applications, the representation is agnostic
    to the scale of the problem, implying function calls with 10 or 10000 qubits 
    can be represented by the same object. The actual unfolding to a circuit-level
    description is outsourced to 
    `established, classical compilation infrastructure <https://mlir.llvm.org/>`_,
    implying state-of-the-art compilation speed can be reached.
    
    As a subtype of ``jax.core.Jaxpr``, Jisprs are embedded into the well matured 
    `Jax ecosystem <https://github.com/n2cholas/awesome-jax>`_,
    which facilitates the compilation of classical `real-time computation <https://arxiv.org/abs/2206.12950>`_
    using some of the most advanced libraries in the world such as 
    `CUDA <https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html>`_. 
    Especially `machine learning <https://ai.google.dev/gemma/docs/jax_inference>`_
    and other scientific computations tasks are particularly well supported.
    
    To get a better understanding of the syntax and semantics of Jaxpr (and with
    that also Jispr) please check `this link <https://jax.readthedocs.io/en/latest/jaxpr.html>`__.
    
    Similar to Jaxpr, Jispr objects represent (hybrid) quantum
    algorithms in the form of a `functional programming language <https://en.wikipedia.org/wiki/Functional_programming>`_
    in `SSA-form <https://en.wikipedia.org/wiki/Static_single-assignment_form>`_.
    
    It is possible to compile Jispr objects into QIR, which is facilitated by the
    `Catalyst framework <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__
    (check :meth:`qrisp.jisp.Jispr.to_qir` for more details).
    
    Qrisp scripts can be turned into Jispr objects by
    calling the ``make_jispr`` function, which has similar semantics as 
    `jax.make_jaxpr <https://jax.readthedocs.io/en/latest/_autosummary/jax.make_jaxpr.html>`_.
    
    ::
    
        from qrisp import *
        from qrisp.jisp import make_jispr
        
        def test_fun(i):
            
            qv = QuantumFloat(i, -1)
            x(qv[0])
            cx(qv[0], qv[i-1])
            meas_res = measure(qv)
            meas_res += 1
            return meas_res
            
        
        jispr = make_jispr(test_fun)(4)
        print(jispr)
    
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
            l:f32[] = convert_element_type[new_dtype=float32 weak_type=True] k
            m:f32[] = mul l 0.5
            n:f32[] = add m 1.0
          in (j, n) }
        

    A defining feature of the Jispr class is that the first input and the 
    first output are always of QuantumCircuit type. Therefore, Jispr objects always 
    represent some (hybrid) quantum operation.
    
    Qrisp comes with a built-in Jispr interpreter. For that you simply have to
    call the object like a function:
        
        
    >>> print(jispr(2))
    2.5
    >>> print(jispr(4))
    5.5
    """
    
    __slots__ = "permeability", "isqfree", "hashvalue"
    
    def __init__(self, *args, permeability = None, isqfree = None, **kwargs):
        
        if len(args) == 1:
            kwargs["jaxpr"] = args[0]
        
        if "jaxpr" in kwargs:
            jaxpr = kwargs["jaxpr"]

            self.hashvalue = hash(jaxpr)
        
            Jaxpr.__init__(self,
                           constvars = jaxpr.constvars,
                           invars = jaxpr.invars,
                           outvars = jaxpr.outvars,
                           eqns = jaxpr.eqns,
                           effects = jaxpr.effects,
                           debug_info = jaxpr.debug_info
                           )
        else:
            self.hashvalue = id(self)
            
            Jaxpr.__init__(self, **kwargs)
            
        self.permeability = {}
        if permeability is None:
            permeability = {}
        for var in self.constvars + self.invars + self.outvars:
            if isinstance(var, Literal):
                continue
            self.permeability[var] = permeability.get(var, None)
        
        self.isqfree = isqfree
            
        if not isinstance(self.invars[0].aval, AbstractQuantumCircuit):
            raise Exception(f"Tried to create a Jispr from data that doesn't have a QuantumCircuit as first argument (got {type(self.invars[0].aval)} instead)")
        
        if not isinstance(self.outvars[0].aval, AbstractQuantumCircuit):
            raise Exception(f"Tried to create a Jispr from data that doesn't have a QuantumCircuit as first entry of return type (got {type(self.outvars[0].aval)} instead)")
        
    def __hash__(self):
        
        return self.hashvalue
    
    def __eq__(self, other):
        if not isinstance(other, Jaxpr):
            return False
        return self.hashvalue == hash(other)
    
    def inverse(self):
        """
        Returns the inverse Jispr (if applicable). For Jispr that contain realtime
        computations or measurements, the inverse does not exist.

        Returns
        -------
        Jispr
            The daggered Jispr.
            
        Examples
        --------
        
        We create a simple script and inspect the daggered version:
        
        ::
            
            from qrisp import *
            from qrisp.jisp import make_jispr
            
            def example_function(i):
                
                qv = QuantumVariable(i)
                cx(qv[0], qv[1])
                t(qv[1])
                return qv
            
            jispr = make_jispr(example_function)(2)
            
            print(jispr.inverse())
            # Yields
            # { lambda ; a:QuantumCircuit b:i32[]. let
            #     c:QuantumCircuit d:QubitArray = create_qubits a b
            #     e:Qubit = get_qubit d 0
            #     f:Qubit = get_qubit d 1
            #     g:QuantumCircuit = t_dg c f
            #     h:QuantumCircuit = cx g e f
            #   in (h, d) }
        """
        return invert_jispr(self)
    
    def control(self, num_ctrl, ctrl_state = -1):
        """
        Returns the controlled version of the Jispr. The control qubits are added
        to the signature of the Jispr as the arguments after the QuantumCircuit.

        Parameters
        ----------
        num_ctrl : int
            The amount of controls to be added.
        ctrl_state : int of str, optional
            The control state on which to activate. The default is -1.

        Returns
        -------
        Jispr
            The controlled Jispr.
            
        Examples
        --------
        
        We create a simple script and inspect the controlled version:
            
        ::
            
            from qrisp import *
            from qrisp.jisp import make_jispr
            
            def example_function(i):
                
                qv = QuantumVariable(i)
                cx(qv[0], qv[1])
                t(qv[1])
                return qv
            
            jispr = make_jispr(example_function)(2)
            
            print(jispr.control(2))
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
        
        return multi_control_jispr(self, num_ctrl, ctrl_state)
    
    def to_qc(self, *args):
        """
        Converts the Jispr into a :ref:`QuantumCircuit` if applicable. Circuit 
        conversion of algorithms involving realtime computations is not possible.

        Parameters
        ----------
        *args : tuple
            The arguments to call the Jispr with.

        Returns
        -------
        :ref:`QuantumCircuit`
            The resulting QuantumCircuit.
        return_values : tuple
            The return values of the Jispr. QuantumVariable return types are
            returned as lists of Qubits.
            
        Examples
        --------

        We create a simple script and inspect the QuantumCircuit:
            
        ::
        
            from qrisp import *
            from qrisp.jisp import make_jispr
            
            def example_function(i):
                
                qv = QuantumVariable(i)
                cx(qv[0], qv[1])
                t(qv[1])
                return qv
            
            jispr = make_jispr(example_function)(2)
            
            qc, qb_list = jispr.to_qc(2)
            print(qc)
            # Yields
            # qb_0: ──■───────
            #       ┌─┴─┐┌───┐
            # qb_1: ┤ X ├┤ T ├
            #       └───┘└───┘

        """
        from qrisp import QuantumCircuit
        jispr = flatten_environments(self)
        
        def eqn_evaluator(primitive, *args, **kwargs):
            if primitive.name == "pjit":
                return pjit_to_gate(primitive, *args, **kwargs)
            else:
                return primitive.bind(*args, **kwargs)
        
        res = eval_jaxpr(jispr, eqn_evaluator = eqn_evaluator)(*([QuantumCircuit()] + list(args)))
        
        return res
    
    def eval(self, *args, eqn_evaluator = None):
        if eqn_evaluator is None:
            
            def eqn_evaluator(primitive, *args, **kwargs):
                return primitive.bind(*args, **kwargs)
        
        return eval_jaxpr(self, eqn_evaluator = eqn_evaluator)(*args)
        
    def flatten_environments(self):
        """
        Flattens all environments by applying the corresponding compilation 
        routines such that no more ``q_env`` primitives are left.

        Returns
        -------
        Jispr
            The Jispr with flattened environments.
            
        Examples
        --------
        
        We create a Jispr containing an :ref:`InversionEnvironment` and flatten:
            
        ::

        	def test_function(i):
        		qv = QuantumVariable(i)
        		
        		with invert():
        			t(qv[0])
        			cx(qv[0], qv[1])
        		
        		return qv
        
        	jispr = make_jispr(test_function)(2)
        	print(jispr)
        
        ::
        	
        	{ lambda ; a:QuantumCircuit b:i32[]. let
        		c:QuantumCircuit d:QubitArray = create_qubits a b
        		e:QuantumCircuit = q_env[
        		jispr={ lambda ; f:QuantumCircuit d:QubitArray. let
                  g:Qubit = get_qubit d 0
                  h:QuantumCircuit = t f g
                  i:Qubit = get_qubit d 1
                  j:QuantumCircuit = cx h g i
                in (j,) }
        		type=InversionEnvironment
        		] c d
        	  in (e, d) }
        
        You can see how the body of the :ref:`InversionEnvironment` is __collected__
        into another Jispr. This reflects the fact that at their core, 
        :ref:`QuantumEnvironment <QuantumEnvironment>` describe `higher-order 
        quantum functions <https://en.wikipedia.org/wiki/Higher-order_function>`_
        (ie. functions that operate on functions). In order to apply the 
        transformations induced by the QuantumEnvironment, we can call 
        ``Jispr.flatten_environments``:
        
        >>> print(jispr.flatten_environments)
        { lambda ; a:QuantumCircuit b:i32[]. let
            c:QuantumCircuit d:QubitArray = create_qubits a b
            e:Qubit = get_qubit d 0
            f:Qubit = get_qubit d 1
            g:QuantumCircuit = cx c e f
            h:QuantumCircuit = t_dg g e
          in (h, d) }
        
        We see that as expected, the order of the ``cx`` and the ``t`` gate has been switched and the ``t`` gate has been turned into a ``t_dg``.

        """
        return flatten_environments(self)
    
    def __call__(self, *args):
        
        if len(self.outvars) == 1:
            return None
        
        from qrisp.simulator import BufferedQuantumState
        args = [BufferedQuantumState()] + list(args)
        
        flattened_jispr = self.flatten_environments()
        
        def eqn_evaluator(primitive, *args, **kwargs):
            if primitive.name == "pjit":
                return pjit_to_gate(primitive, *args, **kwargs)
            else:
                return primitive.bind(*args, **kwargs)
        
        res = eval_jaxpr(flattened_jispr, eqn_evaluator = eqn_evaluator)(*args)
        
        if len(self.outvars) == 2:
            return res[1]
        else:
            return res[1:]
        
    def inline(self, *args):
        
        from qrisp.jisp import TracingQuantumSession
        
        qs = TracingQuantumSession.get_instance()
        abs_qc = qs.abs_qc
        
        res = eval_jaxpr(self)(*([abs_qc] + list(args)))
        
        if isinstance(res, tuple):
            new_abs_qc = res[0]
            res = res[1:]
        else:
            new_abs_qc = res
            res = None
        qs.abs_qc = new_abs_qc
        return res
        
    
    def qjit(self, *args, function_name = "jispr_function"):
        """
        Leverages the Catalyst pipeline to compile a QIR representation of 
        this function and executes that function using the Catalyst QIR runtime.

        Parameters
        ----------
        *args : iterable
            The arguments to call the function with.

        Returns
        -------
            The values returned by the compiled, executed function.

        """
        flattened_jispr = self.flatten_environments()
        
        from qrisp.jisp.catalyst_interface import jispr_to_catalyst_qjit
        qjit_obj = jispr_to_catalyst_qjit(flattened_jispr, function_name = function_name)(*args)
        return qjit_obj.compiled_function(*args)
    
    @classmethod
    @lru_cache(maxsize = int(1E5))
    def from_cache(cls, jaxpr):
        return Jispr(jaxpr = jaxpr)
    
    def to_qir(self, *args):
        """
        Compiles the Jispr to QIR using the `Catalyst framework <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__.

        Parameters
        ----------
        *args : tuple
            The arguments to call the Jispr with.

        Returns
        -------
        str
            The QIR string.
            
        Examples
        --------
        
        We create a simple script and inspect the QIR string:
            
        ::
        
            from qrisp import *
            from qrisp.jisp import make_jispr
            
            def example_function(i):
                
                qv = QuantumFloat(i)
                cx(qv[0], qv[1])
                t(qv[1])
                meas_res = measure(qv)
                meas_res += 1
                return meas_res
            
            jispr = make_jispr(example_function)(2)
            print(jispr.to_qir(4))
            
        Yields:
        
        ::
            
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
            
            define { ptr, ptr, i64 } @jit_jispr_function(ptr nocapture readnone %0, ptr nocapture readonly %1, i64 %2) local_unnamed_addr {
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
            
            define void @_catalyst_pyface_jit_jispr_function(ptr nocapture writeonly %0, ptr nocapture readonly %1) local_unnamed_addr {
              %.unpack = load ptr, ptr %1, align 8
              %.elt1.i = getelementptr inbounds { ptr, ptr, i64 }, ptr %.unpack, i64 0, i32 1
              %.unpack2.i = load ptr, ptr %.elt1.i, align 8
              %3 = tail call { ptr, ptr, i64 } @jit_jispr_function(ptr poison, ptr %.unpack2.i, i64 poison)
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
            
            define void @_catalyst_ciface_jit_jispr_function(ptr nocapture writeonly %0, ptr nocapture readonly %1) local_unnamed_addr {
              %.elt1 = getelementptr inbounds { ptr, ptr, i64 }, ptr %1, i64 0, i32 1
              %.unpack2 = load ptr, ptr %.elt1, align 8
              %3 = tail call { ptr, ptr, i64 } @jit_jispr_function(ptr poison, ptr %.unpack2, i64 poison)
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
        from qrisp.jisp.catalyst_interface import jispr_to_qir
        return jispr_to_qir(self.flatten_environments(), args)
    
    def to_mlir(self, *args):
        """
        Compiles the Jispr to MLIR using the `Catalyst dialect <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__.

        Parameters
        ----------
        *args : tuple
            The arguments to call the Jispr with.

        Returns
        -------
        str
            The MLIR string.
            
        Examples
        --------
        
        We create a simple script and inspect the MLIR string:
            
        ::
        
            from qrisp import *
            from qrisp.jisp import make_jispr
            
            def example_function(i):
                
                qv = QuantumFloat(i)
                cx(qv[0], qv[1])
                t(qv[1])
                meas_res = measure(qv)
                meas_res += 1
                return meas_res
            
            jispr = make_jispr(example_function)(2)
            print(jispr.to_mlir(4))
            
        ::
            
                        module @jispr_function {
              func.func public @jit_jispr_function(%arg0: tensor<i64>) -> tensor<i32> attributes {llvm.emit_c_interface} {
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
        from qrisp.jisp.catalyst_interface import jispr_to_mlir
        return jispr_to_mlir(self.flatten_environments(), args)
    
    def to_catalyst_jaxpr(self):
        """
        Compiles the Jispr to the corresponding `Catalyst jaxpr <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__.

        Parameters
        ----------
        *args : tuple
            The arguments to call the Jispr with.

        Returns
        -------
        jax.core.Jaxpr
            The Jaxpr using Catalyst primitives.
            
        Examples
        --------
        
        We create a simple script and inspect the Catalyst Jaxpr:
            
        ::
        
            from qrisp import *
            from qrisp.jisp import make_jispr
            
            def example_function(i):
                
                qv = QuantumFloat(i)
                cx(qv[0], qv[1])
                t(qv[1])
                meas_res = measure(qv)
                meas_res += 1
                return meas_res
            
            jispr = make_jispr(example_function)(2)
            
            print(jispr.to_catalyst_jaxpr())
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
            #   bp:i32[] = convert_element_type[new_dtype=int32 weak_type=False] s
            #   bq:i32[] = mul bp 1
            #   br:i32[] = add bq 1
            # in (t, e, br) }
        
        """
        from qrisp.jisp.catalyst_interface import jispr_to_catalyst_jaxpr
        return jispr_to_catalyst_jaxpr(self.flatten_environments())
    
    
    


def make_jispr(fun):
    from qrisp.jisp import AbstractQuantumCircuit, TracingQuantumSession
    def jispr_creator(*args, **kwargs):
        
        def ammended_function(abs_qc, *args, **kwargs):
            
            qs = TracingQuantumSession(abs_qc)
            
            res = fun(*args, **kwargs)
            res_qc = qs.abs_qc
            
            TracingQuantumSession.release()
            
            return res_qc, res
        
        jaxpr = make_jaxpr(ammended_function)(AbstractQuantumCircuit(), *args, **kwargs).jaxpr
        
        return recursive_convert(jaxpr)
    
    return jispr_creator

def recursive_convert(jaxpr):
    
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pjit" and isinstance(eqn.outvars[0].aval, AbstractQuantumCircuit):
            eqn.params["jaxpr"] = ClosedJaxpr(recursive_convert(eqn.params["jaxpr"].jaxpr), eqn.params["jaxpr"].consts)
    
    # We "collect" the QuantumEnvironments.
    # Collect means that the enter/exit statements are transformed into Jispr
    # which are subsequently called. Example:
        
    # from qrisp import *
    # from qrisp.jisp import *
    # import jax

    # def outer_function(x):
    #     qv = QuantumVariable(x)
    #     with QuantumEnvironment():
    #         cx(qv[0], qv[1])
    #         h(qv[0])
    #     return qv

    # jaxpr = make_jaxpr(outer_function)(2).jaxpr
    
    # This piece of code results in the following jaxpr
    
    # { lambda ; a:i32[]. let
    #     b:QuantumCircuit = qdef 
    #     c:QuantumCircuit d:QubitArray = create_qubits b a
    #     e:QuantumCircuit = q_env[stage=enter type=quantumenvironment] c
    #     f:Qubit = get_qubit d 0
    #     g:Qubit = get_qubit d 1
    #     h:QuantumCircuit = cx e f g
    #     i:Qubit = get_qubit d 0
    #     j:QuantumCircuit = h h i
    #     _:QuantumCircuit = q_env[stage=exit type=quantumenvironment] j
    #   in (d,) }
    
    jaxpr = collect_environments(jaxpr)
    
    return Jispr.from_cache(jaxpr)


def qjit(function):
    """
    Decorator to leverage the Jisp + Catalyst infrastructure to compile the given
    function to QIR and run it on the Catalyst QIR runtime.

    Parameters
    ----------
    function : callable
        A function performing Qrisp code.

    Returns
    -------
    callable
        A function executing the compiled code.
        
    Examples
    --------
    
    We write a simple function using the QuantumFloat quantum type and execute
    via ``qjit``:
        
    ::
        
        from qrisp import *
        from qrisp.jisp import qjit

        @qjit
        def test_fun(i):
            qv = QuantumFloat(i, -2)
            with invert():
                cx(qv[0], qv[qv.size-1])
                h(qv[0])
            meas_res = measure(qv)
            return meas_res + 3
            
    
    We execute the function a couple of times to demonstrate the randomness
    
    >>> test_fun(4)
    [array(5.25, dtype=float32)]
    >>> test_fun(5)
    [array(3., dtype=float32)]
    >>> test_fun(5)
    [array(7.25, dtype=float32)]

    """
    
    def jitted_function(*args):
        jispr = make_jispr(function)(*args)
        return jispr.qjit(*args, function_name = function.__name__)
    
    return jitted_function