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


class ProcessedMeasurement:
    """
    Placeholder object used to represent the result of classical post-processing
    of measurement results in the QuantumCircuit representation.
    
    Since QuantumCircuit objects cannot represent classical computation on measurement
    results, this class serves as a placeholder to emulate the computation without
    actually performing it.
    """
    pass


def make_qc_extraction_eqn_evaluator(qc):

    def qc_extraction_eqn_evaluator(eqn, context_dic):
        """
        Equation evaluator for converting a Jaspr to a QuantumCircuit.
        
        This evaluator processes Jaxpr equations and builds a QuantumCircuit representation.
        It handles quantum primitives directly and creates ProcessedMeasurement placeholders
        for classical post-processing operations.
        
        Parameters
        ----------
        eqn : JaxprEqn
            The equation to evaluate.
        context_dic : dict
            Context dictionary mapping variables to their values.
        
        Returns
        -------
        bool or None
            Returns True to indicate default evaluation should be used,
            False to skip default evaluation, or None if the equation was fully processed.
        """
        # Import here to avoid circular imports
        from qrisp import Clbit
        from qrisp.jasp import (
            Jaspr,
            extract_invalues,
            insert_outvalues,
            QuantumPrimitive,
            ParityOperation
        )
        from qrisp.jasp.interpreter_tools.interpreters import pjit_to_gate, cond_to_cl_control
        print(eqn.primitive.name)
        invalues = extract_invalues(eqn, context_dic)
        if eqn.primitive.name == "jit" and isinstance(
            eqn.params["jaxpr"], Jaspr
        ):
            return pjit_to_gate(eqn, context_dic, qc_extraction_eqn_evaluator)
        elif eqn.primitive.name == "cond":
            return cond_to_cl_control(eqn, context_dic, qc_extraction_eqn_evaluator)
        elif eqn.primitive.name == "jasp.parity":
            res = Clbit("cb_" + str(len(qc.clbits)))
            qc.clbits.insert(0, res)
            qc.append(ParityOperation(len(invalues)), clbits = invalues + [res])
            insert_outvalues(eqn, context_dic, res)
        elif eqn.primitive.name == "while":
            return True
        elif eqn.primitive.name == "convert_element_type":
            if isinstance(context_dic[eqn.invars[0]], (ProcessedMeasurement, Clbit)):
                context_dic[eqn.outvars[0]] = context_dic[eqn.invars[0]]
                return
            elif isinstance(context_dic[eqn.invars[0]], list) and isinstance(
                context_dic[eqn.invars[0]][0], (ProcessedMeasurement, Clbit)
            ):
                context_dic[eqn.outvars[0]] = context_dic[eqn.invars[0]]
                return
            return True
        else:
            for val in invalues:
                if isinstance(val, list) and len(val):
                    if isinstance(val[0], (ProcessedMeasurement, Clbit)):
                        break
                elif isinstance(val, (ProcessedMeasurement, Clbit)):
                    break
            else:
                if isinstance(eqn.primitive, QuantumPrimitive):
                    outvalues = eqn.primitive.impl(*invalues, **eqn.params)
                    insert_outvalues(eqn, context_dic, outvalues)
                    return
                else:
                    return True
            
        if len(eqn.outvars) == 0:
            return
        elif len(eqn.outvars) == 1 and not eqn.primitive.multiple_results:
            outvalues = ProcessedMeasurement()
        elif len(eqn.outvars) >= 1:
            outvalues = [ProcessedMeasurement() for _ in range(len(eqn.outvars))]
        
        insert_outvalues(eqn, context_dic, outvalues)
        
    return qc_extraction_eqn_evaluator


def jaspr_to_qc(jaspr, *args):
    """
    Converts a Jaspr into a QuantumCircuit if applicable.
    
    Circuit conversion of algorithms involving realtime computations is not possible.
    Any computations that perform classical postprocessing of measurements
    can not be reflected within the QuantumCircuit object itself and will
    generate an object of type ``ProcessedMeasurement``. These objects hold
    no further information and are simply used as placeholders to emulate
    the computation.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr object to convert.
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

        qb_list, qc = jaspr_to_qc(jaspr, 2)
        print(qc)
        # Yields
        # qb_0: ──■───────
        #       ┌─┴─┐┌───┐
        # qb_1: ┤ X ├┤ T ├
        #       └───┘└───┘
        
    To demonstrate the behavior under measurement post-processing, we build
    a similar script:
        
    ::
        
        from qrisp.jasp.interpreter_tools.interpreters import ProcessedMeasurement
        
        def example_function(i):
        
            qf = QuantumFloat(i)
            cx(qf[0], qf[1])
            t(qf[1])
            
            meas_res = measure(qf)
            # Perform classical post processing
            meas_res *= 2
            return meas_res
        
        jaspr = make_jaspr(example_function)(2)
        
        meas_res, qc = jaspr_to_qc(jaspr, 2)
        print(isinstance(meas_res, ProcessedMeasurement))
        # True
    """
    from qrisp import QuantumCircuit
    from qrisp.jasp import eval_jaxpr

    qc = QuantumCircuit()
    ammended_args = list(args) + [qc]
    
    if len(ammended_args) != len(jaspr.invars):
        raise Exception(
            "Supplied invalid number of arguments to Jaspr.to_qc (please exclude any static arguments, in particular callables)"
        )

    res = eval_jaxpr(jaspr, eqn_evaluator=make_qc_extraction_eqn_evaluator(qc))(*(ammended_args))

    return res
