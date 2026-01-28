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

import weakref

import jax

from qrisp.jasp.primitives import create_qubits, delete_qubits_p, quantum_gate_p
from qrisp.jasp.tracing_logic.dynamic_qubit_array import DynamicQubitArray
from qrisp.core.quantum_variable import QuantumVariable

from sympy import symbols

greek_letters = symbols(
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
)


class TracingQuantumSession:
    tr_qs_container = [None]
    abs_qc_stack = []
    qubit_cache_stack = []

    def __init__(self):

        self.abs_qc = None
        self.qv_list = []
        self.deleted_qv_list = []
        self.qubit_cache = None
        TracingQuantumSession.tr_qs_container.insert(0, self)
        self.qv_stack = []

    def start_tracing(self, abs_qc):
        self.abs_qc_stack.append(self.abs_qc)
        self.qubit_cache_stack.append(self.qubit_cache)

        self.abs_qc = abs_qc
        self.qubit_cache = {}

        self.qv_stack.append((self.qv_list, self.deleted_qv_list))
        self.qv_list = []
        self.deleted_qv_list = []

    def conclude_tracing(self):

        temp = self.abs_qc
        self.abs_qc = self.abs_qc_stack.pop(-1)
        self.qubit_cache = self.qubit_cache_stack.pop(-1)
        self.qv_list, self.deleted_qv_list = self.qv_stack.pop(-1)

        return temp

    def append(self, operation, qubits=[], clbits=[], param_tracers=[]):
        
        if not self.abs_qc._trace is jax.core.trace_ctx.trace:
            raise Exception(
                """Lost track of QuantumCircuit during tracing. This might have been caused by a missing quantum_kernel decorator or not using quantum prefix control (like q_fori_loop, q_cond). Please visit https://www.qrisp.eu/reference/Jasp/Quantum%20Kernel.html for more details"""
            )

        if len(clbits):
            raise Exception(
                "Tried to append Operation with non-zero classical bits in JAX mode."
            )

        from qrisp.core import QuantumVariable, QuantumArray
        from qrisp.jasp import jrange

        if isinstance(qubits[0], (QuantumVariable, DynamicQubitArray)):

            for i in jrange(qubits[0].size):
                self.append(
                    operation,
                    [qubits[j][i] for j in range(operation.num_qubits)],
                    param_tracers=param_tracers,
                )
            return

        elif isinstance(qubits[0], list):

            for i in range(len(qubits[0])):
                self.append(
                    operation,
                    [qubits[j][i] for j in range(operation.num_qubits)],
                    param_tracers=param_tracers,
                )
            return
        
        elif isinstance(qubits[0], QuantumArray):
            
            for i in range(1, len(qubits)):
                if not isinstance(qubits[i], QuantumArray):
                    raise Exception(f"Tried to apply multi-qubit gate to mixed qubit argument types (QuantumArray + {type(qubits[i])})")
                
                if qubits[i].shape != qubits[0].shape:
                    raise Exception("Tried to apply multi-qubit quantum gate to QuantumArrays of differing shape.")
            
            flattened_qubits = [qubits[i].flatten() for i in range(len(qubits))]
            
            for i in jrange(flattened_qubits[0].size):
                self.append(
                    operation,
                    [flattened_qubits[j][i] for j in range(operation.num_qubits)],
                    param_tracers=param_tracers,
                )
            return

        temp_op = operation.copy()

        temp_op.params = list(temp_op.params)
        for i in range(len(temp_op.params)):
            temp_op.params[i] = greek_letters[i]

        self.abs_qc = quantum_gate_p.bind(
            *([b for b in qubits] + param_tracers + [self.abs_qc]),
            gate = operation
        )

    def register_qv(self, qv, size):

        if not self.abs_qc._trace is jax.core.trace_ctx.trace:
            raise Exception(
                """Lost track of QuantumCircuit during tracing. This might have been caused by a missing quantum_kernel decorator or not using quantum prefix control (like q_fori_loop, q_cond). Please visit https://www.qrisp.eu/reference/Jasp/Quantum%20Kernel.html for more details"""
            )

        # Determine amount of required qubits
        if size is not None:
            qv.reg = self.request_qubits(size)
            
        # Register in the list of active quantum variable
        self.qv_list.append(qv)
        qv.qs = self

        QuantumVariable.live_qvs.append(weakref.ref(qv))
        qv.creation_time = int(QuantumVariable.creation_counter[0])
        QuantumVariable.creation_counter += 1
        
    def request_qubits(self, amount):
        qb_array_tracer, self.abs_qc = create_qubits(amount, self.abs_qc)
        return DynamicQubitArray(qb_array_tracer)
        

    def delete_qv(self, qv, verify=False):
        
        if not self.abs_qc._trace is jax.core.trace_ctx.trace:
            raise Exception(
                """Lost track of QuantumCircuit during tracing. This might have been caused by a missing quantum_kernel decorator or not using quantum prefix control (like q_fori_loop, q_cond). Please visit https://www.qrisp.eu/reference/Jasp/Quantum%20Kernel.html for more details"""
            )

        if verify == True:
            raise Exception("Tried to verify deletion in tracing mode.")

        # Check if quantum variable appears in this session
        if qv.name not in [qv.name for qv in self.qv_list]:
            raise Exception(
                "Tried to remove a non existent quantum variable from quantum session"
            )

        self.clear_qubits(qv.reg, verify)

        # Remove quantum variable from list
        for i in range(len(self.qv_list)):
            temp_qv = self.qv_list[i]

            if temp_qv.name == qv.name:
                self.qv_list.pop(i)
                break

        self.deleted_qv_list.append(qv)

    def clear_qubits(self, qubits, verify=False):

        self.abs_qc = delete_qubits_p.bind(qubits.tracer, self.abs_qc)

    @classmethod
    def release(cls):
        cls.tr_qs_container.pop(0)

    @classmethod
    def get_instance(cls):
        return cls.tr_qs_container[0]


tracing_qs_singleton = TracingQuantumSession()


def check_for_tracing_mode():
    return hasattr(jax._src.core.trace_ctx.trace, "frame")

def get_last_equation(i = -1):
    return jax._src.core.trace_ctx.trace.frame.tracing_eqns[i]()

def check_live(tracer):
    if tracer is None:
        return True
    return not not tracer._trace.main.jaxpr_stack
