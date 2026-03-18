from qrisp import *
import jax.numpy as jnp
from qrisp.jasp import jrange, qache
from qrisp.core import x, cx, QuantumVariable, mcx
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control
import numpy as np

@custom_control
def gidney_adder(a, b, c_in=None, c_out=None, ctrl=None):
    """
    Unified Gidney adder compatible with standard Python and JAX modes.
    """
    

    def extract_boolean_digit(integer, digit):
        if isinstance(integer, (int, np.integer)):
            return jnp.bool(integer >> digit & 1)
        return jnp.bool((integer >> digit) & 1)


    if not isinstance(a, QuantumVariable):
        if isinstance(b, list):
            n = len(b)
        else:
            n = b.size
            
        if isinstance(a, int):
            a = jnp.array(a, dtype=jnp.int64)
            

        with control(n > 1):
            gidney_anc = QuantumVariable(n - 1, name="gidney_anc*")
            
            i = 0
            # Initial Toffoli
            with control(extract_boolean_digit(a, i)):
                if ctrl is None:
                    cx(b[i], gidney_anc[i])
                else:
                    mcx([ctrl, b[i]], gidney_anc[i], method="gidney")
            
            # Left part of the V shape
            for j in jrange(n - 2):
                i = j + 1
                cx(gidney_anc[i - 1], b[i])
                with control(extract_boolean_digit(a, i)):
                    if ctrl is None:
                        x(gidney_anc[i - 1])
                    else:
                        cx(ctrl, gidney_anc[i - 1])
                mcx([gidney_anc[i - 1], b[i]], gidney_anc[i], method="gidney")
                with control(extract_boolean_digit(a, i)):
                    if ctrl is None:
                        x(gidney_anc[i - 1])
                    else:
                        cx(ctrl, gidney_anc[i - 1])
                cx(gidney_anc[i - 1], gidney_anc[i])
            
            # Tip of the V shape
            cx(gidney_anc[n - 2], b[n - 1])
            
            # Right part of the V shape
            for j in jrange(n - 2):
                i = n - j - 2
                cx(gidney_anc[i - 1], gidney_anc[i])
                if ctrl is not None:
                    with control(extract_boolean_digit(a, i)):
                        cx(ctrl, gidney_anc[i - 1])
                    mcx([gidney_anc[i - 1], b[i]], gidney_anc[i], method="gidney_inv")
                    with control(extract_boolean_digit(a, i)):
                        cx(ctrl, gidney_anc[i - 1])
                else:
                    with control(extract_boolean_digit(a, i)):
                        x(gidney_anc[i - 1])
                    mcx([gidney_anc[i - 1], b[i]], gidney_anc[i], method="gidney_inv")
                    with control(extract_boolean_digit(a, i)):
                        x(gidney_anc[i - 1])
            
            # Final Toffoli
            with control(extract_boolean_digit(a, 0)):
                if ctrl is None:
                    cx(b[0], gidney_anc[0])
                else:
                    mcx([ctrl, b[0]], gidney_anc[0], method="gidney_inv")
            gidney_anc.delete()
            
        # CX gates at the right side of the circuit
        for i in jrange(n):
            with control(extract_boolean_digit(a, i)):
                if ctrl is not None:
                    cx(ctrl, b[i])
                else:
                    x(b[i])
        return


    def c_inv(c):
        if c == "0": return "1"
        if c == "1": return "0"
        raise Exception

    if isinstance(b, list):
        if len(a) != len(b):
            raise Exception("Tried to call Gidney adder with inputs of unequal length")
        if c_out is not None:
            if isinstance(c_out, QuantumBool): c_out = c_out[0]
            b = list(b) + [c_out]
        if len(b) == 1:
            if ctrl is not None:
                mcx([ctrl, a[0]], b)
            else:
                cx(a[0], b[0])
            if c_in is not None:
                if isinstance(c_in, QuantumBool): c_in = c_in[0]
                if ctrl is not None: cx(c_in, b[0])
                else: mcx([c_in, b[0]], b)
            return

        if ctrl is not None:
            gidney_control_anc = QuantumBool(name="gidney_control_anc*", qs=b[0].qs())
        
        gidney_anc = QuantumVariable(len(b) - 1, name="gidney_anc*", qs=b[0].qs())
        
        for i in range(len(b) - 1):
            if i != 0:
                mcx([a[i], b[i]], gidney_anc[i], method="gidney")
                cx(gidney_anc[i - 1], gidney_anc[i])
            elif c_in is not None:
                cx(c_in, b[i])
                cx(c_in, a[i])
                mcx([a[i], b[i]], gidney_anc[i], method="gidney")
                cx(c_in, gidney_anc[i])
            else:
                mcx([a[i], b[i]], gidney_anc[i], method="gidney")
            if i != len(b) - 2:
                cx(gidney_anc[i], a[i + 1])
                cx(gidney_anc[i], b[i + 1])
        
        if ctrl is not None:
            mcx([ctrl, gidney_anc[-1]], b[-1])
            mcx([ctrl, a[-1]], gidney_control_anc[0], method="gidney")
            cx(gidney_control_anc[0], b[-1])
            mcx([ctrl, a[-1]], gidney_control_anc[0], method="gidney_inv")
        else:
            cx(gidney_anc[-1], b[-1])
            
        with invert():
            for i in range(len(b) - 1):
                if i != 0:
                    if i != len(b) - 1:
                        cx(gidney_anc[i - 1], a[i])
                    if ctrl is not None:
                        if i != len(b) - 1:
                            cx(gidney_anc[i - 1], b[i])
                        mcx([ctrl, a[i]], gidney_control_anc[0], method="gidney")
                        cx(gidney_control_anc[0], b[i])
                        mcx([ctrl, a[i]], gidney_control_anc[0], method="gidney_inv")
                    mcx([a[i], b[i]], gidney_anc[i], method="gidney")
                    cx(gidney_anc[i - 1], gidney_anc[i])
                elif c_in is not None:
                    cx(c_in, a[i])
                    if ctrl is not None:
                        cx(c_in, b[i])
                        mcx([ctrl, c_in], gidney_control_anc[0], method="gidney")
                        cx(gidney_control_anc[0], b[i])
                        mcx([ctrl, c_in], gidney_control_anc[0], method="gidney_inv")
                    mcx([a[i], b[i]], gidney_anc[i], method="gidney")
                    cx(c_in, gidney_anc[i])
                else:
                    if ctrl is not None:
                        mcx([ctrl, a[i]], gidney_control_anc[0], method="gidney")
                        cx(gidney_control_anc[0], b[i])
                        mcx([ctrl, a[i]], gidney_control_anc[0], method="gidney_inv")
                    mcx([a[i], b[i]], gidney_anc[i], method="gidney")
                    
        if ctrl is None:
            for i in range(len(a)):
                cx(a[i], b[i])
        else:
            gidney_control_anc.delete()
        gidney_anc.delete(verify=False)
        
    else:
        # If b is a QuantumVariable, use the JASP (JAX compatible) logic
        # Based on jasp_qq_gidney_adder
        
        # Determine sizes safely for tracing
        try:
            n_a = a.size
            n_b = b.size
            n = jnp.minimum(n_a, n_b)
            perform_incrementation = (n < n_b)
        except Exception:
            n_a = len(a)
            n_b = len(b)
            n = min(n_a, n_b)
            perform_incrementation = (n < n_b)

        if ctrl is not None:
            ctrl_anc = QuantumBool(name="gidney_anc_2*")

        with control(n > 1):
            gidney_anc = QuantumVariable(n - 1, name="gidney_anc*")
            
            i = 0
            # Perform the initial mcx
            mcx([a[i], b[i]], gidney_anc[i], method="gidney")
            
            # Perform the left part of the V-Shape
            for j in jrange(n - 2):
                i = j + 1
                cx(gidney_anc[i - 1], a[i])
                cx(gidney_anc[i - 1], b[i])
                mcx([a[i], b[i]], gidney_anc[i], method="gidney")
                cx(gidney_anc[i - 1], gidney_anc[i])
            
            # Handle the case that the addition target has more qubit than the control value.
            # Perform a 1-incrementation on the remainder if the carry out value is True
            with control(perform_incrementation):
                # Compute the carry out
                carry_out = QuantumBool()
                cx(gidney_anc[n - 2], a[n - 1])
                cx(gidney_anc[n - 2], b[n - 1])
                mcx([a[n - 1], b[n - 1]], carry_out[0], method="gidney")
                cx(gidney_anc[n - 2], carry_out[0])
                
                # Perform a controlled incrementation
                # Note: We use the unified gidney_adder recursively here
                ctrl_list = [carry_out[0]]
                if ctrl is not None:
                    ctrl_list.append(ctrl)
                
                with control(ctrl_list):
                    gidney_adder(1, b[n:], ctrl=None)
                
                # Uncompute the carry
                cx(gidney_anc[n - 2], carry_out[0])
                mcx([a[n - 1], b[n - 1]], carry_out[0], method="gidney_inv")
                carry_out.delete()
                cx(gidney_anc[n - 2], a[n - 1])
                if ctrl is not None:
                    cx(gidney_anc[n - 2], b[n - 1])
            
            # Tip and Right part logic handling
            if ctrl is not None:
                mcx([ctrl, gidney_anc[n - 2]], ctrl_anc[0], method="gidney")
                cx(ctrl_anc[0], b[n - 1])
                mcx([ctrl, gidney_anc[n - 2]], ctrl_anc[0], method="gidney_inv")
                
                mcx([ctrl, a[n - 1]], ctrl_anc[0], method="gidney")
                cx(ctrl_anc[0], b[n - 1])
                mcx([ctrl, a[n - 1]], ctrl_anc[0], method="gidney_inv")
            else:
                with control(jnp.logical_not(perform_incrementation)):
                    cx(gidney_anc[n - 2], b[n - 1])
                cx(a[n - 1], b[n - 1])
                
            # Perform the right part of the V shape
            for j in jrange(n - 2):
                i = n - j - 2
                cx(gidney_anc[i - 1], gidney_anc[i])
                mcx([a[i], b[i]], gidney_anc[i], method="gidney_inv")
                
                if ctrl is not None:
                    mcx([ctrl, a[i]], ctrl_anc[0], method="gidney")
                    cx(ctrl_anc[0], b[i])
                    mcx([ctrl, a[i]], ctrl_anc[0], method="gidney_inv")
                    cx(gidney_anc[i - 1], a[i])
                    cx(gidney_anc[i - 1], b[i])
                else:
                    cx(gidney_anc[i - 1], a[i])
                    cx(gidney_anc[i - 1], b[i])
            
            # The final uncomputation
            mcx([a[0], b[0]], gidney_anc[0], method="gidney_inv")
            gidney_anc.delete()
            
        # Handle the case where n == 1 but we still have bits to increment
        with control((n == 1) & perform_incrementation):
            ctrl_list = [a[0], b[0]]
            if ctrl is not None:
                ctrl_list.append(ctrl)
            with control(ctrl_list):
                gidney_adder(1, b[n:], ctrl=None)
            
        # Perform the CX gate at the top right of the circuit
        if ctrl is not None:
            mcx([ctrl, a[0]], ctrl_anc[0], method="gidney")
            cx(ctrl_anc[0], b[0])
            mcx([ctrl, a[0]], ctrl_anc[0], method="gidney_inv")
        else:
            cx(a[0], b[0])
            
        if ctrl is not None:
            ctrl_anc.delete()