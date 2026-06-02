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
from qrisp import (
    QuantumBool,
    invert,
    rz,
    h,
    mcx,
)
from qrisp.environments import conjugate, control
from qrisp.algorithms.gqsp.gqsp_angles import qsvt_angles
from qrisp.algorithms.gqsp.helper_functions import poly2cheb, _rescale_poly
from qrisp.block_encodings import BlockEncoding
from qrisp.jasp import jrange, q_cond
from qrisp.operators import QubitOperator, FermionicOperator
from jax import numpy as jnp
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import ArrayLike

def QSVT(
    H: BlockEncoding | FermionicOperator | QubitOperator,
    p: "ArrayLike",
    kind: Literal["Polynomial", "Chebyshev"] = "Polynomial",
    parity: Literal["odd", "even"] = "odd",
    rescale: bool = True,
) -> BlockEncoding:
    
    ALLOWED_KINDS = {"Polynomial", "Chebyshev"}
    if kind not in ALLOWED_KINDS:
        raise ValueError(
            f"Invalid kind specified: '{kind}'. "
            f"Allowed kinds are: {', '.join(ALLOWED_KINDS)}"
        )
    
    if isinstance(H, (QubitOperator, FermionicOperator)):
        H = BlockEncoding.from_operator(H)
    
    # Rescaling of the polynomial to account for scaling factor alpha of block-encoding
    if rescale:
        p = _rescale_poly(H.alpha, p, kind=kind)
    if kind == "Polynomial":
        p = poly2cheb(p)
    
    phi, alpha = qsvt_angles(p)
        
    print(phi)
    
    m = len(H._anc_templates)

    def reflection(args, phase):
        qubits = sum([arg.reg for arg in args[1:m + 1]], []) 
        with conjugate(mcx)(qubits, args[0], ctrl_state=0):
            rz(phase, args[0])

    def even(args):
        H.unitary(*args[1:])

    def odd(args):
        with invert():
            H.unitary(*args[1:])

    def new_unitary(*args):
        h(args[0]) 

        d = len(phi) - 1
                
        for i in jrange(0, d): 
            reflection(args, phase = 2 * phi[d-i])
            q_cond(i%2==0, even, odd, args) 
        reflection(args, phase=2 * phi[0])
            
        h(args[0])

    new_anc_templates = [QuantumBool().template()] + H._anc_templates
    new_alpha = alpha

    return BlockEncoding(new_alpha, new_anc_templates, new_unitary, is_hermitian = False)