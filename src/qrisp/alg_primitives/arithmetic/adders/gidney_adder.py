from qrisp import *
import jax.numpy as jnp
from qrisp.jasp import jrange, qache
from qrisp.core import x, cx, QuantumVariable, mcx
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control
import numpy as np

def gidney_adder(a, b, c_out=None, ctrl = None):
    """In-place adder as introduced in https://arxiv.org/abs/1709.06648
    
    Parameters
    ----------
    a : int or QuantumVariable
        The value that should be added.
    b : QuantumVariable or list[Qubit]
        The value that should be modified in the in-place addition.
    c_out : QuantumVariable, optional
        An optional carry out value. The default is None.

    Raises
    ------
    ValueError
        If the inputs are not valid quantum or classical types.
    
    Returns
    -------
    None
        The function modifies the second input in place.
    
    Examples
    --------
    Static mode with both quantum inputs:

    >>> from qrisp import QuantumFloat, gidney_adder
    >>> a = QuantumFloat(4)
    >>> b = QuantumFloat(4)
    >>> a[:] = 4
    >>> b[:] = 5
    >>> gidney_adder(a,b)
    >>> print(b)
    {9: 1.0}
    """