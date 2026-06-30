"""********************************************************************************
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

from enum import Enum
from math import log2, ceil
from typing import Self, Union
from qrisp import QuantumVariable
from qrisp import cp

class QuantumEnum(QuantumVariable):
    r"""A quantum meta type for auto encoding python enums in a QuantumVariable
    
    >>> from qrisp import QuantumEnum
    >>> from enum import auto
    >>> 
    >>> class Color(QuantumEnum.OneHot):
    >>>     RED = auto()
    >>>     GREEN = auto()
    >>>     BLUE = auto()
    >>>
    >>> @QuantumEnum.auto(Color)
    >>> class QuantumColor(QuantumEnum):
    >>>     pass
    >>> 
    >>> q_color = QuantumColor()
    >>> q_color[:] = Color.RED

    ``QuantumEnum.OneHot`` encoding uses one qubit per enum value, 
    depending on the usecase ``QuantumEnum.Binary`` encoding might be beneficial,
    only requiring $\lceil\log_2{N}\rceil$ qubits instead of $N$.

    The :meth:`QuantumEnum.auto` decorator attaches encoding-dependent methods
    to concrete subclasses, most notably ``apply_phase_if_eq(self, other, gamma)``.
    This method is available on any class created via ``@QuantumEnum.auto(...)``,
    such as ``QuantumColor`` in the usage examples.
    """
    
    def __init__(self, qs=None, name=None):
        size = 0
        if self.encoding == "Binary":
            size = ceil(log2(len(self.enum.__members__)))
        elif self.encoding == "OneHot":
            size = len(self.enum.__members__)
        super().__init__(size, qs=qs, name=name)

    class Binary(Enum):
        r"""Binary encoding for python enums resulting in values in the range of 0 to n-1

        >>> from qrisp import QuantumEnum
        >>> from enum import auto
        >>> 
        >>> class Color(QuantumEnum.Binary):
        >>>     RED = auto()    # 0
        >>>     GREEN = auto()  # 1
        >>>     BLUE = auto()   # 2
        >>>     PURPLE = auto() # 3
        """
        def _generate_next_value_(name, start, count, last_values):
            return count
    
        @staticmethod
        def auto_implement(enum_cls: Self) -> Self:
            if len(enum_cls.__members__) > 0:
                values = [member.value for member in enum_cls]
    
                values.sort()
    
                n = len(enum_cls.__members__.items())
                if values != list(range(n)):
                    raise ValueError(f"{enum_cls} Enum values must be unique and consecutive integers from 0 to {n-1}.")
    
            def decorator(cls):
                cls.enum = enum_cls
                cls.bitlength = ceil(log2(len(enum_cls.__members__)))
    
                cls.encoding = "Binary"
    
                def encoder(self, value):
                    if (type(value) != enum_cls):
                        raise ValueError(f"Can only encode values of type {enum_cls}")
                    return value.value
                cls.encoder = encoder
    
                def decoder(self, i):
                    if i in range(len(enum_cls.__members__.items())):
                        return enum_cls(i)
                    else:
                        raise ValueError("Can not decode value outside of range")
                cls.decoder = decoder
    
                def apply_phase_if_eq(self, other: Self, gamma):
                    cx(self, other)
                    mcp(2*gamma, other, ctrl_state=0)
                    cx(self, other)
                cls.apply_phase_if_eq = apply_phase_if_eq
    
                return cls
            return decorator
    
    class OneHot(Enum):
        r"""One hot encoding for python enums resulting in values as powers of 2

        >>> from qrisp import QuantumEnum
        >>> from enum import auto
        >>> 
        >>> class Color(QuantumEnum.OneHot):
        >>>     RED = auto()    # 1
        >>>     GREEN = auto()  # 2
        >>>     BLUE = auto()   # 4
        >>>     PURPLE = auto() # 8
        """
        def _generate_next_value_(name, start, count, last_values):
            return 1 << count
        
        @staticmethod
        def auto_implement(enum_cls: Self) -> Self:
            if len(enum_cls.__members__) > 0:
                values = [member.value for member in enum_cls]  
    
                values.sort()   
    
                n = [2**x for x in range(len(enum_cls.__members__.items()))]
                if values != n:
                    raise ValueError(f"{enum_cls} Enum values must be unique and follow the one hot encoding.") 
    
            def decorator(cls):
                cls.enum = enum_cls
                cls.bitlength = len(enum_cls.__members__)   
    
                cls.encoding = "OneHot"
    
                def encoder(self, value):
                    if (type(value) != enum_cls):
                        raise ValueError(f"Can only encode values of type {enum_cls}")
                    return value.value
                cls.encoder = encoder   
    
                def decoder(self, i):
                    if i in [member.value for member in enum_cls]:
                        return enum_cls(i)
                    else:
                        raise ValueError("Can not decode value")
                cls.decoder = decoder   
    
                def apply_phase_if_eq(self, other: Self, gamma):
                    for i in range(self.size):
                        cp(2 * gamma, self[i], other[i])
                cls.apply_phase_if_eq = apply_phase_if_eq
                
                return cls
            return decorator

    def auto(enum_cls: Union[Binary, OneHot]) -> Self:
        return enum_cls.auto_implement(enum_cls)
