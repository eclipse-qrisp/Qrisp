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


from qrisp.core import QuantumVariable


class QuantumChar(QuantumVariable):
    r"""
    A QuantumVariable which represents characters. By default, the QuantumChar is
    initialized in NISQ mode, meaning that instead of 256 characters it can only hold
    32, saving almost 40% in qubit cost.

    >>> from qrisp import QuantumChar
    >>> q_ch = QuantumChar(nisq_char = True)

    The chars which can be represented in ``nisq mode`` are

    +---+---+---+---+---+---+---+---+---+---+----+----+----+----+----+----+----+
    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
    +---+---+---+---+---+---+---+---+---+---+----+----+----+----+----+----+----+
    | a | b | c | d | e | f | g | h | i | j | k  | l  | m  | n  | o  | p  | q  |
    +---+---+---+---+---+---+---+---+---+---+----+----+----+----+----+----+----+

    +----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
    | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |
    +----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
    | r  | s  | t  | u  | v  | w  | x  | y  | z  |    | .  | !  | ?  | :  | ,  |
    +----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+

    If ``nisq_mode`` is set to False, the encoder uses the Python-inbuild chr function.
    """

    def decoder(self, i):
        if self.nisq_char:
            return "abcdefghijklmnopqrstuvwxyz .!?:,"[i]
        else:
            return chr(i)

    def __init__(self, qs=None, name=None, nisq_char=True):
        self.nisq_char = nisq_char

        if nisq_char:
            super().__init__(5, qs=qs, name=name)
        else:
            super().__init__(8, qs=qs, name=name)
