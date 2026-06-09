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

from .block_encoding_base import BlockEncoding
from .constructors import (
    build_from_array,
    build_from_eye,
    build_from_lcu,
    build_from_operator,
    build_from_projector,
)

BlockEncoding.from_array = classmethod(build_from_array)
BlockEncoding.from_eye = classmethod(build_from_eye)
BlockEncoding.from_lcu = classmethod(build_from_lcu)
BlockEncoding.from_operator = classmethod(build_from_operator)
BlockEncoding.from_projector = classmethod(build_from_projector)

__all__ = ["BlockEncoding"]
