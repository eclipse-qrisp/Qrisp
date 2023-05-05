"""
/*********************************************************************
* Copyright (c) 2023 the Qrisp Authors
*
* This program and the accompanying materials are made
* available under the terms of the Eclipse Public License 2.0
* which is available at https://www.eclipse.org/legal/epl-2.0/
*
* SPDX-License-Identifier: EPL-2.0
**********************************************************************/
"""


interface = "thrift"


if interface == "thrift":
    from qrisp.interface.thrift_interface import *
else:
    from qrisp.interface.openapi_interface import *

from qrisp.interface.backends import *
from qrisp.interface.circuit_converter import *
