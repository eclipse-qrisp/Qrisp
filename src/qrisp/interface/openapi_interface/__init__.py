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


path = str(__file__)[:-11]
client_path = path + "/codegen/client"
server_path = path + "/codegen/server"

import sys

sys.path.insert(0, client_path)
sys.path.insert(0, server_path)

from qrisp.interface.openapi_interface.backend_client import BackendClient
from qrisp.interface.openapi_interface.backend_server import BackendServer
from qrisp.interface.openapi_interface.interface_types import *

sys.path.pop(0)
sys.path.pop(0)
