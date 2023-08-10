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
