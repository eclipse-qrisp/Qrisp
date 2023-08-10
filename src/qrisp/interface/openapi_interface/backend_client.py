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


from qrisp.interface.circuit_converter import convert_circuit
from qrisp.interface.openapi_interface.codegen.client.openapi_client import (
    ApiClient,
    Configuration,
)
from qrisp.interface.openapi_interface.codegen.client.openapi_client.api.default_api \
    import DefaultApi
from qrisp.interface.openapi_interface.codegen.client.openapi_client.models import (
    InlineObject,
)


class BackendClient(DefaultApi):
    def __init__(self, socket_ip, port):
        if socket_ip.find(":") != -1:
            socket_ip = "[" + socket_ip + "]"

        if port is None:
            port = 9010

        # TO-DO Allow API token/Secure connections ...
        # check Configuration class definition
        config = Configuration(host="http://" + socket_ip + ":" + str(port))
        client = ApiClient(configuration=config)

        super().__init__(client)

    def run(self, qc, shots, token=""):
        request_object = InlineObject(
            qc=convert_circuit(qc, "open_api"), shots=shots, token=token
        )
        return super().run(inline_object=request_object)
