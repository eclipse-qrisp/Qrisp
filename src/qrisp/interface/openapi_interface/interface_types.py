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


from qrisp.interface.openapi_interface.codegen.client.openapi_client.models import (
    Clbit as PortableClbit,
)
from qrisp.interface.openapi_interface.codegen.client.openapi_client.models import (
    Instruction as PortableInstruction,
)
from qrisp.interface.openapi_interface.codegen.client.openapi_client.models import (
    Operation as PortableOperation,
)
from qrisp.interface.openapi_interface.codegen.client.openapi_client.models import (
    QuantumCircuit as PortableQuantumCircuit,
)
from qrisp.interface.openapi_interface.codegen.client.openapi_client.models import (
    Qubit as PortableQubit,
)
from qrisp.interface.openapi_interface.codegen.server.openapi_server.models import (
    BackendStatus,
    ConnectivityEdge,
)
