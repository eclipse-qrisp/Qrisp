"""
/********************************************************************************
* Copyright (c) 2023 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2 
* or later with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0
********************************************************************************/
"""

import connexion
import six

from openapi_server.models.backend_status import BackendStatus  # noqa: E501
from openapi_server.models.error_message import ErrorMessage  # noqa: E501
from openapi_server.models.inline_object import InlineObject  # noqa: E501
from openapi_server import util


def ping():  # noqa: E501
    """ping

    Returns the status of a backend # noqa: E501


    :rtype: BackendStatus
    """
    return 'do some magic!'


def run(inline_object=None):  # noqa: E501
    """run

    runs a circuit # noqa: E501

    :param inline_object: 
    :type inline_object: dict | bytes

    :rtype: Dict[str, int]
    """
    if connexion.request.is_json:
        inline_object = InlineObject.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'
