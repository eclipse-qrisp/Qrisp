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


import os
import threading

import connexion
import six
from connexion.apps.flask_app import FlaskJSONEncoder
from openapi_server.models.backend_status import BackendStatus
from openapi_server.models.base_model_ import Model
from openapi_server.models.inline_object import InlineObject  # noqa: E501

import qrisp.interface.openapi_interface.codegen.server.openapi_server as openapi_server


class JSONEncoder(FlaskJSONEncoder):
    include_nulls = False

    def default(self, o):
        if isinstance(o, Model):
            dikt = {}
            for attr, _ in six.iteritems(o.openapi_types):
                value = getattr(o, attr)
                if value is None and not self.include_nulls:
                    continue
                attr = o.attribute_map[attr]
                dikt[attr] = value
            return dikt
        return FlaskJSONEncoder.default(self, o)


# Returns the hosts ip
def get_ip():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


# This class describes a Backend Server
class BackendServer:
    # For the constructor we specify a function that runs the circuit, a socket ip
    # adress, a port and the name.
    # Note that the name is supposed to be unique, that is if this python instance
    # already runs a threat with that name, no new server is started
    def __init__(
        self,
        run_func,
        socket_ip_address=None,
        port=None,
        name="generic_quantum_backend_server",
        ping_func=None,
        online_status=True,
    ):
        self.app = connexion.App(
            __name__, specification_dir="./codegen/server/openapi_server/openapi/"
        )
        self.app.app.json_encoder = JSONEncoder

        @self.app.route("/run", methods=["POST"])
        def run():
            inline_object = InlineObject.from_dict(connexion.request.get_json())
            qc = inline_object.qc
            shots = inline_object.shots
            token = inline_object.token

            return run_func(qc, shots, token)

        if ping_func is not None:

            @self.app.route("/ping", methods=["GET"])
            def ping():
                res = ping_func()
                res.online = pass_online_status_bool_by_reference()
                return res.to_dict()

        self.online_status = online_status
        pass_online_status_bool_by_reference = lambda: self.online_status

        self.name = name

        if port is None:
            port = 9010

        if socket_ip_address is None:
            socket_ip_address = get_ip()

        self.socket_ip_address = socket_ip_address
        self.port = port

    # Starts the server
    def start(self):
        # self.app.run(port=self.port)

        # import logging
        # log = logging.getLogger('werkzeug')
        # log.setLevel(logging.ERROR)
        # Create thread
        from waitress import serve

        def wrapper():
            serve(self.app, host=self.socket_ip_address, port=self.port)

        thr = threading.Thread(target=wrapper)
        thr.setDaemon(True)

        # Start the thread
        thr.start()
