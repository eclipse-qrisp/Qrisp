"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

from flask import Flask, request, jsonify, make_response
import yaml
from datetime import date, datetime
import threading

"""

This file sets up a server adhering to the interface specified by the Qunicorn middleware
developed in the SeQuenC project: https://sequenc.de/

If you are a backend provider, we don't recommend using this server for production,
since it only exposes the necessities to run quantum circuits and nothing else.
Instead check out the Qunicorn GitHub: https://github.com/qunicorn/qunicorn-core
And it's documentation on how to create new Pilots:
https://qunicorn-core.readthedocs.io/en/latest/tutorials/pilot_tutorial_demo.html

"""


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


class BackendServer:
    """
    This class allows convenient setup of a server respecting the `Qunicorn <https://qunicorn-core.readthedocs.io/en/latest/index.html>`_ interface.

    Parameters
    ----------
    run_func : function
        A function that receives a QuantumCircuit, an integer specifying the amount of
        shots and a token in the form of a string. It returns the counts as a dictionary
        of bitstrings.
    ip_address : str, optional
        The IP address of where the listening socket should be opened. By default, the
        IP address of the hosting machine will be used.
    port : int, optional
        The port on which to listen for requests. By default, 9010 will be used.
    is_simulator : bool, optional
        A bool specifying whether the results returned by this server are due to a
        simulator. The default is True.

    Examples
    --------

    We create a server listening on the localhost IP address using a run function which
    prints the token and queries the Aer-simulator. ::

        def run_func(qasm_str, shots):

            from qiskit import QuantumCircuit
            #Convert to qiskit

            qiskit_qc = QuantumCircuit.from_qasm_str(qasm_str)

            from qiskit_aer import AerSimulator
            qiskit_backend = AerSimulator()

            #Run Circuit on the Qiskit backend
            return qiskit_backend.run(qiskit_qc, shots = shots).result().get_counts()

        from qrisp.interface import BackendServer
        example_server = BackendServer(run_func, ip_address = "127.0.0.1", port = 8080)
        example_server.start()

    """

    def __init__(self, run_func, ip_address=None, port=None):

        self.app = Flask(__name__)
        self.deployments = []
        if ip_address is None:
            ip_address = get_ip()
        if port is None:
            port = 9010
        self.port = port
        self.ip_address = ip_address
        self.jobs = []
        self.run_func = run_func
        self.is_simulator = True
        # To set up the server we route the relevant functions for the flask app.

        # This function receives the deployment data and saves it in the
        # deployment attribute
        @self.app.route("/deployments", methods=["POST"])
        def deployments():

            # Receive the quantum circuit data from the request
            data = request.get_json()

            # Append the deployment data to the library
            self.deployments.append(data)

            # For the response dictionary
            response = {
                "id": len(self.deployments) - 1,
                "deployedBy": {"id": 0, "name": ""},
                "deployedAt": str(date.today()),
                "name": "",
            }

            response["programs"] = data["programs"]

            # Create response object
            response_object = make_response(jsonify(response), 201)

            return response_object

        # This function receives the deployment id of a job and starts the execution
        @self.app.route("/jobs", methods=["POST"])
        def jobs_post():

            # Receive the quantum circuit data from the request
            data = request.get_json()

            # Extract the deployment
            deployment_id = data["deploymentId"]
            deployment = self.deployments[deployment_id]

            # Create the wrapper for asynchronous execution of the job
            def job_wrapper():
                for i in range(len(deployment["programs"])):
                    counts_res = self.run_func(
                        deployment["programs"][i]["quantumCircuit"],
                        data["shots"],
                        data["token"],
                    )
                    job_dic["counts"].append(counts_res)
                # try:
                #     for i in range(len(deployment["programs"])):
                #         counts_res = self.run_func(deployment["programs"][i]["quantumCircuit"],
                #                                     data["shots"],
                #                                     data["token"])
                #         job_dic["counts"].append(counts_res)
                # except Exception as e:
                #     job_dic["exception"] = e

                job_dic["end_time"] = str(datetime.now())

            # Create the thread
            run_thread = threading.Thread(target=job_wrapper)

            # Create the dictionary to hold information about the job
            job_dic = {}

            job_dic["deploymentId"] = deployment_id
            job_dic["counts"] = []
            job_dic["shots"] = data["shots"]
            job_dic["exception"] = None
            job_dic["quantumCircuits"] = deployment["programs"]
            job_dic["run_thread"] = run_thread
            job_dic["start_time"] = str(datetime.now())
            job_dic["end_time"] = "-"

            # Start the thread
            run_thread.start()

            # Append the job to jobs attribute
            self.jobs.append(job_dic)

            # Form the response
            response = {"id": len(self.jobs) - 1, "jobName": "", "jobState": "running"}
            response_object = make_response(jsonify(response), 201)

            return response_object

        # This function retrieves information about a certain job. It is used by
        # the client to inquire the results of the job
        @self.app.route("/jobs/<int:job_id>/", methods=["GET"])
        def get_job(job_id):

            # Search for the job with the specified job_id
            if not job_id < len(self.jobs):
                return jsonify({"error": "Job not found"}), 404

            # Get job information
            job = self.jobs[job_id]

            # Find out wether the job finished
            thread = job["run_thread"]

            if not thread.is_alive():
                state = "finished"
            else:
                state = "running"

            # Find out wether the job suceeded
            if job["exception"] is not None:

                state = "failed"
                response = {
                    "code": 0,
                    "status": "failed",
                    "message": str(job["exception"]),
                    "errors": {},
                }

                response_object = make_response(jsonify(response))
                return response_object

            # Form the response
            response = {
                "id": job_id,
                "executedBy": {"id": 0, "name": ""},
                "executedOn": {
                    "id": 0,
                    "numQubits": -1,
                    "isSimulator": self.is_simulator,
                    "isLocal": True,
                    "provider": {
                        "id": 0,
                        "withToken": True,
                        "supportedLanguages": {
                            "id": 0,
                            "providerId": "string",
                            "name": "",
                        },
                        "name": "",
                    },
                },
                "progress": 0,
                "state": state,
                "type": "",
                "startedAt": job["start_time"],
                "finishedAt": job["end_time"],
                "data": "",
                "results": [
                    {
                        "id": 0,
                        "circuit": job["quantumCircuits"][i],
                        "results": job["counts"][i],
                        "resultType": "COUNTS",
                        "metaData": {},
                    }
                    for i in range(len(job["counts"]))
                ],
                "parameters": "",
            }

            response_object = make_response(jsonify(response), 201)

            return response_object

    def start(self):

        from waitress import serve

        # Set up the thread to run the server
        def wrapper():
            serve(self.app, host=self.ip_address, port=self.port)
            # self.app.run(host=self.ip_address, port=self.port)

        thr = threading.Thread(target=wrapper)
        thr.setDaemon(True)

        # Start the thread
        thr.start()

        import requests

        # Hold programm until the server answers
        while True:
            try:
                response = requests.get(
                    "http://" + self.ip_address + ":" + str(self.port) + "/jobs/0/"
                )
            except:
                continue
            break
