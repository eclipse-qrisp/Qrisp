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


from qrisp.interface import BackendClient

api_endpoint = "127.0.0.1"

def CirqSim():
    return BackendClient(api_endpoint, port = 8083)

def PennylaneSim():
    return BackendClient(api_endpoint, port = 8084)
    
def MQTSim():
    return BackendClient(api_endpoint, port = 8085)

def PennylaneRigettiSim():
    return BackendClient(api_endpoint, port = 8086)

def PyTketStimSim():
    return BackendClient(api_endpoint, port = 8087)

def QulacsSim():
    return BackendClient(api_endpoint, port = 8088)

def QSimCirq():
    return BackendClient(api_endpoint, port = 8089)
    
def QiboSim():
    return BackendClient(api_endpoint, port = 8090)

    