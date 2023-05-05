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

# Created by ann81984 at 06.05.2022
# import pytest

from qrisp import QuantumString


def test_hello_world():
    q_str = QuantumString(size=len("hello world"))
    q_str.encode("hello world")

    assert list(q_str.get_measurement().keys())[0] == "hello world"
