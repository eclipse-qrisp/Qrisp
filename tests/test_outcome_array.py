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


def outome_array_hashing_test():
    from qrisp import OutcomeArray
    import numpy as np

    test = OutcomeArray(np.eye(3))

    test_dic = {test: 1}

    test_dic[str(np.eye(3))]

    assert str(np.eye(3)) == str(test)
