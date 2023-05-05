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

# coding: utf-8

from __future__ import absolute_import

import unittest

from flask import json
from openapi_server.models.backend_status import BackendStatus  # noqa: E501
from openapi_server.models.error_message import ErrorMessage  # noqa: E501
from openapi_server.models.inline_object import InlineObject  # noqa: E501
from openapi_server.test import BaseTestCase
from six import BytesIO


class TestDefaultController(BaseTestCase):
    """DefaultController integration test stubs"""

    def test_ping(self):
        """Test case for ping"""
        headers = {
            "Accept": "application/json",
        }
        response = self.client.open("/ping", method="GET", headers=headers)
        self.assert200(response, "Response body is : " + response.data.decode("utf-8"))

    def test_run(self):
        """Test case for run"""
        inline_object = openapi_server.InlineObject()
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        response = self.client.open(
            "/run",
            method="POST",
            headers=headers,
            data=json.dumps(inline_object),
            content_type="application/json",
        )
        self.assert200(response, "Response body is : " + response.data.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
