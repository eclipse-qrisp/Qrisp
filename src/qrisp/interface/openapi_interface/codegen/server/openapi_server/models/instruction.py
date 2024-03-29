# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from openapi_server.models.base_model_ import Model
from openapi_server.models.clbit import Clbit
from openapi_server.models.operation import Operation
from openapi_server.models.qubit import Qubit
from openapi_server import util

from openapi_server.models.clbit import Clbit  # noqa: E501
from openapi_server.models.operation import Operation  # noqa: E501
from openapi_server.models.qubit import Qubit  # noqa: E501

class Instruction(Model):
    """NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).

    Do not edit the class manually.
    """

    def __init__(self, op=None, qubits=None, clbits=None):  # noqa: E501
        """Instruction - a model defined in OpenAPI

        :param op: The op of this Instruction.  # noqa: E501
        :type op: Operation
        :param qubits: The qubits of this Instruction.  # noqa: E501
        :type qubits: List[Qubit]
        :param clbits: The clbits of this Instruction.  # noqa: E501
        :type clbits: List[Clbit]
        """
        self.openapi_types = {
            'op': Operation,
            'qubits': List[Qubit],
            'clbits': List[Clbit]
        }

        self.attribute_map = {
            'op': 'op',
            'qubits': 'qubits',
            'clbits': 'clbits'
        }

        self._op = op
        self._qubits = qubits
        self._clbits = clbits

    @classmethod
    def from_dict(cls, dikt) -> 'Instruction':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The Instruction of this Instruction.  # noqa: E501
        :rtype: Instruction
        """
        return util.deserialize_model(dikt, cls)

    @property
    def op(self):
        """Gets the op of this Instruction.


        :return: The op of this Instruction.
        :rtype: Operation
        """
        return self._op

    @op.setter
    def op(self, op):
        """Sets the op of this Instruction.


        :param op: The op of this Instruction.
        :type op: Operation
        """
        if op is None:
            raise ValueError("Invalid value for `op`, must not be `None`")  # noqa: E501

        self._op = op

    @property
    def qubits(self):
        """Gets the qubits of this Instruction.


        :return: The qubits of this Instruction.
        :rtype: List[Qubit]
        """
        return self._qubits

    @qubits.setter
    def qubits(self, qubits):
        """Sets the qubits of this Instruction.


        :param qubits: The qubits of this Instruction.
        :type qubits: List[Qubit]
        """
        if qubits is None:
            raise ValueError("Invalid value for `qubits`, must not be `None`")  # noqa: E501

        self._qubits = qubits

    @property
    def clbits(self):
        """Gets the clbits of this Instruction.


        :return: The clbits of this Instruction.
        :rtype: List[Clbit]
        """
        return self._clbits

    @clbits.setter
    def clbits(self, clbits):
        """Sets the clbits of this Instruction.


        :param clbits: The clbits of this Instruction.
        :type clbits: List[Clbit]
        """
        if clbits is None:
            raise ValueError("Invalid value for `clbits`, must not be `None`")  # noqa: E501

        self._clbits = clbits
