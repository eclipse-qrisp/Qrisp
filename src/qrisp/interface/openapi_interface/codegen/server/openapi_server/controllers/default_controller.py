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
