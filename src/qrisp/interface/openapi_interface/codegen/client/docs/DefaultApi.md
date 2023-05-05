# openapi_client.DefaultApi

All URIs are relative to *http://localhost:8080*

Method | HTTP request | Description
------------- | ------------- | -------------
[**ping**](DefaultApi.md#ping) | **GET** /ping | 
[**run**](DefaultApi.md#run) | **POST** /run | 


# **ping**
> BackendStatus ping()



Returns the status of a backend

### Example


```python
import time
import openapi_client
from openapi_client.api import default_api
from openapi_client.model.error_message import ErrorMessage
from openapi_client.model.backend_status import BackendStatus
from pprint import pprint
# Defining the host is optional and defaults to http://localhost:8080
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost:8080"
)


# Enter a context with an instance of the API client
with openapi_client.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        api_response = api_instance.ping()
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling DefaultApi->ping: %s\n" % e)
```


### Parameters
This endpoint does not need any parameter.

### Return type

[**BackendStatus**](BackendStatus.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json


### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Backend successfully pinged |  -  |
**0** | Unexpected error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **run**
> {str: (int,)} run()



runs a circuit

### Example


```python
import time
import openapi_client
from openapi_client.api import default_api
from openapi_client.model.inline_object import InlineObject
from openapi_client.model.error_message import ErrorMessage
from pprint import pprint
# Defining the host is optional and defaults to http://localhost:8080
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost:8080"
)


# Enter a context with an instance of the API client
with openapi_client.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)
    inline_object = InlineObject(
        qc=QuantumCircuit(
            qubits=[
                Qubit(
                    identifier="identifier_example",
                ),
            ],
            clbits=[
                Clbit(
                    identifier="identifier_example",
                ),
            ],
            data=[
                Instruction(
                    op=Operation(
                        name="name_example",
                        num_qubits=1,
                        num_clbits=1,
                        params=[
null,
                        ],
                        definition=Operation(),
                    ),
                    qubits=[
                        Qubit(
                            identifier="identifier_example",
                        ),
                    ],
                    clbits=[
                        Clbit(
                            identifier="identifier_example",
                        ),
                    ],
                ),
            ],
        ),
        shots=1,
        token="token_example",
    ) # InlineObject |  (optional)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        api_response = api_instance.run(inline_object=inline_object)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling DefaultApi->run: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **inline_object** | [**InlineObject**](InlineObject.md)|  | [optional]

### Return type

**{str: (int,)}**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json


### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Circuit run successfull |  -  |
**0** | Unexpected error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

