.. _DockerBackend:

Docker Backend
==============

.. toctree::
   :hidden:

   DockerSims
   Docker configs in this repo
   How to build the image and run the container
   How to execute circuits on the DockerBackend


   Create your own BackendServer

**ASSOCIATED REPO** : `QrispDocker <https://gitlab.fokus.fraunhofer.de/nik40643/qrispdockerbackend>`_ |br|
The repo can also be pulled from `Docker Hub <https://docs.docker.com/engine/reference/commandline/pull/>`_ (Name:  niklassteinmann/qrispdockerbackend:latest) 
   
This module describes the inbuilt Docker container to enable utilization of alternative simulation frameworks.  |br|
It supports:  `Cirq <https://quantumai.google/cirq>`_ (cirq.Simulator), `MQT <https://mqt.readthedocs.io/projects/ddsim/en/latest/>`_ (ddsim qasm_simulator), `Qiskit <https://qiskit.org/>`_ (Aer Backend),  
`PyTket <https://cqcl.github.io/tket/pytket/api/>`_ (AerBackend), `Pennylane <https://pennylane.ai/>`_ (default.qubit simulator), `Rigetti <https://docs.pennylane.ai/projects/rigetti/en/latest/>`_ (numpy wavefunction simulator) and `Qulacs <https://docs.qulacs.org/en/latest/index.html>`_ (sampler) |br|

This functionality is meant to be accessed via a Docker container. The latest **Qrisp** version is required and appropriate setup should be conducted. This setup includes: |br|
Installation of Docker, building of the image provided in this repo, and starting the container associated to the image. Specific steps/ info for utilization in this context is given below. |br|

DockerSims
----------

This module contains different ``run_func``'s for :ref:`BackendServer` instances, which are all initiated simulatiously.
The supported ``run_func``'s and respective ``port``'s are

*	**Cirq** (cirq Simulator) - **port: 8083**
*  **Pennylane** (default.qubit simulator) - **port: 8084**
*	**MQT** (ddsim qasm_simulator) - **port: 8085**
*	**Rigetti** (rigetti np.wavefunction) - **port: 8086**
*	**PyTket** (AerBackend) - **port: 8087**
*	**Qiskit** (Aer Backend) - **port: 8089**
*	**Qulacs** (sampler) - **port: 8088**

The Backends can be accessed via assigning the respective ``port`` to a :ref:`BackendClient` instance:

.. code-block:: pytho

   api_endpoint = "[::1]"
   #port associated to cirq Backend
   port = 8083
   test_client = BackendClient(api_endpoint, port)

   print(qc.get_measurement(backend = test_client))

An examplary circuit the execution of the measurement on all available backends is implemented in the file ```vpn_test_example.py``


Docker configs in this repo
---------------------------

**Dockerfile** : |br|
The image is based on the basic ``python:3.9.16-bullseye`` public image.

.. code-block:: python
   

   FROM python:3.9.16-bullseye
   WORKDIR /app
   COPY . /app/

   #run setup
   #instead of code below do pip install qrisp
   RUN --mount=type=cache,target=/root/.cache \
      pip install pip install qrisp
   # install additional dependencies
   RUN --mount=type=cache,target=/root/.cache \
      pip install -r requirementsdocker.txt

   #Expose Statement determines with ports can be interacted with
   EXPOSE 8083 8084 8085 8086 8087 8089 
   #run bash script to start servers
   CMD ./docker_wrapper.sh


**requirementsdocker.txt** : |br|
Contains additional requirements to run the available backends.


**docker_wrapper.sh** : |br|
Defers to the files that contain the execution code for the BackendServers. This is a bash script that starts a backend server, assigns the app for gunicorn and assigns the port of the app for communication outside of the container. |br|
**Attention!** The file format may cause errors here due to line format conversions. The solution involves multiple simple steps to be conducted **before** building the image: |br|
First, open a console and run the two following comands, which configure Git locally to employ the proper conversions.

.. code-block:: python
   
   git config --global core.autocrlf input
   git config --global core.eol lf

Next, employ a tool like `dos2unix <https://sourceforge.net/projects/dos2unix/>`_ to convert line endings. Execute the following shell comand:

.. code-block:: python

   dos2unix your_file.sh

After these steps you can start building the image and run the container as described below.


How to build the image and run the container
--------------------------------------------

The following steps are specific for a Windows based system. Setup for alternative systems are described on the docker website. 
Download Docker Desktop, further Info can be found in `Get Started - Docker <https://www.docker.com/get-started/>`_. Go through the tutorials if you are not familiar with building an image from a ``Dockerfile``. Before building the image read through the section above which considers the Docker specific configurations. |br|
After Downloading Qrisp, i.e. cloning the repository, you have the Dockerfile for image-building available. Build the image from the ``Windows-Console`` with: |br|

   ``docker build -t <ImageName> <Path\to\your\local\repo>\qrisp`` 

The image building may take several minutes. After it is finished run a container based on your Image <ImageName>: |br|

   ``docker run -it -p  8083:8083 -p 8085:8085 -p 8087:8087 -p 8089:8089 -p 8086:8086 -p 8084:8084 --name <ContainerName> <ImageName>`` |br|

* ``-p`` : publish the ports specified in the ``Dockerfile`` of the :ref:`BackendServers <BackendServer>` to interact outside of the container 
* ``<ImageName>`` : The name of the image as created in the previous step 
* ``<ContainerName>`` : May be chosen freely 

For further information consult the `Docker documentation <https://docs.docker.com/>`_.

How to execute circuits on the DockerBackend
--------------------------------------------

After the previous step has beem conducted and your container is up and running any simulation of a QuantumVariable can be conducted! See the code below for an example of measuring on all available simulators one by one.


::

   from qrisp import QuantumCircuit
   from qrisp.core import QuantumVariable
   from qrisp import x, h, cx
   from qrisp.interface.qunicorn import BackendClient

   api_endpoint = "127.0.0.1"
   #api_endpoint = "[::1]"
   port  = 8083
   port2 = 8085
   port3 = 8087
   port4 = 8086
   port5 = 8089
   port6 = 8084
   port7 = 8088

   qv1 = QuantumVariable(2)
   qv2 = QuantumVariable(10)

   h(qv1[0])
   x(qv2)
   for index in range(3,7):
      cx(qv2[index], qv2[index +1])
      
   print("Cirq Simulator")
   test_client = BackendClient(api_endpoint, port)
   print(qv2.get_measurement(backend = test_client)) 

   print("Pennylane Simulator")
   test_client4 = BackendClient(api_endpoint, port4)
   print(qv2.get_measurement(backend = test_client4))

   print("Qiskit Simulator")
   test_client5 = BackendClient(api_endpoint, port5)
   print(qv2.get_measurement(backend = test_client5))

   print("Pennylane-Rigetti")
   test_client6 = BackendClient(api_endpoint, port6)
   print(qv2.get_measurement(backend = test_client6))

   print("PyTket Simulator")
   test_client3 = BackendClient(api_endpoint, port3)
   print(qv2.get_measurement(backend = test_client3))

   print("Qulacs Simulator")
   test_client7 = BackendClient(api_endpoint, port7)
   print(qv2.get_measurement(backend = test_client7))

   """ 
   unfortunately buggy as of the Qrisp 0.4 release
   print("MQT Simulator")
   test_client2 = BackendClient(api_endpoint, port2)
   print(qv2.get_measurement(backend = test_client2))   """



Create your own BackendServer
-----------------------------

If you want create you own ``BackendServer``, see the ``run_funcs`` as a blueprint


.. |br| raw:: html

      <br>
