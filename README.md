<p align="center" width="100%"><img src="https://raw.githubusercontent.com/eclipse-qrisp/Qrisp/main/logo/logo_with_contour.png" width=30%></p>

</h1><br>
<div align="center">

[![License](https://img.shields.io/badge/License-EPL_2.0-brightgreen.svg)](https://opensource.org/licenses/EPL-2.0)
![PyPI - Version](https://img.shields.io/pypi/v/qrisp?color=brightgreen)
[![Slack](https://img.shields.io/badge/Slack-4A154B?style=flat&logo=slack&logoColor=white)](https://join.slack.com/t/qrisp-workspace/shared_invite/zt-20yv9bbvo-igspbQpslCBK9ZlYSVijsw)
[![Pytest](https://github.com/eclipse-qrisp/Qrisp/actions/workflows/qrisp_test.yml/badge.svg)](https://github.com/eclipse-qrisp/Qrisp/actions/workflows/qrisp_test.yml)
[![Downloads](https://img.shields.io/pypi/dm/qrisp.svg)](https://pypi.org/project/qrisp/)
[![CodeFactor](https://www.codefactor.io/repository/github/eclipse-qrisp/qrisp/badge/main)](https://www.codefactor.io/repository/github/eclipse-qrisp/qrisp/overview/main)

[![Paper](https://img.shields.io/badge/DOI-10.1038%2Fs41586--020--2649--2-brightgreen)](https://doi.org/10.48550/arXiv.2406.14792)
[![Forks](https://img.shields.io/github/forks/eclipse-qrisp/Qrisp.svg)](https://github.com/eclipse-qrisp/Qrisp/network/members)
[![Open Issues](https://img.shields.io/github/issues/eclipse-qrisp/Qrisp.svg)](https://github.com/eclipse-qrisp/Qrisp/issues)
[![Stars](https://img.shields.io/github/stars/eclipse-qrisp/Qrisp.svg)](https://github.com/eclipse-qrisp/Qrisp/stargazers)
[![Contributors](https://img.shields.io/github/contributors/eclipse-qrisp/Qrisp.svg)](https://github.com/eclipse-qrisp/Qrisp/graphs/contributors)

</div>

## About

Qrisp is a high-level quantum programming framework that allows for intuitive development of quantum algorithms. It provides a rich set of tools and abstractions to make quantum computing more accessible to developers and researchers. By automating many steps one usually encounters when programming a quantum computer, introducing quantum types, and many more features Qrisp makes quantum programming more user-friendly yet stays performant when it comes to compiling programs to the circuit level.

## Features

- Intuitive quantum program design
- High-level quantum programming
- Efficient quantum algorithm implementation
- Extensive documentation and examples

## Installation

You can install Qrisp using pip:

```bash
pip install qrisp
```

Qrisp has been confirmed to work with Python version 3.11 & 3.12.

Qrisp is compatible with any QASM-capable quantum backend! In particular, it offers convenient interfaces for using IBM, IQM and AQT quantum computers, and any quantum backend provider is invited to reach out for a tight integration! 

If you want to work with IQM quantum computers as a backend, you need to install additional dependencies using
```bash
pip install qrisp[iqm]
```

## Documentation
The full documentation, alongside with many tutorials and examples, is available under [Qrisp Documentation](https://www.qrisp.eu/).

## Shor's Algorithm with Qrisp

Shor's algorithm is among the most famous quantum algorithm since it provides a provably exponential speed-up for a practically relevant problem: Facotrizing integers. This is an important application because much of modern cryptography is based on RSA, which heavily relies on integer factorization being insurmountable.

Despite this importance, the amount of software that is actually able to compile the algorithm to the circuit level is extremely limited. This is because a key operation within the algorithm (modular in-place multiplication) is difficult to implement and has strong requirements for the underlying compiler. These problems highlight how the Qrisp programming-model delivers significant advantages to quantum programmers because the quantum part of the algorithm can be expressend within a few lines of code:

```python

from qrisp import QuantumFloat, QuantumModulus, h, QFT, control

def find_order(a, N):
    qg = QuantumModulus(N)
    qg[:] = 1
    qpe_res = QuantumFloat(2*qg.size + 1, exponent = -(2*qg.size + 1))
    h(qpe_res)
    for i in range(len(qpe_res)):
        with control(qpe_res[i]):
            qg *= a
            a = (a*a)%N
    QFT(qpe_res, inv = True)
    return qpe_res.get_measurement()
```

To find out how this can be used to break encryption be sure to check the [tutorial](https://qrisp.eu/general/tutorial/Shor.html).

Qrisp offers much more than just factoring! More examples, like simulating molecules at the quantum level or how to solve the Travelling Salesman Problem, can be found [here](https://qrisp.eu/general/tutorial/index.html).

## Authors and Citation
Qrisp was mainly devised and implemented by Raphael Seidel, supported by Sebastian Bock, Nikolay Tcholtchev, René Zander, Niklas Steinmann and Matic Petric.

If you have comments, questions or love letters, feel free to reach out to us:

raphael.seidel [at] meetiqm.com

sebastian.bock [at] fokus.fraunhofer.de

nikolay.tcholtchev [at] fokus.fraunhofer.de

rene.zander [at] fokus.fraunhofer.de

matic.petric [at] fokus.fraunhofer.de

If you want to cite Qrisp in your work, please use:

```
@misc{seidel2024qrisp,
      title={Qrisp: A Framework for Compilable High-Level Programming of Gate-Based Quantum Computers}, 
      author={Raphael Seidel and Sebastian Bock and René Zander and Matic Petrič and Niklas Steinmann and Nikolay Tcholtchev and Manfred Hauswirth},
      year={2024},
      eprint={2406.14792},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2406.14792}, 
}
```


## License
[Eclipse Public License 2.0](https://github.com/fraunhoferfokus/Qrisp/blob/main/LICENSE)

