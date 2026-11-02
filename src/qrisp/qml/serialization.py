# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Provides save_model and load_model, which store the parts of a model in a file and load them again."""

from importlib.metadata import PackageNotFoundError, version

import dill

__all__ = ["load_model", "save_model"]

# Increase whenever the layout of the saved record changes.
_FORMAT_VERSION = 1


def _qrisp_version():
    try:
        return version("qrisp")
    except PackageNotFoundError:
        return None


def save_model(filename, **parts):
    r"""
    Saves the parts of a model to a file.

    Each part is passed as a keyword argument and stored under that name, which
    is the name :func:`load_model` returns it under. Parts keep their
    configuration, such as the scaling range of an encoding. A class you define
    yourself, such as an :class:`Encoding` subclass written in a notebook, is
    stored together with its code.

    **Preprocessing is not saved**

    Only the parts you pass are saved, which means preprocessing done before
    encoding has to be repeated on new raw data after loading. A loaded model
    encodes features prepared differently without an error. What you need to
    keep depends on the step.

    * Steps derived from the training data, such as :func:`minmax_scale`, need
      the statistics that they were derived from. :func:`minmax_scale` returns the
      scaled arrays but not the range, while an encoder configured with
      ``data`` or ``data_range`` stores the range itself. You can also save the
      training range as a part of its own and scale with it after loading.
    * Steps with a fixed setting, such as the threshold of :func:`binarize`,
      need that setting applied again. Save it as a part if the code that uses
      the model does not fix it.
    * Steps acting on each sample alone, such as :func:`normalize`, store no
      statistics but still have to be applied to every new sample.

    Parameters
    ----------
    filename : str or os.PathLike
        The file to write. An existing file is overwritten.
    **parts
        The parts of the model. Any object that can be pickled is accepted,
        including encodings, ansatzes, operators, and NumPy or Jax arrays.

    Raises
    ------
    ValueError
        If no parts are given.

    Examples
    --------

    ::

        import numpy as np
        from qrisp.operators import Z
        from qrisp.qml import Ansatz, Encoding, save_model

        encoding = Encoding.angle(n_features=2, data_range=([3.0, 1.0], [6.9, 2.5]))
        ansatz = Ansatz.real_amplitudes(num_qubits=2, layers=1)
        parameters = np.array([0.4, 1.1, 2.0, 0.7])

        save_model("model.pkl", encoding=encoding, ansatz=ansatz, observable=Z(0), parameters=parameters)

    """
    if not parts:
        raise ValueError(
            "Tried to save a model with no parts. Pass each part as a keyword argument, for example parameters=w."
        )

    record = {"format_version": _FORMAT_VERSION, "qrisp_version": _qrisp_version(), "parts": parts}

    # Without recurse, dill saves a notebook class without the functions its
    # methods call, and the class fails once it is used after loading.
    with open(filename, "wb") as file:
        dill.dump(record, file, recurse=True)


def load_model(filename):
    r"""
    Loads the parts of a model saved with :func:`save_model`.

    .. warning::

        Loading a file runs the code stored in it, as with any pickle-based
        format. Only load files from a source you trust.

    Parameters
    ----------
    filename : str or os.PathLike
        The file written by :func:`save_model`.

    Returns
    -------
    dict
        The saved parts, by the names they were saved under.

    Raises
    ------
    ValueError
        If the file was not written by :func:`save_model`, or uses a newer
        file format than this version of Qrisp reads.

    Examples
    --------

    We save a small model and load it again, similar to a new session. The
    parts come back by name, and the encoding still holds its scaling range, which
    means it takes measurements in their original units.

    ::

        import numpy as np
        from qrisp import QuantumFloat
        from qrisp.operators import Z
        from qrisp.qml import Ansatz, Encoding, expectation, load_model, save_model

        save_model(
            "model.pkl",
            encoding=Encoding.angle(n_features=2, data_range=([3.0, 1.0], [6.9, 2.5])),
            ansatz=Ansatz.real_amplitudes(num_qubits=2, layers=1),
            observable=Z(0),
            parameters=np.array([0.4, 1.1, 2.0, 0.7]),
        )

        model = load_model("model.pkl")

        def prepare():
            qf = QuantumFloat(2)
            model["encoding"].apply(qf, [5.1, 1.6])
            model["ansatz"].apply(qf, model["parameters"])
            return qf

        prediction = expectation(prepare, model["observable"])()

    """
    with open(filename, "rb") as file:
        record = dill.load(file)

    if not isinstance(record, dict) or "format_version" not in record:
        raise ValueError(f"Tried to load {filename}, which was not written by save_model.")

    if record["format_version"] > _FORMAT_VERSION:
        raise ValueError(
            f"Tried to load {filename}. It uses model file format {record['format_version']}, written by Qrisp "
            f"{record['qrisp_version']}. This version of Qrisp reads up to format {_FORMAT_VERSION}. Update Qrisp to "
            "load it."
        )

    if not isinstance(record.get("parts"), dict):
        raise ValueError(f"Tried to load {filename}, which was not written by save_model.")

    return dict(record["parts"])
