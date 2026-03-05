"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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
********************************************************************************
"""

import numpy as np

from qrisp.alg_primitives.arithmetic.modular_arithmetic import modinv
from qrisp.algorithms.shor import shors_alg


def rsa_decrypt(e, N, cipher_int, backend=None):
    """
    Decrypts an integer using factorization powered by Shor's algorithm.

    Parameters
    ----------
    e : int
        Public key 1.
    N : int
        Public key 2.
    cipher_int : int
        The integer to decrypt.
    backend : :ref:`BackendClient`, optional
        The backend to execute the quantum algorithm. By default the Qrisp simulator will be used.

    Returns
    -------
    plaintext : int
        The decrypted integer.

    Examples
    --------

    We decrypt the integer 2 using $N = 33$ and $e = 7$

    >>> from qrisp.shor import rsa_decrypt
    >>> rsa_decrypt(7, 33, 2)
    8

    """

    if not backend is None:
        mes_kwargs = {"backend": backend}
    else:
        mes_kwargs = {}

    p = shors_alg(N, mes_kwargs=mes_kwargs)

    # Calculate the other factor
    q = N // p

    # Calculate the totient
    phi = (p - 1) * (q - 1)

    # Calculate the private key
    d = modinv(e, phi)

    # Decrypt the ciphertext
    plaintext = pow(cipher_int, d, N)

    return plaintext


def rsa_encrypt(e, N, message_int):
    """
    Encrypts an integer using a public key pair $(e,N)$.

    Parameters
    ----------
    e : int
        Public key 1.
    N : int
        Public key 2.
    message_int : int
        The integer to encrypt.

    Returns
    -------
    ciphertext : int
        The encrypted integer.

    Examples
    --------

    We encrypt the integer 8 using $N=33$ ($p = 11$, $q = 3$) and $e = 7$

    >>> from qrisp.shor import rsa_encrypt
    >>> rsa_encrypt(e = 7, N = 33, message_int = 8)
    2
    """
    # Convert the message to an integer
    # message_int = int.from_bytes(message.encode(), 'big')

    # Encrypt the message
    ciphertext = pow(message_int, e, N)

    return ciphertext


def rsa_encrypt_string(e, N, message):
    """
    Encrypts an arbitrary Python string using RSA.

    Parameters
    ----------
    e : int
        Public key 1.
    N : int
        Public key 2.
    message : string
        The message to encrypt.

    Returns
    -------
    ciphertext : string
        A bitstring containing the encrypted message.

    Examples
    --------

    We encrypt a string containing an important message

    >>> from qrisp.shor import rsa_encrypt_string
    >>> rsa_encrypt_string(e = 7, N = 65, message = "Qrisp is awesome!")
    '01010000000101001010001100100110010010000101000010001101000010100011010101110011101000100100011100000100000100110111101000011000111110111111'

    """

    message_bitstring = " ".join(
        format(x, "b").zfill(7) for x in bytearray(message, "ascii")
    ).replace(" ", "")

    chunksize = N.bit_length() - 1

    chunks = [
        message_bitstring[i * chunksize : (i + 1) * chunksize][::-1].zfill(chunksize)[
            ::-1
        ]
        for i in range(int(np.ceil(len(message_bitstring) / chunksize)))
    ]

    ciphertext = ""

    for i in range(len(chunks)):

        encrypted_int = rsa_encrypt(e, N, int(chunks[i], 2))
        ciphertext += bin(encrypted_int)[2:].zfill(chunksize + 1)

    return ciphertext


def rsa_decrypt_string(e, N, ciphertext, backend=None):
    """
    Decrypts a bitstring into a human readable string.

    Parameters
    ----------
    e : int
        Public key 1.
    N : int
        Public key 2.
    ciphertext : string
        A bitstring, containing the encrypted message.
    backend : :ref:`BackendClient`, optional
        The backend to execute the quantum algorithm. By default the Qrisp simulator will be used.

    Returns
    -------
    plaintext : string
        The decrypted string.

    Examples
    --------

    We decrypt the message we encrypted in the example of :meth:`rsa_encrypt_string <qrisp.shor.rsa_encrypt_string>`.

    >>> ciphertext = '01010000000101001010001100100110010010000101000010001101000010100011010101110011101000100100011100000100000100110111101000011000111110111111'
    >>> from qrisp.shor import rsa_decrypt_string
    >>> rsa_decrypt_string(e = 7, N = 65, ciphertext = ciphertext)
    'Qrisp is awesome!'

    """

    if not backend is None:
        mes_kwargs = {"backend": backend}
    else:
        mes_kwargs = {}

    p = shors_alg(N, mes_kwargs=mes_kwargs)

    # Calculate the other factor
    q = N // p

    # Calculate the totient
    phi = (p - 1) * (q - 1)

    # Calculate the private key
    d = modinv(e, phi)

    chunksize = (N).bit_length()

    chunks = [
        ciphertext[i * chunksize : (i + 1) * chunksize]
        for i in range(int(np.ceil(len(ciphertext) / chunksize)))
    ]

    plaintext_bitstring = ""
    for i in range(len(chunks)):
        cipher_int = int(chunks[i], 2)
        plaintext_int = pow(cipher_int, d, N)
        plaintext_bitstring += bin(plaintext_int)[2:].zfill(chunksize - 1)

    return bitstring_to_string(plaintext_bitstring)


def bitstring_to_string(bitstring):
    chars = []
    for i in range(0, len(bitstring) // 7 * 7, 7):
        byte = bitstring[i : i + 7] + " "
        chars.append(chr(int(byte, 2)))
    return "".join(chars)
