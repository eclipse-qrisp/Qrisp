"""********************************************************************************
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

import os
from functools import lru_cache as _lru_cache

QRISP_COMPILATION_CACHE_SIZE = int(os.environ.get("QRISP_COMPILATION_CACHE_SIZE", 10000))

_ALL_LRU_CACHES = []


def qrisp_lru_compilation_cache(func=None, *, maxsize=None):
    """Decorator that applies ``functools.lru_cache`` with *maxsize* taken from
    ``QRISP_COMPILATION_CACHE_SIZE`` (overridable via the environment variable
    ``QRISP_COMPILATION_CACHE_SIZE``) and registers the decorated function for
    centralized cache clearing via :func:`clear_all_caches`.

    Can be used as ``@qrisp_lru_compilation_cache`` or ``@qrisp_lru_compilation_cache()``.
    """
    if maxsize is None:
        maxsize = QRISP_COMPILATION_CACHE_SIZE

    def decorator(f):
        cached = _lru_cache(maxsize=maxsize)(f)
        _ALL_LRU_CACHES.append(cached)
        return cached

    if func is None:
        return decorator
    return decorator(func)


def clear_all_caches():
    """Clear all LRU caches registered via :func:`qrisp_lru_compilation_cache`."""
    for cached in _ALL_LRU_CACHES:
        cached.cache_clear()
