# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from copy import deepcopy

# copying LRU cache decorator a la:
# https://stackoverflow.com/questions/54909357/how-to-get-functools-lru-cache-to-return-new-instances
def lru_cache(maxsize=20, typed=False, copy=False):
    def decorator(f):
        cached_func = functools.lru_cache(maxsize=maxsize, typed=typed)(f)
        def wrapper(*args, **kwargs):
            res = cached_func(*args, **kwargs)
            if copy:
                return deepcopy(res)
            else:
                return res
                
        return wrapper
    return decorator
