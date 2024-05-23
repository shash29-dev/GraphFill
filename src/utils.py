import bisect
import functools
import logging
import numbers
import os
import signal
import sys
import traceback
import warnings

import torch
from pytorch_lightning import seed_everything
from collections import defaultdict
LOGGER = logging.getLogger(__name__)

class DotDict(defaultdict):
    # https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """dot.notation access to dictionary attributes"""
    __getattr__ = defaultdict.get
    __setattr__ = defaultdict.__setitem__
    __delattr__ = defaultdict.__delitem__


def check_and_warn_input_range(tensor, min_value, max_value, name):
    actual_min = tensor.min()
    actual_max = tensor.max()
    if actual_min < min_value or actual_max > max_value:
        warnings.warn(f"{name} must be in {min_value}..{max_value} range, but it ranges {actual_min}..{actual_max}")


def sum_dict_with_prefix(target, cur_dict, prefix, default=0):
    for k, v in cur_dict.items():
        target_key = prefix + k
        target[target_key] = target.get(target_key, default) + v


def average_dicts(dict_list):
    result = {}
    norm = 1e-3
    for dct in dict_list:
        sum_dict_with_prefix(result, dct, '')
        norm += 1
    for k in list(result):
        result[k] /= norm
    return result


def add_prefix_to_keys(dct, prefix):
    return {prefix + k: v for k, v in dct.items()}


def set_requires_grad(module, value):
    for param in module.parameters():
        param.requires_grad = value


def flatten_dict(dct):
    result = {}
    for k, v in dct.items():
        if isinstance(k, tuple):
            k = '_'.join(k)
        if isinstance(v, dict):
            for sub_k, sub_v in flatten_dict(v).items():
                result[f'{k}_{sub_k}'] = sub_v
        else:
            result[k] = v
    return result


class LinearRamp:
    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part


class LadderRamp:
    def __init__(self, start_iters, values):
        self.start_iters = start_iters
        self.values = values
        assert len(values) == len(start_iters) + 1, (len(values), len(start_iters))

    def __call__(self, i):
        segment_i = bisect.bisect_right(self.start_iters, i)
        return self.values[segment_i]


def get_ramp(kind='ladder', **kwargs):
    if kind == 'linear':
        return LinearRamp(**kwargs)
    if kind == 'ladder':
        return LadderRamp(**kwargs)
    raise ValueError(f'Unexpected ramp kind: {kind}')





def get_shape(t):
    if torch.is_tensor(t):
        return tuple(t.shape)
    elif isinstance(t, dict):
        return {n: get_shape(q) for n, q in t.items()}
    elif isinstance(t, (list, tuple)):
        return [get_shape(q) for q in t]
    elif isinstance(t, numbers.Number):
        return type(t)
    else:
        raise ValueError('unexpected type {}'.format(type(t)))

