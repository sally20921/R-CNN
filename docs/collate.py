"""
collate samples fetched from dataset into Tensor(s).
"""

import torch
import re
import collections
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

def default_convert(data):
    '''
    converts each NumPy array data field into a tensor
    '''
    elem_type = type(data)
    
    # if data is already a torch.Tensor
    if isinstance(data, torch.Tensor):
        return data
    
    # if elem_type is numpy 
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        # if array of string classes and object
        if elem_type.__name__ == 'ndarray' and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    
    elif isinstance(data, collections.abc.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'): # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return [default_convert(d) for d in data]
    else:
        return data

default_collate_err_message_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")

def default_collate(batch):
    '''
    put each data field into a tensor with outer dimension batch size
    '''
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        # if we're in a background process, concatenate directly into a shared memory tensor to avoid extra copy
        if torch.utils.data.get_worker_info() is not None:

            # torch.numel(input) -> int
            # returns the total number of elements in the input tensor
            numel = sum(x.numel() for x in batch)
            
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)

            # torch.stack(tensors, dim=0, *, out=None) -> Tensor
            # concatenates a sequence of tensors along a new dimension
            # all tensors need to be of the same size




