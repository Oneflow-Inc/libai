# Copyright (c) Facebook, Inc. and its affiliates.
from itertools import count

from libai.config import LazyCall

from .dir1.dir1_a import dir1a_dict, dir1a_str

dir1a_dict.a = "modified"

# modification above won't affect future imports
from .dir1.dir1_b import dir1b_dict, dir1b_str


lazyobj = LazyCall(count)(x=dir1a_str, y=dir1b_str)
