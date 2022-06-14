# coding=utf-8

"""
Unittests followed
https://github.com/facebookresearch/detectron2/blob/main/tests/config/test_instantiate_config.py
"""

from collections import namedtuple
import os
import unittest
import yaml
import tempfile
from libai.config import instantiate, LazyCall
from omegaconf import OmegaConf
from dataclasses import dataclass
from omegaconf import __version__ as oc_version

OC_VERSION = tuple(int(x) for x in oc_version.split(".")[:2])


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "width"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    Attributes:
        channels:
        width:
    """

    def __new__(cls, channels=None, width=None):
        return super().__new__(cls, channels, width)


class TestClass:
    def __init__(self, int_arg, list_arg=None, dict_arg=None, extra_arg=None) -> None:
        self.int_arg = int_arg
        self.list_arg = list_arg
        self.dict_arg = dict_arg
        self.extra_arg = extra_arg

    def __call__(self, call_arg):
        return call_arg + self.int_arg


@dataclass
class TestDataClass:
    x: int
    y: str


@unittest.skipIf(OC_VERSION < (2, 1), "omegaconf version too old")
class TestConstruction(unittest.TestCase):
    def test_basic_construct(self):
        objconf = LazyCall(TestClass)(
            int_arg=3,
            list_arg=[10],
            dict_arg={},
            extra_arg=LazyCall(TestClass)(int_arg=4, list_arg="${..list_arg}"),
        )

        obj = instantiate(objconf)
        self.assertIsInstance(obj, TestClass)
        self.assertEqual(obj.int_arg, 3)
        self.assertEqual(obj.extra_arg.int_arg, 4)
        self.assertEqual(obj.extra_arg.list_arg, obj.list_arg)

        objconf.extra_arg.list_arg = [5]
        obj = instantiate(objconf)
        self.assertIsInstance(obj, TestClass)
        self.assertEqual(obj.extra_arg.list_arg, [5])

    def test_instantiate_other_obj(self):
        # do nothing for other obj
        self.assertEqual(instantiate(5), 5)
        x = [3, 4, 5]
        self.assertEqual(instantiate(x), x)
        x = TestClass(1)
        self.assertIs(instantiate(x), x)
        x = {"xx": "yy"}
        self.assertEqual(instantiate(x), x)

    def test_instantiate_lazy_target(self):
        # _target_ is result of instantiate
        objconf = LazyCall(LazyCall(len)(int_arg=3))(call_arg=4)
        objconf._target_._target_ = TestClass
        self.assertEqual(instantiate(objconf), 7)

    def test_instantiate_lst(self):
        lst = [1, 2, LazyCall(TestClass)(int_arg=1)]
        x = LazyCall(TestClass)(
            int_arg=lst
        )  # list as an argument should be recursively instantiated
        x = instantiate(x).int_arg
        self.assertEqual(x[:2], [1, 2])
        self.assertIsInstance(x[2], TestClass)
        self.assertEqual(x[2].int_arg, 1)

    def test_instantiate_namedtuple(self):
        x = LazyCall(TestClass)(int_arg=ShapeSpec(channels=1, width=3))
        # test serialization
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "lb_test.yaml")
            OmegaConf.save(x, fname)
            with open(fname) as f:
                x = yaml.unsafe_load(f)

        x = instantiate(x)
        self.assertIsInstance(x.int_arg, ShapeSpec)
        self.assertEqual(x.int_arg.channels, 1)

    def test_bad_lazycall(self):
        with self.assertRaises(Exception):
            LazyCall(3)

    def test_instantiate_dataclass(self):
        a = LazyCall(TestDataClass)(x=1, y="s")
        a = instantiate(a)
        self.assertEqual(a.x, 1)
        self.assertEqual(a.y, "s")

    def test_instantiate_no_recursive(self):
        def helper_func(obj):
            self.assertNotIsInstance(obj, TestClass)
            obj = instantiate(obj)
            self.assertIsInstance(obj, TestClass)
            return obj.int_arg

        objconf = LazyCall(helper_func)(obj=LazyCall(TestClass)(int_arg=4))
        self.assertEqual(instantiate(objconf, _recursive_=False), 4)


if __name__ == "__main__":
    unittest.main()
