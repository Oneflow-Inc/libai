# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class Registry(object):
    def __init__(self, name):
        self._name = name
        self._registry = dict()

    def register(self, module_class):
        module_name = module_class.__name__
        if module_name in self._registry:
            raise KeyError('Cannot register duplicate component ({})'.format(module_name))
        self._registry[module_name] = module_class
        return module_class

    def get(self, key):
        if key not in self._registry:
            raise KeyError('Cannot get unregistered component ({})'.format(key))
        return self._registry[key]
    
    def __getitem__(self, key):
        if key not in self._registry:
            raise KeyError('Cannot get unregistered component ({})'.format(key))
        return self._registry[key]