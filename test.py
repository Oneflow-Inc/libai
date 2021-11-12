

class Registry(object):
    def __init__(self, name):
        self._name = name
        self._registry = dict()

    def register(self, module_name, module_class):
        if module_name in self._registry:
            raise KeyError('Cannot register duplicate component ({})'.format(module_name))
        self._registry[module_name] = module_class
        print('register a module')
        return module_class

    def get(self, key):
        if key not in self._registry:
            raise KeyError('Cannot get unregistered component ({})'.format(key))
        return self._registry[key]
    
    def __getitem__(self, key):
        if key not in self._registry:
            raise KeyError('Cannot get unregistered component ({})'.format(key))
        return self._registry[key]

    def keys(self):
        return self._registry.keys()



MODELS = Registry('model')

    
def register_model(name):
    def _register_model(cls):
        return MODELS.register(name, cls)
    return _register_model


@register_model("abc")
class abc:
    def __init__(self):
        print('a')

@register_model("efg")
class efg:
    def __init__(self):
        print('a')

