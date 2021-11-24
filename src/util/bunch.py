class Bunch(dict):
    """Container object exposing keys as attributes.
    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.
    Examples
    --------
    # >>> from utils import Bunch
    # >>> b = Bunch(a=1, b=2)
    # >>> b['b']
    # 2
    # >>> b.b
    # 2
    # >>> b.a = 3
    # >>> b['a']
    # 3
    # >>> b.c = 6
    # >>> b['c']
    # 6
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        if type(value) is dict:
            self[key] = Bunch(**value) # Made this mofo recursive :-)
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            if type(self[key]) is dict:
                return Bunch(**self[key])
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def to_dict(self):
        return dict(**self)

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass
